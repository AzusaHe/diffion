"""The total-loss PINNs of pinn/jax_pinn.py in PyTorch, so they run on the GPU (CUDA), many fits in one batch.

    python -m pinn.torch_pinn --build cv --starts fno_ttt,center --materials 0:8 --out results/pinn_gpu
    python -m pinn.torch_pinn --build c  --starts fno_ttt,center --materials 0:8 --out results/pinn_gpu

Same model, networks, losses, weights, optimizer and defaults as pinn/jax_pinn.py (see its docstring):
    L = L_data + w_pde L_pde + w_vol L_vol + w_bc L_bc   (all terms from the first step, no staging)
    Build 1 'cv': c and v networks; penalized conservation, volume constraint, wall fluxes.
    Build 2 'c' : c network, J by integration of c_t, v = (J - q) / c; penalized volume constraint, walls.
    one Adam over networks and phi (lr 1e-3, cosine decay to 1e-5, phi at phi_lr_mult x), 5000 steps.
The material (D alpha, t+, V) is pinn/material.py, the torch copy of solver/properties.py.
Every (start, material) pair is one independent fit; all of them are stacked along a leading batch axis and
trained together with torch.func.vmap (the loss is the sum of the per-fit losses and Adam is elementwise, so the
fits do not interact).  Results go to <out>/<A_|B_><start>/<build>/m<k>.json in the format of jax_pinn.py, so
pinn.jax_pinn.summarize and the sweep table read them.  Random draws differ from the JAX runs (other RNG).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.func import grad, grad_and_value, vmap

from pinn.material import material
from solver.dataset import TAU

N_BINS = 20
TARGETS_REL = (1e-2, 1e-3, 1e-4)
GL3 = (np.array([-np.sqrt(0.6), 0.0, np.sqrt(0.6)]), np.array([5.0, 8.0, 5.0]) / 9.0)
GL4 = np.polynomial.legendre.leggauss(4)


# ------------------------------------------------------------------------------------------------ networks
def init_mlp(gen, sizes, dtype):
    return [(torch.randn(a, b, generator=gen, dtype=dtype) * math.sqrt(1.0 / a), torch.zeros(b, dtype=dtype))
            for a, b in zip(sizes[:-1], sizes[1:])]


def mlp(params, z):
    for W, b in params[:-1]:
        z = torch.tanh(z @ W + b)
    W, b = params[-1]
    return (z @ W + b)[0]


def _sqrt_time(t, tmax):
    pos = t > 0
    return torch.where(pos, torch.sqrt(torch.where(pos, t, torch.ones_like(t)) / tmax), torch.zeros_like(t))


def c_point(p, x, t, io, tmax):
    s = _sqrt_time(t, tmax)
    return 1.0 + s * mlp(p["c"], torch.stack([2 * x - 1, 2 * s - 1, io / 5.0]))


def v_point(p, x, t, io, tmax, vscale):
    s = _sqrt_time(t, tmax)
    return vscale * x * mlp(p["v"], torch.stack([2 * x - 1, 2 * s - 1, io / 5.0]))


def q_point(p, th, x, t, io, tmax):
    cx, c = grad_and_value(c_point, argnums=1)(p, x, t, io, tmax)
    dal, tp, _ = material(c, th)
    return -dal * cx + tp * io


def fields(p, th, x, t, io, tmax):
    """c, c_x, c_t, q, q_x, V at one point."""
    (cx, ct), c = grad_and_value(c_point, argnums=(1, 2))(p, x, t, io, tmax)
    qx, q = grad_and_value(q_point, argnums=2)(p, th, x, t, io, tmax)
    V = material(c, th)[2]
    return c, cx, ct, q, qx, V


# ------------------------------------------------------------------------------------------------ quadrature
def bin_points(dtype, device):
    xg, wg = GL4
    edges = np.linspace(0, 1, N_BINS + 1)
    x = (0.5 * (edges[:-1, None] + edges[1:, None]) + 0.5 * np.diff(edges)[:, None] * xg[None]).ravel()
    return torch.tensor(x, dtype=dtype, device=device), torch.tensor(np.tile(wg / 2.0, N_BINS), dtype=dtype, device=device)


def cell_nodes(n_cells, dtype, device):
    xg, wg = GL3
    edges = np.linspace(0, 1, n_cells + 1)
    h = np.diff(edges)
    xq = (0.5 * (edges[:-1, None] + edges[1:, None]) + 0.5 * h[:, None] * xg[None]).ravel()
    wq = (0.5 * h[:, None] * wg[None]).ravel()
    return [torch.tensor(a, dtype=dtype, device=device) for a in (edges, xq, wq)]


# ------------------------------------------------------------------------------------------------ problem
def problem_stats(y, iotas, res_window="observed"):
    """Per-material constants, as in jax_pinn.make_problem."""
    from surrogate.data import velocity_from_c
    mask = np.isfinite(y[0, :, 0])
    K = int(mask.sum())
    tmax = float(TAU[K - 1])
    vscale = float(np.nanmax(np.abs(velocity_from_c(y[None], np.asarray(iotas)[None])[0])))
    ct = np.diff(y[:, :K], axis=1) / np.diff(TAU[:K])[None, :, None]
    st = float(np.sqrt(np.mean(ct ** 2)))
    s_lo = 0.05 if res_window == "early" else float(np.sqrt(TAU[0] / tmax))
    return dict(y=np.nan_to_num(y), mask=mask, iotas=np.asarray(iotas, float), tmax=tmax, vscale=vscale, st=st,
                sigma=1e-3 * float(np.nanmean(y)), mean_c=float(np.nanmean(y)), s_lo=s_lo)


class Losses:
    """Per-fit loss functions (one material, unbatched) with the shared constants bound in."""

    def __init__(self, build, dtype, device, n_res=1024, n_times=32, n_cells=32, w=None):
        self.build, self.n_res, self.n_times, self.n_cells = build, n_res, n_times, n_cells
        self.w = dict(pde=100.0, vol=100.0, bc=100.0) | (w or {})
        self.XB, self.WB = bin_points(dtype, device)
        self.edges, self.xq, self.wq = cell_nodes(n_cells, dtype, device)
        self.t_obs = torch.tensor(TAU, dtype=dtype, device=device)

    def bin_average(self, p, t, io, tmax):
        c = vmap(lambda x: c_point(p, x, t, io, tmax))(self.XB)
        return (c * self.WB).reshape(N_BINS, 4).sum(1)

    def data_loss(self, p, S):
        pred = vmap(lambda io: vmap(lambda t: self.bin_average(p, t, io, S["tmax"]))(self.t_obs))(S["iotas"])
        r = torch.where(S["mask"][None, :, None], pred - S["y"], torch.zeros_like(pred))
        return (r ** 2).sum() / (S["mask"].sum() * 3 * N_BINS)

    def loss_cv(self, p, S, x, t):
        th = torch.sigmoid(p["phi"])
        tm, vs = S["tmax"], S["vscale"]

        def J_point(x, t, io):
            return q_point(p, th, x, t, io, tm) + c_point(p, x, t, io, tm) * v_point(p, x, t, io, tm, vs)

        def res(x, t, io):
            c, cx, ct, q, qx, V = fields(p, th, x, t, io, tm)
            Jx = grad(J_point, argnums=0)(x, t, io)
            vx = grad(v_point, argnums=1)(p, x, t, io, tm, vs)
            return ct + Jx, vx + V * qx

        rc, rv = vmap(lambda xs, ts, io: vmap(lambda a, b: res(a, b, io))(xs, ts))(x, t, S["iotas"])
        tb = t[:, :256]
        wall = vmap(lambda ts, io: vmap(lambda b: torch.stack([J_point(torch.zeros_like(b), b, io) - io,
                                                               J_point(torch.ones_like(b), b, io) - io]))(ts))(tb, S["iotas"])
        L = dict(data=self.data_loss(p, S) / S["sigma"] ** 2, pde=(rc ** 2).mean() / S["st"] ** 2,
                 vol=(rv ** 2).mean() / (vs * S["st"]) ** 2, bc=(wall ** 2).mean() / (S["iotas"] ** 2).mean())
        w = self.w
        return L["data"] + w["pde"] * L["pde"] + w["vol"] * L["vol"] + w["bc"] * L["bc"], L

    def build2_line(self, p, th, t, io, S):
        """Along x at one time: r_v at the cell edges, J(1) - iota, q(0) - iota."""
        tm = S["tmax"]
        ct_q = vmap(lambda x: grad(c_point, argnums=2)(p, x, t, io, tm))(self.xq)
        integ = torch.cat([torch.zeros_like(ct_q[:1]), torch.cumsum((ct_q * self.wq).reshape(self.n_cells, 3).sum(1), 0)])
        J = io - integ
        c, cx, ct, q, qx, V = vmap(lambda x: fields(p, th, x, t, io, tm))(self.edges)
        v = (J - q) / c
        vx = (-ct - qx - v * cx) / c
        return vx + V * qx, J[-1] - io, q[0] - io

    def loss_c(self, p, S, x, t):
        th = torch.sigmoid(p["phi"])
        rv, j1, q0 = vmap(lambda ts, io: vmap(lambda b: self.build2_line(p, th, b, io, S))(ts))(t, S["iotas"])
        L = dict(data=self.data_loss(p, S) / S["sigma"] ** 2, pde=torch.zeros_like(rv[0, 0, 0]),
                 vol=(rv ** 2).mean() / (S["vscale"] * S["st"]) ** 2,
                 bc=((j1 ** 2).mean() + (q0 ** 2).mean()) / (S["iotas"] ** 2).mean())
        w = self.w
        return L["data"] + w["vol"] * L["vol"] + w["bc"] * L["bc"], L

    def loss(self, p, S, x, t):
        return (self.loss_cv if self.build == "cv" else self.loss_c)(p, S, x, t)


# ------------------------------------------------------------------------------------------------ training
def _stack(trees):
    if isinstance(trees[0], dict):
        return {k: _stack([t[k] for t in trees]) for k in trees[0]}
    if isinstance(trees[0], (list, tuple)):
        return type(trees[0])(_stack(list(z)) for z in zip(*trees))
    return torch.stack(trees)


def _leaves(tree):
    if isinstance(tree, dict):
        return [x for k in tree for x in _leaves(tree[k])]
    if isinstance(tree, (list, tuple)):
        return [x for z in tree for x in _leaves(z)]
    return [tree]


def _rebuild(tree, it):
    if isinstance(tree, dict):
        return {k: _rebuild(tree[k], it) for k in tree}
    if isinstance(tree, (list, tuple)):
        return type(tree)(_rebuild(z, it) for z in tree)
    return next(it)


def train_batch(ys, iotas, theta0s, build, seeds, steps=5000, lr=1e-3, phi_lr_mult=0.1, w=None, log_every=100,
                max_wall_s=float("inf"), res_window="observed", width=64, depth=4, device="cuda",
                dtype=torch.float64, rng_seed=0, n_res=1024, n_times=32, n_cells=32, progress=None):
    """Train len(ys) independent PINNs together. Returns one result dict per fit (format of jax_pinn.train)."""
    B = len(ys)
    LS = Losses(build, dtype, device, n_res, n_times, n_cells, w)
    st = [problem_stats(y, io, res_window) for y, io in zip(ys, iotas)]
    S = {k: torch.tensor(np.stack([s[k] for s in st]), device=device, dtype=torch.bool if k == "mask" else dtype)
         for k in ("y", "mask", "iotas", "tmax", "vscale", "st", "sigma", "s_lo")}
    sizes = [3] + [width] * depth + [1]
    P = []
    for th0, sd in zip(theta0s, seeds):
        g = torch.Generator().manual_seed(int(sd))
        p = {"c": init_mlp(g, sizes, dtype)}
        if build == "cv":
            p["v"] = init_mlp(g, sizes, dtype)
        p["phi"] = torch.tensor(np.log(th0 / (1 - th0)), dtype=dtype)     # last leaf
        P.append(p)
    stacked = _stack(P)
    params = _rebuild(stacked, iter([x.to(device) for x in _leaves(stacked)]))
    is_phi = [False] * (len(_leaves(params)) - 1) + [True]
    m =[torch.zeros_like(x) for x in _leaves(params)]
    v = [torch.zeros_like(x) for x in _leaves(params)]
    b1, b2, eps = 0.9, 0.999, 1e-8
    gen = torch.Generator(device=device).manual_seed(rng_seed)

    def total(p, x, t):
        L, parts = vmap(LS.loss)(p, S, x, t)
        return L.sum(), (L, parts)

    step_fn = grad_and_value(total, has_aux=True)
    data_rmse = vmap(lambda p, S: torch.sqrt(LS.data_loss(p, S)))
    mean_c = np.array([s["mean_c"] for s in st])

    def draw():
        if build == "cv":
            x = torch.rand(B, 3, n_res, generator=gen, device=device, dtype=dtype)
            u = torch.rand(B, 3, n_res, generator=gen, device=device, dtype=dtype)
        else:
            x = torch.zeros(B, 3, 1, device=device, dtype=dtype)
            u = torch.rand(B, 3, n_times, generator=gen, device=device, dtype=dtype)
        lo = S["s_lo"][:, None, None]
        return x, S["tmax"][:, None, None] * (lo + (1 - lo) * u) ** 2

    logs = [[] for _ in range(B)]
    hit = [{str(r): [None, None] for r in TARGETS_REL} for _ in range(B)]
    t0 = time.perf_counter()
    for i in range(steps + 1):
        if i % log_every == 0 or i == steps:
            with torch.no_grad():
                rm = data_rmse(params, S).cpu().numpy()
                th = torch.sigmoid(params["phi"]).cpu().numpy()
            wall = time.perf_counter() - t0
            for b in range(B):
                logs[b].append(dict(step=i, wall_s=wall, rel_rmse=float(rm[b] / mean_c[b]), theta=th[b].tolist()))
                for r in TARGETS_REL:
                    if hit[b][str(r)][0] is None and rm[b] <= r * mean_c[b]:
                        hit[b][str(r)] = [i, wall]
            if progress:
                progress(i, wall, rm / mean_c)
            if wall > max_wall_s:
                break
        if i == steps:
            break
        x, t = draw()
        g, (_, (L, parts)) = step_fn(params, x, t)
        sched = lr * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(i, steps) / steps)))
        with torch.no_grad():
            leaves, gl = _leaves(params), _leaves(g)
            c1, c2 = 1 - b1 ** (i + 1), 1 - b2 ** (i + 1)
            new = []
            for j, (pj, gj) in enumerate(zip(leaves, gl)):
                m[j].mul_(b1).add_(gj, alpha=1 - b1)
                v[j].mul_(b2).addcmul_(gj, gj, value=1 - b2)
                step_lr = sched * (phi_lr_mult if is_phi[j] else 1.0)
                new.append(pj - step_lr * (m[j] / c1) / (torch.sqrt(v[j] / c2) + eps))
            params = _rebuild(params, iter(new))
        if i % log_every == 0:
            Lc, pc = L.detach().cpu().numpy(), {k: vv.detach().cpu().numpy() for k, vv in parts.items()}
            for b in range(B):
                logs[b][-1].update(loss=float(Lc[b]), **{f"L_{k}": float(vv[b]) for k, vv in pc.items()})
    w_used = LS.w
    return [dict(build=build, theta=logs[b][-1]["theta"], rel_rmse=logs[b][-1]["rel_rmse"], steps=logs[b][-1]["step"],
                 wall_s=logs[b][-1]["wall_s"], to_target=hit[b], weights=w_used, lr=lr, phi_lr_mult=phi_lr_mult,
                 v_scale=st[b]["vscale"], s_t=st[b]["st"], sigma=st[b]["sigma"], s_lo=st[b]["s_lo"],
                 device=str(device), dtype=str(dtype).replace("torch.", ""), batch=B, log=logs[b]) for b in range(B)]


def main():
    from inverse.lbfgs import load_test
    from pinn.jax_pinn import load_starts, summarize
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--build", choices=["cv", "c"], required=True)
    p.add_argument("--starts", default="fno_ttt,center", help="comma list: fno_ttt | fno | mlp | mlp_ttt (A), center (B)")
    p.add_argument("--materials", default="0:8")
    p.add_argument("--out", type=Path, default=Path("results/pinn_gpu"))
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--w-pde", type=float, default=100.0)
    p.add_argument("--w-vol", type=float, default=100.0)
    p.add_argument("--w-bc", type=float, default=100.0)
    p.add_argument("--phi-lr-mult", type=float, default=0.1)
    p.add_argument("--res-window", choices=["early", "observed"], default="observed")
    p.add_argument("--max-wall-s", type=float, default=4 * 3600.0)
    p.add_argument("--dtype", choices=["float64", "float32"], default="float64")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    T = load_test()
    lo, hi = map(int, a.materials.split(":"))
    jobs = []
    for st in a.starts.split(","):
        d = a.out / (("B_" if st == "center" or st.startswith("sobol") else "A_") + st) / a.build
        d.mkdir(parents=True, exist_ok=True)
        th0_all = load_starts(st, T)
        jobs += [(d, st, k, np.clip(th0_all[k], 1e-4, 1 - 1e-4)) for k in range(lo, hi)
                 if not (d / f"m{k:03d}.json").exists()]
    print(f"{a.build}: {len(jobs)} fits on {a.device} ({a.dtype})", flush=True)
    if jobs:
        def progress(i, wall, rel):
            print(f"step {i:5d}  {wall:7.0f} s  median rel rmse {np.median(rel):.2e}", flush=True)
        R = train_batch([T["y"][k] for _, _, k, _ in jobs], [T["iotas"][k] for _, _, k, _ in jobs],
                        [th0 for *_, th0 in jobs], a.build, [k for _, _, k, _ in jobs], steps=a.steps, lr=a.lr,
                        phi_lr_mult=a.phi_lr_mult, w=dict(pde=a.w_pde, vol=a.w_vol, bc=a.w_bc),
                        max_wall_s=a.max_wall_s, res_window=a.res_window, device=a.device,
                        dtype=getattr(torch, a.dtype), progress=progress, log_every=100)
        for (d, st, k, th0), r in zip(jobs, R):
            r.update(k=k, index=int(T["index"][k]), start=st, theta0=th0.tolist())
            (d / f"m{k:03d}.json").write_text(json.dumps(r))
    for d in sorted({d for d, *_ in jobs} or {a.out / (("B_" if s == "center" else "A_") + s) / a.build
                                                for s in a.starts.split(",")}):
        print(d, json.dumps(summarize(d)), flush=True)


if __name__ == "__main__":
    main()
