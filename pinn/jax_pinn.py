"""Per-material PINNs for D(c), t+(c) in JAX: Build 1 (c_net + v_net) and Build 2 (c_net only, v by integration).

    python -m pinn.jax_pinn --build cv --start fno_ttt --materials 0:10 --out "results/pinn/A_fno_ttt/cv"
    python -m pinn.jax_pinn --build c  --start center  --materials 0:10 --out "results/pinn/B_center/c"

Data and starts are the ones of inverse/lbfgs.py: the first 300 'id_test' materials (3 currents x up to 13 frames
x 20 bins of c), and theta started at input A (network prediction, e.g. fno_ttt) or input B (center = 0.5).

Model (the full model of solver/fv.py, dimensionless):
    dc/dt + dJ/dx = 0,  J = q + c v,  q = -D(c) alpha(c) dc/dx + t+(c) iota,  dv/dx = -V(c) dq/dx,
    v(0, t) = 0,  J(0, t) = J(1, t) = iota,  c(x, 0) = 1.
D, t+, alpha, V come from solver/properties.py (the solver's own material code).  theta = sigmoid(phi).

Networks take (x, s, iota / 5) with s = sqrt(t / t_max) (t_max = the material's last valid frame); one network
serves all three currents.  c = 1 + s N_c, so c(x, 0) = 1 holds exactly.

Build 1 'cv':  v = v_scale x N_v, so v(0) = 0 exactly.  Penalized: conservation r_c = c_t + J_x, volume
               r_v = v_x + V q_x, and J = iota at both walls.
Build 2 'c':   J(x, t) = iota - int_0^x c_t dx' (composite 3-point Gauss-Legendre on 32 cells), v = (J - q) / c.
               Conservation and J(0) = iota then hold exactly.  Penalized: volume r_v = v_x + V q_x with
               v_x = (-c_t - q_x - v c_x) / c, v(0, t) = 0 (i.e. q(0) = iota) and J(1, t) = iota.

Total loss, every term from the first step (no staged training):
    L = L_data + w_pde L_pde + w_vol L_vol + w_bc L_bc
    L_data = mean over observed (current, frame, bin) of (bin-average of c - data)^2 / sigma_d^2
    L_pde  = mean r_c^2 / s_t^2          (Build 1 only; s_t = RMS of dc/dt in the data, as in pinn/pde_loss.py)
    L_vol  = mean r_v^2 / (v_scale s_t)^2  (v_scale = max |v| reconstructed from the data)
    L_bc   = mean (wall residual)^2 / mean(iota^2)
with sigma_d = 1e-3 x mean c (so L_data = 1 at an RMS misfit of 0.1 % of mean c).
Optimizer: one Adam (optax) on all parameters at once (networks and phi), learning rate 1e-3 with cosine decay
to 1e-5 over the run, phi at phi_lr_mult x that rate.  No staging, no L-BFGS polish.
Residual points are redrawn every step: x uniform in [0, 1] and s uniform in [s_lo, 1], where s_lo is
sqrt(first frame / t_max) (res_window='observed') or 0.05 ('early').
Defaults are the setting picked by the sweep in pinn/sweep.py: w_pde = w_vol = w_bc = 100, phi_lr_mult = 0.1,
res_window = 'observed', 5000 steps, networks 4 x 64 tanh.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402

from solver.dataset import TAU  # noqa: E402
from solver.properties import RHO_COEFF, BernsteinMaterial  # noqa: E402

jax.config.update("jax_enable_x64", True)

MODEL = BernsteinMaterial(jnp.asarray(RHO_COEFF))
N_BINS = 20
TARGETS_REL = (1e-2, 1e-3, 1e-4)
GL3 = (np.array([-np.sqrt(0.6), 0.0, np.sqrt(0.6)]), np.array([5.0, 8.0, 5.0]) / 9.0)
GL4 = np.polynomial.legendre.leggauss(4)


# ------------------------------------------------------------------------------------------------ networks
def init_mlp(key, sizes):
    params = []
    for k, (a, b) in zip(jax.random.split(key, len(sizes) - 1), zip(sizes[:-1], sizes[1:])):
        params.append((jax.random.normal(k, (a, b)) * jnp.sqrt(1.0 / a), jnp.zeros(b)))
    return params


def mlp(params, z):
    for W, b in params[:-1]:
        z = jnp.tanh(z @ W + b)
    W, b = params[-1]
    return (z @ W + b)[0]


def theta_of(phi):
    return jax.nn.sigmoid(phi)


def _sqrt_time(t, tmax):
    """s = sqrt(t / tmax), exactly 0 at t = 0 (the double where keeps the gradient finite there)."""
    pos = t > 0
    return jnp.where(pos, jnp.sqrt(jnp.where(pos, t, 1.0) / tmax), 0.0)


def c_point(p, x, t, io, tmax):
    s = _sqrt_time(t, tmax)
    return 1.0 + s * mlp(p["c"], jnp.stack([2 * x - 1, 2 * s - 1, io / 5.0]))


def v_point(p, x, t, io, tmax, vscale):
    s = _sqrt_time(t, tmax)
    return vscale * x * mlp(p["v"], jnp.stack([2 * x - 1, 2 * s - 1, io / 5.0]))


def q_point(p, th, x, t, io, tmax):
    c, cx = jax.value_and_grad(c_point, argnums=1)(p, x, t, io, tmax)
    dal, tp, _ = MODEL(c, th)
    return -dal * cx + tp * io


def fields(p, th, x, t, io, tmax):
    """c, c_x, c_t, q, q_x, V at one point."""
    c, (cx, ct) = jax.value_and_grad(c_point, argnums=(1, 2))(p, x, t, io, tmax)
    q, qx = jax.value_and_grad(q_point, argnums=2)(p, th, x, t, io, tmax)
    V = MODEL(c, th)[2]
    return c, cx, ct, q, qx, V


# ------------------------------------------------------------------------------------------------ data
def bin_points():
    """Gauss points (80,) and weights (80,) for the 20 bin averages, plus the bin index of each point."""
    xg, wg = GL4
    edges = np.linspace(0, 1, N_BINS + 1)
    x = (0.5 * (edges[:-1, None] + edges[1:, None]) + 0.5 * np.diff(edges)[:, None] * xg[None]).ravel()
    w = np.tile(wg / 2.0, N_BINS)                    # bin average = sum w f over the bin's 4 points
    return jnp.asarray(x), jnp.asarray(w)


XB, WB = bin_points()


def bin_average(p, t, io, tmax):
    c = jax.vmap(lambda x: c_point(p, x, t, io, tmax))(XB)
    return (c * WB).reshape(N_BINS, 4).sum(1)


# ------------------------------------------------------------------------------------------------ losses
def make_problem(y, iotas, theta0, build, key, width=64, depth=4, n_res=1024, n_times=32, n_cells=32,
                 res_window="observed"):
    """Everything the loss needs for one material; returns (params, static dict).

    res_window: where residual points go in time, "early" = s in [0.05, 1] (t from 0.25 % of t_max),
    "observed" = t in [first frame, t_max] (skips the sqrt(t) start-up layer before the first frame).
    """
    mask = np.isfinite(y[0, :, 0])
    K = int(mask.sum())
    tmax = float(TAU[K - 1])
    from surrogate.data import velocity_from_c
    vfc = velocity_from_c(y[None], np.asarray(iotas)[None])[0]
    vscale = float(np.nanmax(np.abs(vfc)))
    ct = np.diff(y[:, :K], axis=1) / np.diff(TAU[:K])[None, :, None]
    st = float(np.sqrt(np.mean(ct ** 2)))
    s_lo = 0.05 if res_window == "early" else float(np.sqrt(TAU[0] / tmax))
    sizes = [3] + [width] * depth + [1]
    kc, kv = jax.random.split(key)
    params = {"c": init_mlp(kc, sizes), "phi": jnp.asarray(np.log(theta0 / (1 - theta0)))}
    if build == "cv":
        params["v"] = init_mlp(kv, sizes)
    S = dict(y=jnp.asarray(np.nan_to_num(y)), mask=jnp.asarray(mask), iotas=jnp.asarray(iotas, dtype=jnp.float64),
             t_obs=jnp.asarray(TAU), tmax=tmax, vscale=vscale, st=st, sigma=1e-3 * float(np.nanmean(y)),
             mean_c=float(np.nanmean(y)), n_res=n_res, n_times=n_times, n_cells=n_cells, s_lo=s_lo, build=build)
    return params, S


def data_loss(p, S):
    pred = jax.vmap(lambda io: jax.vmap(lambda t: bin_average(p, t, io, S["tmax"]))(S["t_obs"]))(S["iotas"])
    r = jnp.where(S["mask"][None, :, None], pred - S["y"], 0.0)
    mse = jnp.sum(r ** 2) / (jnp.sum(S["mask"]) * 3 * N_BINS)
    return mse


def loss_cv(p, S, key, w):
    th = theta_of(p["phi"])
    k1, k2 = jax.random.split(key)
    n = S["n_res"]
    x = jax.random.uniform(k1, (3, n))
    t = S["tmax"] * jax.random.uniform(k2, (3, n), minval=S["s_lo"], maxval=1.0) ** 2

    def J_point(x, t, io):
        c = c_point(p, x, t, io, S["tmax"])
        return q_point(p, th, x, t, io, S["tmax"]) + c * v_point(p, x, t, io, S["tmax"], S["vscale"])

    def res(x, t, io):
        c, cx, ct, q, qx, V = fields(p, th, x, t, io, S["tmax"])
        Jx = jax.grad(J_point, argnums=0)(x, t, io)
        vx = jax.grad(v_point, argnums=1)(p, x, t, io, S["tmax"], S["vscale"])
        return ct + Jx, vx + V * qx

    rc, rv = jax.vmap(lambda xs, ts, io: jax.vmap(lambda a, b: res(a, b, io))(xs, ts))(x, t, S["iotas"])
    tb = t[:, :256]
    wall = jax.vmap(lambda ts, io: jax.vmap(lambda b: jnp.stack([J_point(0.0, b, io) - io, J_point(1.0, b, io) - io]))(ts))(tb, S["iotas"])
    L = dict(data=data_loss(p, S) / S["sigma"] ** 2, pde=jnp.mean(rc ** 2) / S["st"] ** 2,
             vol=jnp.mean(rv ** 2) / (S["vscale"] * S["st"]) ** 2, bc=jnp.mean(wall ** 2) / jnp.mean(S["iotas"] ** 2))
    return L["data"] + w["pde"] * L["pde"] + w["vol"] * L["vol"] + w["bc"] * L["bc"], L


def _cell_nodes(n_cells):
    xg, wg = GL3
    edges = np.linspace(0, 1, n_cells + 1)
    h = np.diff(edges)
    xq = (0.5 * (edges[:-1, None] + edges[1:, None]) + 0.5 * h[:, None] * xg[None]).ravel()
    wq = (0.5 * h[:, None] * wg[None]).ravel()
    return jnp.asarray(edges), jnp.asarray(xq), jnp.asarray(wq)


def build2_line(p, th, t, io, S):
    """Along x at one time: nodes x_m, J, v, residual r_v at the nodes; J(1) - iota; q(0) - iota."""
    edges, xq, wq = _cell_nodes(S["n_cells"])
    ct_q = jax.vmap(lambda x: jax.grad(c_point, argnums=2)(p, x, t, io, S["tmax"]))(xq)
    integ = jnp.concatenate([jnp.zeros(1), jnp.cumsum((ct_q * wq).reshape(S["n_cells"], 3).sum(1))])
    J = io - integ                                                      # at the cell edges
    c, cx, ct, q, qx, V = jax.vmap(lambda x: fields(p, th, x, t, io, S["tmax"]))(edges)
    v = (J - q) / c
    vx = (-ct - qx - v * cx) / c
    rv = vx + V * qx
    return rv, J[-1] - io, q[0] - io


def loss_c(p, S, key, w):
    th = theta_of(p["phi"])
    t = S["tmax"] * jax.random.uniform(key, (3, S["n_times"]), minval=S["s_lo"], maxval=1.0) ** 2
    rv, j1, q0 = jax.vmap(lambda ts, io: jax.vmap(lambda b: build2_line(p, th, b, io, S))(ts))(t, S["iotas"])
    L = dict(data=data_loss(p, S) / S["sigma"] ** 2, pde=jnp.zeros(()),
             vol=jnp.mean(rv ** 2) / (S["vscale"] * S["st"]) ** 2,
             bc=(jnp.mean(j1 ** 2) + jnp.mean(q0 ** 2)) / jnp.mean(S["iotas"] ** 2))
    return L["data"] + w["vol"] * L["vol"] + w["bc"] * L["bc"], L


# ------------------------------------------------------------------------------------------------ training
def train(y, iotas, theta0, build, steps=5000, lr=1e-3, phi_lr_mult=0.1, w=None, seed=0, log_every=100,
          max_wall_s=float("inf"), return_params=False, **kw):
    """Train one PINN with the total loss and one Adam optimizer. Returns the result dict with a log."""
    w = dict(pde=100.0, vol=100.0, bc=100.0) | (w or {})
    key = jax.random.PRNGKey(seed)
    key, k0 = jax.random.split(key)
    params, S = make_problem(y, iotas, theta0, build, k0, **kw)
    loss_fn = loss_cv if build == "cv" else loss_c
    sched = optax.cosine_decay_schedule(lr, steps, alpha=0.01)
    labels = {k: ("phi" if k == "phi" else "net") for k in params}
    opt = optax.multi_transform({"net": optax.adam(sched),
                                 "phi": optax.adam(lambda i: phi_lr_mult * sched(i))}, labels)
    state = opt.init(params)
    static = {k: S[k] for k in ("n_res", "n_times", "n_cells", "tmax", "vscale", "st", "sigma", "s_lo")}
    Sj = {k: v for k, v in S.items() if k not in static and k != "build"}

    @jax.jit
    def step(params, state, key, Sj):
        Sfull = Sj | static
        (L, parts), g = jax.value_and_grad(lambda q: loss_fn(q, Sfull, key, w), has_aux=True)(params)
        upd, state = opt.update(g, state, params)
        return optax.apply_updates(params, upd), state, L, parts

    log, t0, hit = [], time.perf_counter(), {str(r): [None, None] for r in TARGETS_REL}
    for i in range(steps + 1):
        key, k = jax.random.split(key)
        if i % log_every == 0 or i == steps:
            rmse = float(jnp.sqrt(data_loss(params, Sj | static)))
            th = np.asarray(theta_of(params["phi"]))
            wall = time.perf_counter() - t0
            log.append(dict(step=i, wall_s=wall, rel_rmse=rmse / S["mean_c"], theta=th.tolist()))
            for r in TARGETS_REL:
                if hit[str(r)][0] is None and rmse <= r * S["mean_c"]:
                    hit[str(r)] = [i, wall]
            if wall > max_wall_s:
                break
        if i == steps:
            break
        params, state, L, parts = step(params, state, k, Sj)
        if i % log_every == 0:
            log[-1].update(loss=float(L), **{f"L_{k}": float(v) for k, v in parts.items()})
    out = dict(build=build, theta=log[-1]["theta"], rel_rmse=log[-1]["rel_rmse"], steps=log[-1]["step"],
               wall_s=log[-1]["wall_s"], to_target=hit, weights=w, lr=lr, phi_lr_mult=phi_lr_mult,
               v_scale=S["vscale"], s_t=S["st"], sigma=S["sigma"], s_lo=S["s_lo"], log=log)
    if return_params:
        out.update(params=params, problem=S)
    return out


def summarize(out: Path):
    from analysis.metrics import score_all
    from inverse.lbfgs import load_test
    T = load_test()
    rows = sorted((json.loads(p.read_text()) for p in out.glob("m*.json")), key=lambda r: r["k"])
    if not rows:
        return {}
    ks = [r["k"] for r in rows]
    sc = score_all(np.array([r["theta"] for r in rows]), T["theta"][ks], T["window"][ks])
    sc0 = score_all(np.array([r["theta0"] for r in rows]), T["theta"][ks], T["window"][ks])

    def med(a):
        a = [x for x in a if x is not None]
        return float(np.median(a)) if a else None
    summ = dict(n=len(rows), build=rows[0]["build"], start=rows[0]["start"],
                median_start={k: float(np.median(v)) for k, v in sc0.items()},
                median_fit={k: float(np.median(v)) for k, v in sc.items()},
                D_err_improved=int(sum(a < b for a, b in zip(sc["D_relL2_pct"], sc0["D_relL2_pct"]))),
                median_rel_rmse=float(np.median([r["rel_rmse"] for r in rows])),
                median_wall_s=float(np.median([r["wall_s"] for r in rows])))
    for rel in TARGETS_REL:
        hits = [r["to_target"][str(rel)] for r in rows]
        summ[f"target_{rel:g}"] = dict(reached=sum(h[0] is not None for h in hits),
                                       median_steps=med([h[0] for h in hits]), median_wall_s=med([h[1] for h in hits]))
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    return summ


def load_starts(start, T, net_seed=0):
    """The starting thetas of inverse/lbfgs.py and inverse/gradfree.py (same cache), computed if neither exists."""
    if start == "truth":                                    # diagnostic only: theta held or started at the truth
        return np.asarray(T["theta"])
    d = ("B_" if start == "center" or start.startswith("sobol") else "A_") + start
    for root in ("results/lbfgs", "results/gradfree"):
        if (Path(root) / d / "starts.npy").exists():
            return np.load(Path(root) / d / "starts.npy")
    from inverse.lbfgs import starts
    return starts(start, T, net_seed)


def main():
    from inverse.lbfgs import load_test
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--build", choices=["cv", "c"], required=True)
    p.add_argument("--start", required=True, help="fno_ttt | fno | mlp | mlp_ttt (input A) or center (input B)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--materials", default="0:10")
    p.add_argument("--shard", default="0/1")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--w-pde", type=float, default=100.0)
    p.add_argument("--w-vol", type=float, default=100.0)
    p.add_argument("--w-bc", type=float, default=100.0)
    p.add_argument("--phi-lr-mult", type=float, default=0.1, help="learning rate of phi = this x the network rate")
    p.add_argument("--res-window", choices=["early", "observed"], default="observed")
    p.add_argument("--max-wall-s", type=float, default=3600.0)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    T = load_test()
    th0_all = load_starts(a.start, T)
    lo, hi = map(int, a.materials.split(":"))
    i, n = map(int, a.shard.split("/"))
    for k in [k for k in range(lo, hi) if k % n == i and not (a.out / f"m{k:03d}.json").exists()]:
        th0 = np.clip(th0_all[k], 1e-4, 1 - 1e-4)
        r = train(T["y"][k], T["iotas"][k], th0, a.build, steps=a.steps, lr=a.lr, phi_lr_mult=a.phi_lr_mult,
                  w=dict(pde=a.w_pde, vol=a.w_vol, bc=a.w_bc), seed=k, max_wall_s=a.max_wall_s,
                  res_window=a.res_window)
        r.update(k=k, index=int(T["index"][k]), start=a.start, theta0=th0.tolist())
        (a.out / f"m{k:03d}.json").write_text(json.dumps(r))
        print(f"{a.build} {a.start} m{k:03d}: {r['steps']} steps, {r['wall_s']:.0f} s, rel rmse {r['rel_rmse']:.1e}",
              flush=True)
    print(json.dumps(summarize(a.out), indent=1))


if __name__ == "__main__":
    main()
