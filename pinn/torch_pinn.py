"""The total-loss PINNs of pinn/jax_pinn.py in PyTorch, so they run on the GPU (CUDA), many fits in one batch.

    python -m pinn.torch_pinn --build cv --starts fno_ttt,center --materials 0:8 --out results/pinn_gpu
    python -m pinn.torch_pinn --build c  --starts fno_ttt,center --materials 0:8 --out results/pinn_gpu

Same model, networks, losses, weights, optimizer and defaults as pinn/jax_pinn.py (see its docstring):
    L = L_data + w_pde L_pde + w_vol L_vol + w_bc L_bc   (all terms from the first step, no staging)
    Build 1 'cv': c and v networks; penalized conservation, volume constraint, wall fluxes.
    Build 2 'c' : c network, J by integration of c_t, v = (J - q) / c; penalized volume constraint, walls.
    one Adam over networks and phi (lr 1e-3, cosine decay to 1e-5, phi at phi_lr_mult x), 5000 steps.
The material (D alpha, t+, V) is pinn/material.py, the torch copy of solver/properties.py.
Staged training (the workflow doc's schedule; off by default, which gives the total-loss runs above):
    python -m pinn.torch_pinn --build cv --fit-steps 3000 --warmup-steps 1000 --steps 5000 --lr 1e-4 \
        --phi-lr-mult 10 --lbfgs-iters 300 --out results/pinn_staged
  1 fit     Adam on the data loss alone, theta frozen (--fit-steps, --fit-lr)
  2 warmup  Adam on the total loss, theta frozen (--warmup-steps), then
    joint   Adam on the total loss, networks and theta (--steps, --lr, theta at --phi-lr-mult x)
  3 lbfgs   L-BFGS on the total loss over every parameter, fixed residual points (--lbfgs-iters), batched per fit
Data-first schedule (Build 1): fit c AND v to the data until converged, then switch the physics on with a
small data weight:
    python -m pinn.torch_pinn --build cv --fit-v --fit-steps 20000 --fit-lbfgs-iters 2000 \
        --prefit-cache results/pinn_prefit/cache --steps 5000 --lr 1e-4 --phi-lr-mult 10 --lbfgs-iters 300 \
        --w-data 1e-3 --w-pde 1 --w-vol 1 --w-bc 1 --out results/pinn_prefit/wd1e-3
  1 fit     Adam (then L-BFGS, --fit-lbfgs-iters) on the data loss alone, theta frozen.  With --fit-v the v network
            is fitted too, to v reconstructed from the measured c (surrogate.data.velocity_from_c) at the cell
            centres; the stage-1 networks are cached per material in --prefit-cache and reused by later runs.
  2 joint   total loss = w_data (L_data + L_vdata) + w_pde L_pde + w_vol L_vol + w_bc L_bc, theta free.
            --freeze c,v holds the stage-1 networks fixed, so stage 2 fits only theta (D and t+) to the physics
            of the data-fitted fields; --freeze c holds only the c network.
Native-grid data (--data fine): the same test materials re-simulated by solver/dataset_fine.py and kept on the
generator's own grid (40 cells at 104 times, data/fine_test.npz) instead of 20 bins at 13 frames; the data loss
compares cell averages of the c network there and v is reconstructed from c on that grid.  Everything else is
unchanged, so the runs compare directly with the 13 x 20 ones:
    python -m pinn.prefit_sweep --out results/pinn_fine --name wd1e-2 --w-data 1e-2 --extra="--data fine"
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

N_BINS = 20                                   # bins of the benchmark data (data/train.npz)
TARGETS_REL = (1e-2, 1e-3, 1e-4)
GL3 = (np.array([-np.sqrt(0.6), 0.0, np.sqrt(0.6)]), np.array([5.0, 8.0, 5.0]) / 9.0)
GL4 = np.polynomial.legendre.leggauss(4)


class ObsGrid:
    """Where the data sit: output times t (tau), n_bins equal bins (cell averages), Gauss points per bin.
    The default is the benchmark grid (13 frames x 20 bins); fine(t) is the generator's native grid (40 cells)."""

    def __init__(self, t=TAU, n_bins=N_BINS, n_gauss=4):
        self.t, self.n_bins, self.n_gauss = np.asarray(t, float), int(n_bins), int(n_gauss)

    @classmethod
    def fine(cls, t, n_cells=40):
        return cls(t, n_cells, 2)


OBS_DEFAULT = ObsGrid()


def velocity_on_grid(y, iotas, obs):
    """surrogate.data.velocity_from_c on any ObsGrid: v (n, 3, T, n_bins) at the cell centres from c alone."""
    from solver.properties import RHO_COEFF, alpha_V
    tau, nb = obs.t, obs.n_bins
    valid = np.isfinite(y[:, 0, :, 0])
    c = np.where(np.isfinite(y), y, 1.0)
    fwd = (c[:, :, 1:] - c[:, :, :-1]) / np.diff(tau)[None, None, :, None]
    fv = (valid[:, 1:] & valid[:, :-1])[:, None, :, None].astype(float)
    num, den = np.zeros_like(c), np.zeros_like(c)
    num[:, :, 1:] += fwd * fv; den[:, :, 1:] += fv
    num[:, :, :-1] += fwd * fv; den[:, :, :-1] += fv
    dcdt = num / np.maximum(den, 1.0)
    V = np.asarray(alpha_V(c, RHO_COEFF)[1])
    cf = np.concatenate([c[..., :1], 0.5 * (c[..., 1:] + c[..., :-1]), c[..., -1:]], -1)
    J = np.asarray(iotas, float)[:, :, None, None] - np.concatenate(
        [np.zeros_like(c[..., :1]), np.cumsum(dcdt, -1)], -1) / nb
    v, q = np.zeros_like(J), J.copy()
    for j in range(nb):
        v[..., j + 1] = (v[..., j] - V[..., j] * (J[..., j + 1] - q[..., j])) / (1.0 - V[..., j] * cf[..., j + 1])
        q[..., j + 1] = J[..., j + 1] - cf[..., j + 1] * v[..., j + 1]
    v_cell = 0.5 * (v[..., 1:] + v[..., :-1])
    v_cell[~np.broadcast_to(valid[:, None, :, None], v_cell.shape)] = np.nan
    return v_cell


# ------------------------------------------------------------------------------------------------ networks
def init_mlp(gen, sizes, dtype):
    return [(torch.randn(a, b, generator=gen, dtype=dtype) * math.sqrt(1.0 / a), torch.zeros(b, dtype=dtype))
            for a, b in zip(sizes[:-1], sizes[1:])]


def mlp(params, z):
    for W, b in params[:-1]:
        z = torch.tanh(z @ W + b)
    W, b = params[-1]
    return (z @ W + b)[0]


def mlp_sin(params, z):
    """Correction-stage network (multi-stage training, Wang & Lai 2024): sin first layer, tanh after."""
    W, b = params[0]
    z = torch.sin(z @ W + b)
    for W, b in params[1:-1]:
        z = torch.tanh(z @ W + b)
    W, b = params[-1]
    return (z @ W + b)[0]


def _net(p, key, z):
    """Network `key` at inputs z, plus its correction stage when present: N(z) + eps * N1(kappa * z), where
    p["ms"] holds the fixed scale eps_<key> and input frequencies k_<key> of the correction (see pinn/multistage.py)."""
    out = mlp(p[key], z)
    if key + "1" in p:
        out = out + p["ms"]["eps_" + key] * mlp_sin(p[key + "1"], z * p["ms"]["k_" + key])
    return out


def _sqrt_time(t, tmax):
    pos = t > 0
    return torch.where(pos, torch.sqrt(torch.where(pos, t, torch.ones_like(t)) / tmax), torch.zeros_like(t))


def c_point(p, x, t, io, tmax, vs=None):
    """c = 1 + s N(x, s, i) (initial condition c = 1 built in).  With vs (the v scale) given, the wall fluxes are
    built in too: c = B + phi0(x) (a0 - B_x(0)) + phi1(x) (a1 - B_x(1)), where B is the plain network,
    phi0, phi1 are boundary-layer bumps (width s) with phi0'(0) = 1, phi1'(1) = 1 and zero value at both walls
    and zero slope at the far wall, and a0, a1 are the wall slopes that make J(0) = J(1) = i for the current
    theta and v: J(0) = q(0) (v(0) = 0) and J(1) = q(1) + c(1) v(1)."""
    s = _sqrt_time(t, tmax)

    def base(xx):
        return 1.0 + s * _net(p, "c", torch.stack([2 * xx - 1, 2 * s - 1, io / 5.0]))
    if vs is None:
        return base(x)
    th = torch.sigmoid(p["phi"])
    zero, one = torch.zeros_like(x), torch.ones_like(x)
    b0x, b0 = grad_and_value(base)(zero)
    b1x, b1 = grad_and_value(base)(one)
    dal0, tp0, _ = material(b0, th)
    dal1, tp1, _ = material(b1, th)
    v1 = v_point(p, one, t, io, tmax, vs)
    a0 = -(1 - tp0) * io / dal0
    a1 = (b1 * v1 - (1 - tp1) * io) / dal1
    ell = s + 1e-3
    phi0 = x * (1 - x) ** 2 * torch.exp(-x / ell)
    phi1 = (x - 1) * x ** 2 * torch.exp(-(1 - x) / ell)
    return base(x) + phi0 * (a0 - b0x) + phi1 * (a1 - b1x)


def v_point(p, x, t, io, tmax, vscale):
    s = _sqrt_time(t, tmax)
    return vscale * x * _net(p, "v", torch.stack([2 * x - 1, 2 * s - 1, io / 5.0]))


def q_point(p, th, x, t, io, tmax, vs=None):
    cx, c = grad_and_value(c_point, argnums=1)(p, x, t, io, tmax, vs)
    dal, tp, _ = material(c, th)
    return -dal * cx + tp * io


def fields(p, th, x, t, io, tmax, vs=None):
    """c, c_x, c_t, q, q_x, V at one point (vs given: hard wall-flux c, see c_point)."""
    (cx, ct), c = grad_and_value(c_point, argnums=(1, 2))(p, x, t, io, tmax, vs)
    qx, q = grad_and_value(q_point, argnums=2)(p, th, x, t, io, tmax, vs)
    V = material(c, th)[2]
    return c, cx, ct, q, qx, V


# ------------------------------------------------------------------------------------------------ quadrature
def bin_points(dtype, device, obs=OBS_DEFAULT):
    xg, wg = GL4 if obs.n_gauss == 4 else np.polynomial.legendre.leggauss(obs.n_gauss)
    edges = np.linspace(0, 1, obs.n_bins + 1)
    x = (0.5 * (edges[:-1, None] + edges[1:, None]) + 0.5 * np.diff(edges)[:, None] * xg[None]).ravel()
    return torch.tensor(x, dtype=dtype, device=device), torch.tensor(np.tile(wg / 2.0, obs.n_bins), dtype=dtype, device=device)


def cell_nodes(n_cells, dtype, device):
    xg, wg = GL3
    edges = np.linspace(0, 1, n_cells + 1)
    h = np.diff(edges)
    xq = (0.5 * (edges[:-1, None] + edges[1:, None]) + 0.5 * h[:, None] * xg[None]).ravel()
    wq = (0.5 * h[:, None] * wg[None]).ravel()
    return [torch.tensor(a, dtype=dtype, device=device) for a in (edges, xq, wq)]


# ------------------------------------------------------------------------------------------------ problem
def problem_stats(y, iotas, res_window="observed", obs=OBS_DEFAULT):
    """Per-material constants, as in jax_pinn.make_problem.  obs: the grid of y (default 13 frames x 20 bins)."""
    from surrogate.data import velocity_from_c
    vel = velocity_from_c if obs is OBS_DEFAULT else (lambda yy, io: velocity_on_grid(yy, io, obs))
    tau = obs.t
    mask = np.isfinite(y[0, :, 0])
    K = int(mask.sum())
    tmax = float(tau[K - 1])
    vscale = float(np.nanmax(np.abs(vel(y[None], np.asarray(iotas)[None])[0])))
    ct = np.diff(y[:, :K], axis=1) / np.diff(tau[:K])[None, :, None]
    st = float(np.sqrt(np.mean(ct ** 2)))
    s_lo = 0.05 if res_window == "early" else float(np.sqrt(tau[0] / tmax))
    v_obs = vel(y[None], np.asarray(iotas)[None])[0]                              # (3, T, n_bins), NaN off frames
    return dict(y=np.nan_to_num(y), mask=mask, v_obs=np.nan_to_num(v_obs), vmask=np.isfinite(v_obs), iotas=np.asarray(iotas, float), tmax=tmax, vscale=vscale, st=st,
                sigma=1e-3 * float(np.nanmean(y)), mean_c=float(np.nanmean(y)), s_lo=s_lo)


class Losses:
    """Per-fit loss functions (one material, unbatched) with the shared constants bound in."""

    def __init__(self, build, dtype, device, n_res=1024, n_times=32, n_cells=32, w=None, residual="strong",
                 hard_bc=False, n_fv_times=32, obs=OBS_DEFAULT):
        self.build, self.n_res, self.n_times, self.n_cells = build, n_res, n_times, n_cells
        self.w = dict(data=1.0, vdata=1.0, pde=100.0, vol=100.0, bc=100.0, prior=0.0) | (w or {})
        self.residual, self.hard_bc, self.n_fv_times = residual, hard_bc, n_fv_times
        if build == "c" and (hard_bc or residual != "strong"):
            raise ValueError("--hard-bc and --residual fv are for Build 1 (cv) only")
        self.use_vdata = False                                  # set by train_batch (fit_v, Build 1 only)
        self.obs = obs
        self.XC = torch.tensor((np.arange(obs.n_bins) + 0.5) / obs.n_bins, dtype=dtype, device=device)
        self.XB, self.WB = bin_points(dtype, device, obs)
        self.edges, self.xq, self.wq = cell_nodes(n_cells, dtype, device)
        self.t_obs = torch.tensor(obs.t, dtype=dtype, device=device)

    def bin_average(self, p, t, io, tmax, vs=None):
        c = vmap(lambda x: c_point(p, x, t, io, tmax, vs))(self.XB)
        return (c * self.WB).reshape(self.obs.n_bins, self.obs.n_gauss).sum(1)

    def data_loss(self, p, S):
        vs = S["vscale"] if self.hard_bc else None
        pred = vmap(lambda io: vmap(lambda t: self.bin_average(p, t, io, S["tmax"], vs))(self.t_obs))(S["iotas"])
        r = torch.where(S["mask"][None, :, None], pred - S["y"], torch.zeros_like(pred))
        return (r ** 2).sum() / (S["mask"].sum() * 3 * self.obs.n_bins)

    def vdata_loss(self, p, S):
        """Mean squared misfit of the v network to v reconstructed from the data, at the cell centres."""
        pred = vmap(lambda io: vmap(lambda t: vmap(lambda x: v_point(p, x, t, io, S["tmax"], S["vscale"]))(self.XC))(
            self.t_obs))(S["iotas"])
        r = torch.where(S["vmask"], pred - S["v_obs"], torch.zeros_like(pred))
        return (r ** 2).sum() / S["vmask"].sum()

    def data_total(self, p, S):
        """Stage-1 objective (and the data part of the total loss): c misfit / sigma^2 (+ v misfit / sigma_v^2)."""
        L = self.data_loss(p, S) / S["sigma"] ** 2
        if self.use_vdata:
            L = L + self.vdata_loss(p, S) / (1e-3 * S["vscale"]) ** 2
        return L

    def loss_cv(self, p, S, x, t):
        th = torch.sigmoid(p["phi"])
        tm, vs = S["tmax"], S["vscale"]
        hb = vs if self.hard_bc else None

        def J_point(x, t, io):
            return q_point(p, th, x, t, io, tm, hb) + c_point(p, x, t, io, tm, hb) * v_point(p, x, t, io, tm, vs)

        if self.residual == "fv":
            return self.loss_cv_fv(p, S, t, th, J_point, hb)

        def res(x, t, io):
            c, cx, ct, q, qx, V = fields(p, th, x, t, io, tm, hb)
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
        data = L["data"]
        if self.use_vdata:
            L["vdata"] = self.vdata_loss(p, S) / (1e-3 * vs) ** 2
            data = data + w["vdata"] * L["vdata"]              # w_vdata: weight of the v part (stage 2 and 3)
        return w["data"] * data + w["pde"] * L["pde"] + w["vol"] * L["vol"] + w["bc"] * L["bc"], L

    def loss_cv_fv(self, p, S, t, th, J_point, hb):
        """Build 1 with the conservation laws in integral (finite-volume) form on n_cells cells, so only first
        derivatives of c enter:  int_cell c_t dx + J(x_{j+1}) - J(x_j) = 0  and, integrating v_x = -V q_x by
        parts,  v_{j+1} - v_j + [V q]_j^{j+1} - int_cell V'(c) c_x q dx = 0.  Each residual is divided by the
        cell width, so it has the scale of the strong-form residual and the same normalization."""
        tm, vs = S["tmax"], S["vscale"]
        h = 1.0 / self.n_cells

        def V_of(c):
            return material(c, th)[2]

        def line(t, io):
            ct = vmap(lambda x: grad(c_point, argnums=2)(p, x, t, io, tm, hb))(self.xq)
            cx, c = vmap(lambda x: grad_and_value(c_point, argnums=1)(p, x, t, io, tm, hb))(self.xq)
            q = vmap(lambda x: q_point(p, th, x, t, io, tm, hb))(self.xq)
            Vc = vmap(grad(V_of))(c)
            cell = lambda f: (f * self.wq).reshape(self.n_cells, 3).sum(1)  # noqa: E731
            J = vmap(lambda x: J_point(x, t, io))(self.edges)
            ce = vmap(lambda x: c_point(p, x, t, io, tm, hb))(self.edges)
            qe = vmap(lambda x: q_point(p, th, x, t, io, tm, hb))(self.edges)
            ve = vmap(lambda x: v_point(p, x, t, io, tm, vs))(self.edges)
            Vq = V_of(ce) * qe
            rc = (cell(ct) + J[1:] - J[:-1]) / h
            rv = (ve[1:] - ve[:-1] + Vq[1:] - Vq[:-1] - cell(Vc * cx * q)) / h
            return rc, rv, torch.stack([J[0] - io, J[-1] - io])

        ts = t[:, :self.n_fv_times]
        rc, rv, wall = vmap(lambda tt, io: vmap(lambda b: line(b, io))(tt))(ts, S["iotas"])
        L = dict(data=self.data_loss(p, S) / S["sigma"] ** 2, pde=(rc ** 2).mean() / S["st"] ** 2,
                 vol=(rv ** 2).mean() / (vs * S["st"]) ** 2, bc=(wall ** 2).mean() / (S["iotas"] ** 2).mean())
        w = self.w
        data = L["data"]
        if self.use_vdata:
            L["vdata"] = self.vdata_loss(p, S) / (1e-3 * vs) ** 2
            data = data + w["vdata"] * L["vdata"]              # w_vdata: weight of the v part (stage 2 and 3)
        return w["data"] * data + w["pde"] * L["pde"] + w["vol"] * L["vol"] + w["bc"] * L["bc"], L

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
        return w["data"] * L["data"] + w["vol"] * L["vol"] + w["bc"] * L["bc"], L

    def loss(self, p, S, x, t):
        tot, L = (self.loss_cv if self.build == "cv" else self.loss_c)(p, S, x, t)
        if self.w["prior"] > 0:                       # pull toward the starting theta (e.g. the FNO estimate)
            L["prior"] = ((p["phi"] - S["phi0"]) ** 2).mean()
            tot = tot + self.w["prior"] * L["prior"]
        return tot, L


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


def _lbfgs_direction(g, hist, gamma):
    """Two-loop recursion, batched over fits: g (B, P), hist = [(s, y, rho)] oldest first, gamma (B,)."""
    q, alphas = g.clone(), []
    for s, y, rho in reversed(hist):
        a = rho * (s * q).sum(1)
        q -= a[:, None] * y
        alphas.append(a)
    r = gamma[:, None] * q
    for (s, y, rho), a in zip(hist, reversed(alphas)):
        b = rho * (y * r).sum(1)
        r += s * (a - b)[:, None]
    return -r


def train_batch(ys, iotas, theta0s, build, seeds, steps=5000, lr=1e-3, phi_lr_mult=0.1, w=None, log_every=100,
                max_wall_s=float("inf"), res_window="observed", width=64, depth=4, device="cuda",
                dtype=torch.float64, rng_seed=0, n_res=1024, n_times=32, n_cells=32, progress=None,
                fit_steps=0, fit_lr=1e-3, warmup_steps=0, lbfgs_iters=0, lbfgs_memory=20, lbfgs_log_every=10,
                fit_v=False, fit_lbfgs_iters=0, prefit_cache=None, freeze=(), residual="strong", hard_bc=False,
                obs=None, init_params=None, cache_tag="", return_params=False):
    """Train len(ys) independent PINNs together. Returns one result dict per fit (format of jax_pinn.train).

    Stages (each skipped when its count is 0; all 0 except `steps` is the total-loss training):
      1 'fit'    fit_steps Adam steps on the data loss alone, theta frozen (lr fit_lr, cosine decay to 1 %),
        'fitlbfgs' then fit_lbfgs_iters L-BFGS iterations on the same data loss.  fit_v (Build 1) adds the v
                 misfit to v reconstructed from c.  With prefit_cache, the stage-1 networks of each seed are
                 saved there and, when every seed is cached, loaded instead of refitted (theta set to theta0).
    freeze: networks ("c", "v") held fixed after stage 1; freeze=("c", "v") leaves only theta to the physics;
            "phi" holds theta fixed too (e.g. freeze=("c", "phi") trains only v).
    w["vdata"] (default 1) weights the v part of the data term in stages 2 and 3 (stage 1 is unchanged).
    return_params: also return the final parameters (stacked over fits): (results, params).
    residual: "strong" (pointwise PDE residuals) or "fv" (integral form on cells, Build 1; see loss_cv_fv).
    hard_bc: wall fluxes built into the c network (Build 1; see c_point).  c then depends on theta, so stage 1
             fits from each start separately (no stage-1 cache) and theta stays frozen there.
    w["prior"] > 0 adds w_prior * mean((phi - phi0)^2), phi0 = logit of the start.
    obs: ObsGrid of ys (default: the benchmark's 13 frames x 20 bins; ObsGrid.fine for data/fine_test.npz).
    init_params: one parameter dict per fit (CPU tensors) to start from instead of fresh networks, e.g. frozen
             stage-0 networks plus correction networks (pinn/multistage.py); phi is set from theta0s.
    cache_tag: extra suffix of the stage-1 cache file names (keeps fits to different targets apart).
      2 'warmup' warmup_steps Adam steps on the total loss, theta frozen (lr `lr`)
        'joint'  steps Adam steps on the total loss, networks and theta (lr `lr`, theta at phi_lr_mult x),
                 cosine decay to 1 % over warmup + joint
      3 'lbfgs'  lbfgs_iters L-BFGS iterations on the total loss over every parameter, with one fixed set of
                 residual points, batched over the fits (two-loop recursion and Armijo backtracking per fit)
    """
    B = len(ys)
    obs = obs or OBS_DEFAULT
    LS = Losses(build, dtype, device, n_res, n_times, n_cells, w, residual=residual, hard_bc=hard_bc, obs=obs)
    LS.use_vdata = bool(fit_v and build == "cv")
    st = [problem_stats(y, io, res_window, obs) for y, io in zip(ys, iotas)]
    S = {k: torch.tensor(np.stack([s[k] for s in st]), device=device, dtype=torch.bool if k == "mask" else dtype)
         for k in ("y", "mask", "v_obs", "vmask", "iotas", "tmax", "vscale", "st", "sigma", "s_lo")}
    S["vmask"] = S["vmask"].bool()
    sizes = [3] + [width] * depth + [1]
    P = []
    for i, (th0, sd) in enumerate(zip(theta0s, seeds)):
        g = torch.Generator().manual_seed(int(sd))
        if init_params is not None:
            p = {k: v for k, v in init_params[i].items() if k != "phi"}
        else:
            p = {"c": init_mlp(g, sizes, dtype)}
            if build == "cv":
                p["v"] = init_mlp(g, sizes, dtype)
        p["phi"] = torch.tensor(np.log(th0 / (1 - th0)), dtype=dtype)     # last leaf
        P.append(p)
    stacked = _stack(P)
    params = _rebuild(stacked, iter([x.to(device) for x in _leaves(stacked)]))
    S["phi0"] = params["phi"].detach().clone()
    n_leaves = len(_leaves(params))
    is_phi = [False] * (n_leaves - 1) + [True]
    is_frozen = [z for k in params for z in [k in freeze] * len(_leaves(params[k]))]
    b1, b2, eps = 0.9, 0.999, 1e-8
    gen = torch.Generator(device=device).manual_seed(rng_seed)

    def total(p, x, t):
        L, parts = vmap(LS.loss)(p, S, x, t)
        return L.sum(), (L, parts)

    def total_data(p, x, t):
        L = vmap(LS.data_total)(p, S)
        return L.sum(), (L, {"data": L})

    step_full = grad_and_value(total, has_aux=True)
    step_data = grad_and_value(total_data, has_aux=True)
    data_rmse = vmap(lambda p, S: torch.sqrt(LS.data_loss(p, S)))
    v_rmse = vmap(lambda p, S: torch.sqrt(LS.vdata_loss(p, S)) / S["vscale"]) if build == "cv" else None
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
    ctr = dict(step=0, evals=0)
    stage_wall = {}

    def record(stage, p):
        with torch.no_grad():
            rm = data_rmse(p, S).cpu().numpy()
            vr = v_rmse(p, S).cpu().numpy() if v_rmse else np.full(B, np.nan)
            th = torch.sigmoid(p["phi"]).cpu().numpy()
        wall = time.perf_counter() - t0
        for b in range(B):
            logs[b].append(dict(step=ctr["step"], evals=ctr["evals"], stage=stage, wall_s=wall,
                                rel_rmse=float(rm[b] / mean_c[b]), v_rel_rmse=float(vr[b]), theta=th[b].tolist()))
            for r in TARGETS_REL:
                if hit[b][str(r)][0] is None and rm[b] <= r * mean_c[b]:
                    hit[b][str(r)] = [ctr["step"], wall]
        if progress:
            progress(ctr["step"], wall, rm / mean_c, stage)
        return wall

    def attach(L, parts):
        Lc, pc = L.detach().cpu().numpy(), {k: vv.detach().cpu().numpy() for k, vv in parts.items()}
        for b in range(B):
            logs[b][-1].update(loss=float(Lc[b]), **{f"L_{k}": float(vv[b]) for k, vv in pc.items()})

    def adam(params, n, lr0, mult, step_fn, stage, decay_n=None, offset=0, frozen=None):
        """n Adam steps (fresh moments); lr cosine-decays from lr0 to lr0/100 over decay_n steps, phi at mult x."""
        decay_n = decay_n or n
        m = [torch.zeros_like(x) for x in _leaves(params)]
        v = [torch.zeros_like(x) for x in _leaves(params)]
        for i in range(n):
            logged = ctr["step"] % log_every == 0
            if logged and record(stage, params) > max_wall_s:
                return params, True
            x, t = draw()
            g, (_, (L, parts)) = step_fn(params, x, t)
            ctr["evals"] += 1
            k = offset + i
            sched = lr0 * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(k, decay_n) / decay_n)))
            with torch.no_grad():
                new = []
                for j, (pj, gj) in enumerate(zip(_leaves(params), _leaves(g))):
                    m[j].mul_(b1).add_(gj, alpha=1 - b1)
                    v[j].mul_(b2).addcmul_(gj, gj, value=1 - b2)
                    step_lr = sched * (0.0 if frozen and frozen[j] else mult if is_phi[j] else 1.0)
                    new.append(pj - step_lr * (m[j] / (1 - b1 ** (i + 1))) / (torch.sqrt(v[j] / (1 - b2 ** (i + 1))) + eps))
                params = _rebuild(params, iter(new))
            if logged:
                attach(L, parts)
            ctr["step"] += 1
        return params, False

    def lbfgs(params, n, step_fn=step_full, stage="lbfgs", frozen=None):
        shapes = [x.shape for x in _leaves(params)]
        sizes_ = [int(np.prod(s[1:])) for s in shapes]
        flat = lambda tree: torch.cat([z.reshape(B, -1) for z in _leaves(tree)], 1)  # noqa: E731
        unflat = lambda X: _rebuild(params, iter([z.reshape(s) for z, s in zip(torch.split(X, sizes_, 1), shapes)]))  # noqa: E731
        x_fix, t_fix = draw()
        keep = torch.cat([torch.full((B, n_), 0.0 if frozen and frozen[j] else 1.0, dtype=dtype, device=device)
                          for j, n_ in enumerate(sizes_)], 1)          # 0 on frozen parameters

        def fg(X):
            g, (_, (L, parts)) = step_fn(unflat(X), x_fix, t_fix)
            ctr["evals"] += 1
            L = L.detach()
            return torch.where(torch.isfinite(L), L, torch.full_like(L, float("inf"))), flat(g).detach() * keep, parts

        X = flat(params).detach()
        f, g, parts = fg(X)
        hist, gamma = [], 1.0 / g.norm(dim=1).clamp_min(1e-12)
        for it in range(n):
            if it % lbfgs_log_every == 0:
                params = unflat(X)
                if record(stage, params) > max_wall_s:
                    return params, True
                attach(f, parts)
            d = _lbfgs_direction(g, hist, gamma)
            gd = (g * d).sum(1)
            bad = ~(gd < 0)                                        # not a descent direction: restart that fit
            if bad.any():
                d = torch.where(bad[:, None], -g / g.norm(dim=1, keepdim=True).clamp_min(1e-12), d)
                gd = (g * d).sum(1)
                hist = [(s, y, torch.where(bad, torch.zeros_like(r), r)) for s, y, r in hist]
            alpha, done = torch.ones(B, dtype=dtype, device=device), torch.zeros(B, dtype=torch.bool, device=device)
            Xn, fn, gn = X.clone(), f.clone(), g.clone()
            for _ in range(20):
                ft, gt, pt = fg(X + alpha[:, None] * d)
                ok = ~done & (ft <= f + 1e-4 * alpha * gd)
                Xn[ok], fn[ok], gn[ok] = (X + alpha[:, None] * d)[ok], ft[ok], gt[ok]
                if ok.any():
                    parts = {k: torch.where(ok, v, parts[k]) for k, v in pt.items()}
                done |= ok
                if done.all():
                    break
                alpha = torch.where(done, alpha, 0.5 * alpha)
            s, yv = Xn - X, gn - g
            sy = (s * yv).sum(1)
            valid = done & (sy > 1e-12 * s.norm(dim=1) * yv.norm(dim=1))
            hist.append((s, yv, torch.where(valid, 1.0 / torch.where(valid, sy, torch.ones_like(sy)), torch.zeros_like(sy))))
            hist = hist[-lbfgs_memory:]
            gamma = torch.where(valid, sy / (yv * yv).sum(1).clamp_min(1e-300), gamma)
            failed = ~done                                         # no decrease even at a tiny step: restart
            if failed.any():
                hist = [(s_, y_, torch.where(failed, torch.zeros_like(r_), r_)) for s_, y_, r_ in hist]
                gamma = torch.where(failed, 1.0 / gn.norm(dim=1).clamp_min(1e-12), gamma)
            X, f, g = Xn, fn, gn
            ctr["step"] += 1
        return unflat(X), False

    stopped = False
    cache = Path(prefit_cache) if prefit_cache and not hard_bc else None
    tag = f"{build}_v{int(LS.use_vdata)}_f{fit_steps}_lr{fit_lr:g}_l{fit_lbfgs_iters}_w{width}x{depth}"
    if obs is not OBS_DEFAULT:                                   # never mix stage-1 fits of different data grids
        tag += f"_g{len(obs.t)}x{obs.n_bins}"
    tag += cache_tag
    files = [cache / f"{tag}_s{int(sd)}.pt" for sd in seeds] if cache else []
    if cache and all(f.exists() for f in files):
        loaded = [torch.load(f, map_location=device) for f in files]
        new = _leaves(_stack(loaded))
        new[-1] = params["phi"]                                  # theta from this run's start
        params = _rebuild(params, iter([z.to(device=device, dtype=dtype) for z in new]))
        stage_wall["fit_cached"] = time.perf_counter() - t0
    else:
        if fit_steps:
            params, stopped = adam(params, fit_steps, fit_lr, 0.0, step_data, "fit")
            stage_wall["fit"] = time.perf_counter() - t0
        if fit_lbfgs_iters and not stopped:
            params, stopped = lbfgs(params, fit_lbfgs_iters, step_data, "fitlbfgs", frozen=is_phi)
            stage_wall["fitlbfgs"] = time.perf_counter() - t0
        if cache and not stopped:
            cache.mkdir(parents=True, exist_ok=True)
            for b, f in enumerate(files):
                if not f.exists():
                    torch.save(_rebuild(params, iter([z[b].detach().cpu() for z in _leaves(params)])), f)
    with torch.no_grad():
        fit_end = dict(rel_rmse=(data_rmse(params, S).cpu().numpy() / mean_c).tolist(),
                       v_rel_rmse=(v_rmse(params, S).cpu().numpy().tolist() if v_rmse else [None] * B))
    if warmup_steps and not stopped:
        params, stopped = adam(params, warmup_steps, lr, 0.0, step_full, "warmup", decay_n=warmup_steps + steps,
                               frozen=is_frozen)
        stage_wall["warmup"] = time.perf_counter() - t0
    if steps and not stopped:
        params, stopped = adam(params, steps, lr, phi_lr_mult, step_full, "joint",
                               decay_n=warmup_steps + steps, offset=warmup_steps, frozen=is_frozen)
        stage_wall["joint"] = time.perf_counter() - t0
    if lbfgs_iters and not stopped:
        params, stopped = lbfgs(params, lbfgs_iters, frozen=is_frozen)
        stage_wall["lbfgs"] = time.perf_counter() - t0
    record("end", params)
    w_used = LS.w
    sched = dict(fit_steps=fit_steps, fit_lr=fit_lr, fit_lbfgs_iters=fit_lbfgs_iters, fit_v=LS.use_vdata,
                 freeze=list(freeze), residual=residual, hard_bc=hard_bc, warmup_steps=warmup_steps, joint_steps=steps, lr=lr,
                 phi_lr_mult=phi_lr_mult, lbfgs_iters=lbfgs_iters, lbfgs_memory=lbfgs_memory)
    out = [dict(build=build, theta=logs[b][-1]["theta"], rel_rmse=logs[b][-1]["rel_rmse"], steps=logs[b][-1]["step"],
                 evals=logs[b][-1]["evals"], wall_s=logs[b][-1]["wall_s"], v_rel_rmse=logs[b][-1]["v_rel_rmse"],
                 fit_end_rel_rmse=fit_end["rel_rmse"][b], fit_end_v_rel_rmse=fit_end["v_rel_rmse"][b], to_target=hit[b], weights=w_used, lr=lr,
                 phi_lr_mult=phi_lr_mult, schedule=sched, stage_wall_s=stage_wall, stopped_by_wall=stopped,
                 v_scale=st[b]["vscale"], s_t=st[b]["st"], sigma=st[b]["sigma"], s_lo=st[b]["s_lo"],
                 device=str(device), dtype=str(dtype).replace("torch.", ""), batch=B, log=logs[b]) for b in range(B)]
    if return_params:
        return out, params
    return out


def main():
    from inverse.lbfgs import load_test
    from pinn.jax_pinn import load_starts, summarize
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--build", choices=["cv", "c"], required=True)
    p.add_argument("--starts", default="fno_ttt,center", help="comma list: fno_ttt | fno | mlp | mlp_ttt (A), center (B)")
    p.add_argument("--materials", default="0:8")
    p.add_argument("--out", type=Path, default=Path("results/pinn_gpu"))
    p.add_argument("--steps", type=int, default=5000, help="joint Adam steps (networks and theta)")
    p.add_argument("--lr", type=float, default=1e-3, help="network learning rate of the warmup and joint stages")
    p.add_argument("--w-data", type=float, default=1.0, help="weight of the data terms in the total loss")
    p.add_argument("--w-pde", type=float, default=100.0)
    p.add_argument("--w-vol", type=float, default=100.0)
    p.add_argument("--w-bc", type=float, default=100.0)
    p.add_argument("--phi-lr-mult", type=float, default=0.1)
    p.add_argument("--res-window", choices=["early", "observed"], default="observed")
    p.add_argument("--max-wall-s", type=float, default=4 * 3600.0)
    p.add_argument("--dtype", choices=["float64", "float32"], default="float64")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--fit-steps", type=int, default=0, help="stage 1: Adam steps on the data alone, theta frozen")
    p.add_argument("--fit-lr", type=float, default=1e-3)
    p.add_argument("--warmup-steps", type=int, default=0, help="stage 2a: total-loss Adam steps with theta frozen")
    p.add_argument("--lbfgs-iters", type=int, default=0, help="stage 3: L-BFGS iterations on the total loss")
    p.add_argument("--fit-v", action="store_true", help="Build 1: fit v to v reconstructed from c in stage 1 (and keep it as data)")
    p.add_argument("--fit-lbfgs-iters", type=int, default=0, help="stage 1: L-BFGS iterations on the data loss after the Adam fit")
    p.add_argument("--prefit-cache", type=Path, default=None, help="folder caching the stage-1 networks per material")
    p.add_argument("--freeze", default="", help="comma list of networks (c, v) held fixed after stage 1, e.g. c,v")
    p.add_argument("--residual", choices=["strong", "fv"], default="strong", help="pointwise or integral-form physics")
    p.add_argument("--hard-bc", action="store_true", help="build the wall fluxes into the c network (Build 1)")
    p.add_argument("--w-prior", type=float, default=0.0, help="weight of mean((phi - phi_start)^2)")
    p.add_argument("--w-vdata", type=float, default=1.0, help="weight of the v data term in stages 2 and 3 (0 = c data only)")
    p.add_argument("--save-params", action="store_true", help="also save each fit's final networks as m<k>_params.pt")
    p.add_argument("--data", choices=["benchmark", "fine"], default="benchmark",
                   help="benchmark: 13 frames x 20 bins (data/train.npz); fine: native grid (data/fine_test.npz)")
    a = p.parse_args()
    T = load_test()
    obs, Y = None, T["y"]
    if a.data == "fine":
        F = np.load(Path(__file__).resolve().parents[1] / "data" / "fine_test.npz")
        assert np.array_equal(F["index"], T["index"][:len(F["index"])]), "fine_test.npz is not the load_test() order"
        obs, Y = ObsGrid.fine(F["t"]), F["y"]
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
        def progress(i, wall, rel, stage):
            print(f"{stage:6s} step {i:5d}  {wall:7.0f} s  median rel rmse {np.median(rel):.2e}", flush=True)
        R = train_batch([Y[k] for _, _, k, _ in jobs], [T["iotas"][k] for _, _, k, _ in jobs],
                        [th0 for *_, th0 in jobs], a.build, [k for _, _, k, _ in jobs], steps=a.steps, lr=a.lr,
                        phi_lr_mult=a.phi_lr_mult, w=dict(data=a.w_data, vdata=a.w_vdata, prior=a.w_prior, pde=a.w_pde, vol=a.w_vol, bc=a.w_bc),
                        max_wall_s=a.max_wall_s, res_window=a.res_window, device=a.device,
                        fit_v=a.fit_v, fit_lbfgs_iters=a.fit_lbfgs_iters, prefit_cache=a.prefit_cache,
                        freeze=tuple(x for x in a.freeze.split(",") if x), residual=a.residual, hard_bc=a.hard_bc,
                        dtype=getattr(torch, a.dtype), progress=progress, log_every=100,
                        fit_steps=a.fit_steps, fit_lr=a.fit_lr, warmup_steps=a.warmup_steps, lbfgs_iters=a.lbfgs_iters,
                        obs=obs, return_params=a.save_params)
        if a.save_params:
            R, Pf = R
        for b, ((d, st, k, th0), r) in enumerate(zip(jobs, R)):
            r.update(k=k, index=int(T["index"][k]), start=st, theta0=th0.tolist(), data=a.data)
            (d / f"m{k:03d}.json").write_text(json.dumps(r))
            if a.save_params:
                torch.save(_rebuild(Pf, iter([z[b].detach().cpu() for z in _leaves(Pf)])), d / f"m{k:03d}_params.pt")
    for d in sorted({d for d, *_ in jobs} or {a.out / (("B_" if s == "center" else "A_") + s) / a.build
                                                for s in a.starts.split(",")}):
        print(d, json.dumps(summarize(d)), flush=True)


if __name__ == "__main__":
    main()
