"""Per-material fit of D(c), t+(c) with the differentiable solver and L-BFGS-B (the DiffEC-style route).

    python -m inverse.lbfgs --start center --out "results/lbfgs/B_center"       (input B: generic start)
    python -m inverse.lbfgs --start fno    --out "results/lbfgs/A_fno"          (input A: learned start)
    python -m inverse.lbfgs --start fno_ttt --out "results/lbfgs/A_fno_ttt"     (input A: learned start + TTT)

Data: the first 300 'id_test' materials of data/train.npz (the same materials the networks are scored on).
Each has binned c for 3 currents x 13 frames x 20 bins; frames past the valid prefix K are NaN.

Forward model: solver.integrate (cell-centred FV, Crank-Nicolson with Rannacher start, Newton per step),
at the generator's own settings (40 cells, 16 steps per frame interval), binned with the generator's matrix.
The data are noise-free and made with this same discretization, so a perfect fit is reachable.

Loss: mean squared error of binned c over the observed entries (3 x K x 20).  RMSE is reported in units of
c (C_ref = 3 M, so 1e-3 in c is 3 mM).
Gradient: jax reverse mode through simulate(); integrate.py applies the implicit-function theorem at every
step, so one evaluation = one forward solve + one adjoint solve per current.
Optimizer: scipy L-BFGS-B on theta in the box [0, 1]^8 (theta is already normalized, see solver/properties.py),
on the mse divided by (1e-4 mean c)^2 so that its stopping tests are relative to the data scale.
During the fit the [1e-4, 2] concentration window of the data generator is not enforced: a trial theta that
depletes c below zero (common for a generic start at the highest current) still gives a finite, smooth misfit
that steers the search back.  Only a Newton failure or a non-finite value on an observed frame counts as
invalid; it returns a large loss with zero gradient, so the line search backs off.

Starts:
  center      theta = 0.5 everywhere (input B: no learned start, the same for every material)
  sobol<i>    i-th scrambled Sobol point (seed 0), as in inverse/baselines.py (input B)
  mlp, fno    supervised network prediction, seed --net-seed (input A)
  mlp_ttt, fno_ttt   the same after 150 TTT steps (pinn/ttt.py, learning rate from select_<arch>.json)

Cost is logged per evaluation (wall time and objective), so evaluations and wall time to reach a target RMSE
of 1 %, 0.1 % and 0.01 % of mean c can be read off afterwards, as for the optimizer baselines.
Output: <out>/m<k>.json per material and <out>/summary.json (rebuilt from every material file).
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
from scipy.optimize import minimize  # noqa: E402
from scipy.stats import qmc  # noqa: E402

from solver.dataset import N_CELLS, TAU, W_BINS  # noqa: E402
from solver.fv import uniform_grid  # noqa: E402
from solver.integrate import Protocol, SolverOptions, simulate  # noqa: E402
from solver.properties import RHO_COEFF, BernsteinMaterial  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
NET_DIR = REPO / "results" / "vanilla pinn"          # saved MLP / FNO weights and TTT learning rates
N_TEST = 300
TARGETS_REL = (1e-2, 1e-3, 1e-4)                     # target RMSE as a fraction of mean observed c
BAD_LOSS = 1.0                                       # mse returned when a trial simulation fails
LOSS_SCALE_REL = 1e-4                                # L-BFGS sees mse / (1e-4 mean c)^2, so 1 = the 0.01 % target
FIT_OPTIONS = SolverOptions(domain_c_min=-np.inf, domain_c_max=np.inf)


# ----------------------------------------------------------------------------------------------- forward model
def _forward(theta, iotas, opts=FIT_OPTIONS):
    """Binned c (3, 13, 20) and frame validity (3, 13) for theta (8,) and the material's 3 currents.
    With FIT_OPTIONS, validity means Newton converged; SolverOptions() gives the generator's validity."""
    grid = uniform_grid(N_CELLS)
    model = BernsteinMaterial(jnp.asarray(RHO_COEFF))
    W = jnp.asarray(W_BINS)

    def one(io):
        c, info = simulate(theta, Protocol(model=model, c0=jnp.ones(N_CELLS), t_obs=jnp.asarray(TAU), iota=io),
                           grid, opts)
        return c @ W.T, info["valid_obs"]
    return jax.vmap(one)(iotas)


def _loss_parts(theta, y, mask, iotas):
    pred, valid = _forward(theta, iotas)
    m = mask[None, :, None]
    r = jnp.where(m, pred - jnp.nan_to_num(y), 0.0)
    mse = jnp.sum(r ** 2) / (jnp.sum(mask) * y.shape[0] * y.shape[2])
    ok = jnp.all(jnp.where(mask[None, :], valid, True)) & jnp.isfinite(mse)
    return mse, ok


@jax.jit
def loss_and_grad(theta, y, mask, iotas):
    """(mse, d mse / d theta, ok) for one material; y (3, 13, 20) with NaN on unobserved frames, mask (13,)."""
    (mse, ok), g = jax.value_and_grad(_loss_parts, has_aux=True)(theta, y, mask, iotas)
    return mse, g, ok


@jax.jit
def loss_only(theta, y, mask, iotas):
    return _loss_parts(theta, y, mask, iotas)


# ----------------------------------------------------------------------------------------------------- data
def load_test(n=N_TEST):
    d = np.load(DATA / "train.npz")
    split = np.load(DATA / "train_split.npy")
    s = np.flatnonzero(split == "id_test")[:n]
    return dict(y=d["y"][s], iotas=d["iotas"][s], theta=d["theta"][s], window=d["window"][s], index=d["index"][s])


def starts(kind, T, net_seed=0):
    """Starting theta (n, 8) for every test material."""
    n = len(T["theta"])
    if kind == "center":
        return np.full((n, 8), 0.5)
    if kind.startswith("sobol"):
        z = qmc.Sobol(d=8, scramble=True, seed=0).random(8)[int(kind[5:])]
        return np.repeat(z[None], n, 0)
    import torch
    from surrogate.data import load
    from surrogate.models import build, input_scales, predict
    arch, ttt = kind.split("_")[0], kind.endswith("_ttt")
    S = load(("train", "id_test"))
    te = S["id_test"]
    assert np.allclose(te["theta"], T["theta"][:len(te["theta"])]), "network test split differs from the fit's"
    net = build(arch, input_scales(S["train"]["yv"], S["train"]["it"]), net_seed)
    net.load_state_dict(torch.load(NET_DIR / f"{arch}_s{net_seed}" / "model.pt"))
    if not ttt:
        return predict(net, te).numpy().astype(float)[:n]
    from pinn.pde_loss import residual_scale
    from pinn.ttt import STEPS, ttt as run_ttt
    lr = json.loads((NET_DIR / "ttt" / f"select_{arch}.json").read_text())["lr"]
    scale = residual_scale(S["train"]["pde"])
    return np.array([run_ttt(net, te, i, lr, scale)[STEPS]["theta"] for i in range(n)])


# -------------------------------------------------------------------------------------------------- fitting
def fit(theta0, y, iotas, max_iter=500, max_wall_s=float("inf")):
    """L-BFGS-B from theta0. Returns the result dict, with one log row per evaluation."""
    mask = np.isfinite(y[0, :, 0])
    scale = (LOSS_SCALE_REL * float(np.nanmean(y))) ** 2       # scipy's ftol test is absolute when |f| < 1
    args = (jnp.asarray(y), jnp.asarray(mask), jnp.asarray(iotas, dtype=jnp.float64))
    log, t0 = [], time.perf_counter()
    best = [np.inf, np.asarray(theta0, float)]

    class _Stop(Exception):
        pass

    def fun(th):
        if time.perf_counter() - t0 > max_wall_s:
            raise _Stop
        th = np.clip(th, 0.0, 1.0)
        mse, g, ok = loss_and_grad(jnp.asarray(th), *args)
        mse, g, ok = float(mse), np.asarray(g, float), bool(ok)
        if not ok or not np.all(np.isfinite(g)):
            mse, g = BAD_LOSS, np.zeros(8)
        if mse < best[0]:
            best[0], best[1] = mse, th.copy()
        log.append([time.perf_counter() - t0, float(np.sqrt(mse)), float(np.sqrt(best[0])), int(ok)])
        return mse / scale, g / scale

    status = "converged"
    try:
        r = minimize(fun, np.clip(theta0, 0.0, 1.0), jac=True, method="L-BFGS-B", bounds=[(0.0, 1.0)] * 8,
                     options=dict(maxiter=max_iter, maxfun=4 * max_iter, ftol=1e-12, gtol=1e-9, maxcor=20))
        status = "converged" if r.success else str(r.message)
        n_iter = int(r.nit)
    except _Stop:
        status, n_iter = "used_walltime", None
    return dict(theta=best[1].tolist(), rmse=float(np.sqrt(best[0])), evals=len(log), iters=n_iter,
                wall_s=time.perf_counter() - t0, status=status, log=log)


def to_target(log, target):
    """(evaluations, wall seconds) at the first evaluation whose best RMSE <= target, else (None, None)."""
    for k, (w, _, b, _) in enumerate(log):
        if b <= target:
            return k + 1, w
    return None, None


def summarize(out: Path):
    from analysis.metrics import score_all
    T = load_test()
    rows = sorted((json.loads(p.read_text()) for p in out.glob("m*.json")), key=lambda r: r["k"])
    if not rows:
        return {}
    ks = [r["k"] for r in rows]
    sc = score_all(np.array([r["theta"] for r in rows]), T["theta"][ks], T["window"][ks])
    sc0 = score_all(np.array([r["theta0"] for r in rows]), T["theta"][ks], T["window"][ks])
    med = lambda a: float(np.median([x for x in a if x is not None])) if any(x is not None for x in a) else None  # noqa: E731
    summ = dict(n=len(rows), start=rows[0]["start"],
                median_start={k: float(np.median(v)) for k, v in sc0.items()},
                median_fit={k: float(np.median(v)) for k, v in sc.items()},
                median_rmse_c=float(np.median([r["rmse"] for r in rows])),
                median_evals=float(np.median([r["evals"] for r in rows])),
                median_wall_s=float(np.median([r["wall_s"] for r in rows])),
                status={s: sum(r["status"] == s for r in rows) for s in {r["status"] for r in rows}})
    for rel in TARGETS_REL:
        hits = [r["to_target"][str(rel)] for r in rows]
        summ[f"target_{rel:g}"] = dict(reached=sum(h[0] is not None for h in hits),
                                       median_evals=med([h[0] for h in hits]), median_wall_s=med([h[1] for h in hits]))
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    return summ


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", required=True, help="center | sobol<i> | mlp | fno | mlp_ttt | fno_ttt")
    p.add_argument("--net-seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--materials", type=str, default=f"0:{N_TEST}", help="slice of the 300 test materials, a:b")
    p.add_argument("--shard", type=str, default="0/1", help="i/n: this process fits materials k with k % n == i")
    p.add_argument("--max-iter", type=int, default=500)
    p.add_argument("--max-wall-s", type=float, default=600.0, help="wall-time cap per material (s)")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    T = load_test()
    lo, hi = map(int, a.materials.split(":"))
    i, n = map(int, a.shard.split("/"))
    todo = [k for k in range(lo, hi) if k % n == i and not (a.out / f"m{k:03d}.json").exists()]
    cache = a.out / "starts.npy"                         # computed once, shared by parallel shards
    if cache.exists():
        th0_all = np.load(cache)
    else:
        th0_all = starts(a.start, T, a.net_seed)
        np.save(cache, th0_all)
    if not todo:
        print(json.dumps(summarize(a.out), indent=1))
        return
    loss_and_grad(jnp.asarray(th0_all[todo[0]]), jnp.asarray(T["y"][todo[0]]),
                  jnp.asarray(np.isfinite(T["y"][todo[0], 0, :, 0])), jnp.asarray(T["iotas"][todo[0]]))  # compile
    for k in todo:
        y = T["y"][k]
        r = fit(th0_all[k], y, T["iotas"][k], a.max_iter, a.max_wall_s)
        mean_c = float(np.nanmean(y))
        r.update(k=k, index=int(T["index"][k]), start=a.start, theta0=th0_all[k].tolist(), mean_c=mean_c,
                 to_target={str(rel): to_target(r["log"], rel * mean_c) for rel in TARGETS_REL})
        (a.out / f"m{k:03d}.json").write_text(json.dumps(r))
        print(f"{a.start} m{k:03d}: {r['status']}, {r['evals']} evals, {r['wall_s']:.1f} s, "
              f"rmse {r['rmse']:.2e} (rel {r['rmse'] / mean_c:.1e})", flush=True)
    print(json.dumps(summarize(a.out), indent=1))


if __name__ == "__main__":
    main()
