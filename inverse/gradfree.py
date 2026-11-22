"""Gradient-free fits (NM, CMA-ES, BO) of the 300 test materials on the full model, input A or input B.

    python -m inverse.gradfree --start fno_ttt --methods NM CMAES --out "results/gradfree/A_fno_ttt"   (input A)
    python -m inverse.gradfree --start center  --methods NM CMAES --out "results/gradfree/B_center"    (input B)

Same materials, forward model and starts as inverse/lbfgs.py, so the results compare directly with the L-BFGS
fits: the full model in solver/integrate.py at the generator's settings, on the first 300 'id_test' materials.
The optimizers and their settings are the ones of the compact-model baselines (inverse/baselines.py):
NM with an adaptive simplex of side 0.1, CMA-ES with sigma0 = 0.25, BO with 16 initial points and EI.  Every
method searches theta in [0, 1]^8 and calls the loss = RMSE of binned c over the observed entries (units of c).
A trial theta whose simulation fails (Newton failure or non-finite c) scores 1.0.

A run ends at the first of: RMSE <= 0.01 % of mean c (the tightest target), the evaluation budget, the wall-time
cap, or the optimizer's own stopping rule.  Evaluations and wall time to 1 %, 0.1 % and 0.01 % are read from
the per-evaluation log.  Starts: see inverse/lbfgs.py (center, sobol<i>, mlp, fno, mlp_ttt, fno_ttt); the
starting thetas are cached in <out>/starts.npy (shared with parallel shards).
Output: <out>/<METHOD>/m<k>.json per material and <out>/<METHOD>/summary.json.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from inverse import baselines
from inverse.lbfgs import N_TEST, TARGETS_REL, load_test, loss_only, starts  # also sets single-thread XLA

import jax.numpy as jnp  # noqa: E402

BAD_RMSE = 1.0
LOG_AT = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000)      # evaluations at which best RMSE is kept


def make_loss(y, iotas):
    args = (jnp.asarray(y), jnp.asarray(np.isfinite(y[0, :, 0])), jnp.asarray(iotas, dtype=jnp.float64))

    def loss(z):
        mse, ok = loss_only(jnp.asarray(np.asarray(z, float)), *args)
        mse = float(mse)
        return float(np.sqrt(mse)) if bool(ok) and np.isfinite(mse) else BAD_RMSE
    return loss


def fit_one(method, theta0, y, iotas, seed, budget, max_wall_s):
    mean_c = float(np.nanmean(y))
    summary, rows = baselines.run(method, make_loss(y, iotas), theta0, seed=seed, target=TARGETS_REL[-1] * mean_c,
                                  budget=budget, max_wall_s=max_wall_s, to_theta=lambda z: z)
    best, wall = rows[:, 4], rows[:, 1]
    to_target = {}
    for rel in TARGETS_REL:
        hit = np.flatnonzero(best <= rel * mean_c)
        to_target[str(rel)] = [int(hit[0] + 1), float(wall[hit[0]])] if hit.size else [None, None]
    return dict(method=method, status=summary["status"], evals=int(len(rows)), wall_s=float(wall[-1]),
                solver_s=summary["total_solver_s"], rmse=summary["best_rmse_M"], mean_c=mean_c,
                theta=summary["theta_best"], to_target=to_target,
                best_at={str(n): float(best[min(n, len(best)) - 1]) for n in LOG_AT if n <= len(best)})


def summarize(out: Path):
    from analysis.metrics import score_all
    T = load_test()
    rows = sorted((json.loads(p.read_text()) for p in out.glob("m*.json")), key=lambda r: r["k"])
    if not rows:
        return {}
    ks = [r["k"] for r in rows]
    sc = score_all(np.array([r["theta"] for r in rows]), T["theta"][ks], T["window"][ks])

    def med(a):
        a = [x for x in a if x is not None]
        return float(np.median(a)) if a else None
    summ = dict(n=len(rows), start=rows[0]["start"], method=rows[0]["method"],
                median_fit={k: float(np.median(v)) for k, v in sc.items()},
                median_rel_rmse=float(np.median([r["rmse"] / r["mean_c"] for r in rows])),
                median_evals=med([r["evals"] for r in rows]), median_wall_s=med([r["wall_s"] for r in rows]),
                status={s: sum(r["status"] == s for r in rows) for s in sorted({r["status"] for r in rows})})
    for rel in TARGETS_REL:
        hits = [r["to_target"][str(rel)] for r in rows]
        summ[f"target_{rel:g}"] = dict(reached=sum(h[0] is not None for h in hits),
                                       median_evals=med([h[0] for h in hits]), median_wall_s=med([h[1] for h in hits]))
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    return summ


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", required=True)
    p.add_argument("--net-seed", type=int, default=0)
    p.add_argument("--methods", nargs="+", default=["NM", "CMAES"], choices=baselines.METHODS)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--materials", type=str, default=f"0:{N_TEST}", help="slice of the 300 test materials, a:b")
    p.add_argument("--shard", type=str, default="0/1", help="i/n: this process fits materials k with k % n == i")
    p.add_argument("--budget", type=int, default=2000, help="evaluations per fit")
    p.add_argument("--max-wall-s", type=float, default=300.0, help="wall-time cap per fit (s)")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    T = load_test()
    cache = a.out / "starts.npy"
    lb_cache = Path("results/lbfgs") / (("B_" if a.start in ("center",) or a.start.startswith("sobol") else "A_")
                                        + a.start) / "starts.npy"
    if not cache.exists():
        np.save(cache, np.load(lb_cache) if lb_cache.exists() else starts(a.start, T, a.net_seed))
    th0_all = np.load(cache)
    lo, hi = map(int, a.materials.split(":"))
    i, n = map(int, a.shard.split("/"))
    for method in a.methods:
        mdir = a.out / method
        mdir.mkdir(exist_ok=True)
        todo = [k for k in range(lo, hi) if k % n == i and not (mdir / f"m{k:03d}.json").exists()]
        if todo:
            make_loss(T["y"][todo[0]], T["iotas"][todo[0]])(th0_all[todo[0]])              # compile
        for k in todo:
            t = time.perf_counter()
            r = fit_one(method, th0_all[k], T["y"][k], T["iotas"][k], seed=k, budget=a.budget, max_wall_s=a.max_wall_s)
            r.update(k=k, index=int(T["index"][k]), start=a.start, theta0=th0_all[k].tolist())
            (mdir / f"m{k:03d}.json").write_text(json.dumps(r))
            print(f"{a.start} {method} m{k:03d}: {r['status']}, {r['evals']} evals, {time.perf_counter() - t:.1f} s, "
                  f"rel rmse {r['rmse'] / r['mean_c']:.1e}", flush=True)
        print(json.dumps(summarize(mdir), indent=1))


if __name__ == "__main__":
    main()
