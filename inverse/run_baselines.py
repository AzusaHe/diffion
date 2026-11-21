"""Run derivative-free baselines (NM, CMA-ES, BO) on the noise-free synthetic dataset.

Each (method, start) pair writes <out>/<METHOD>_start<i>.json (run summary) and .csv (one row per
evaluation), then <out>/summary.csv is rebuilt from every run in the folder.

Run from the repository root, one method at a time. Example (target RMSE = 0.01% of mean c):
    python -m inverse.run_baselines --target-rel 1e-4 --budget 2000 --max-wall-s 1800 \
        --methods NM --out results/baselines/target_0.01pct
"""
import argparse
import json
from pathlib import Path

import numpy as np

from analysis.benchmark import write_summary
from inverse import baselines
from solver import transport_1d as model


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target-rel", type=float, required=True, help="target RMSE as a fraction of mean c")
    p.add_argument("--budget", type=int, default=2000, help="function evaluations per run")
    p.add_argument("--max-wall-s", type=float, default=float("inf"), help="wall-time cap per run (s)")
    p.add_argument("--methods", nargs="+", default=list(baselines.METHODS), choices=baselines.METHODS)
    p.add_argument("--starts", nargs="+", type=int, default=list(range(5)))
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    observed = model.synthetic_data()
    target = args.target_rel * float(np.mean(observed))
    loss = lambda z: model.loss(z, observed).block_until_ready()
    loss(np.full(8, 0.5))                                   # compile once, outside the timed runs
    starts = baselines.sobol_starts(max(args.starts) + 1)

    for method in args.methods:
        for i in args.starts:
            summary, rows = baselines.run(method, loss, starts[i], seed=i, target=target, budget=args.budget,
                                          max_wall_s=args.max_wall_s, to_theta=model.theta_from_z)
            summary = {"start": i, "target_rel": args.target_rel, **summary}
            name = f"{method}_start{i}"
            (args.out / f"{name}.json").write_text(json.dumps(summary, indent=1))
            np.savetxt(args.out / f"{name}.csv", rows, delimiter=",",
                       header=",".join(baselines.CSV_COLUMNS), comments="")
            print(f"{method} start {i}: {summary['status']}, {summary['total_evals']} evals, "
                  f"{summary['total_wall_s']:.1f} s, best RMSE {1e3 * summary['best_rmse_M']:.3f} mM", flush=True)
    write_summary(args.out)


if __name__ == "__main__":
    main()
