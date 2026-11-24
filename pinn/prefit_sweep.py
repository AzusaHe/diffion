"""Data-first PINN (Build 1): fit c and v to the data until converged, then turn the physics on with a small
data weight.  One run per data weight, inputs A (fno_ttt) and B (center), the same 8 test materials as
results/pinn_staged; the stage-1 fit is done once and cached.  Prints a table against the earlier staged runs.

    python -m pinn.prefit_sweep --out results/pinn_prefit --w-data 1e-2,1e-3,1e-4
    python -m pinn.prefit_sweep --out results/pinn_prefit --table-only
    python -m pinn.prefit_sweep --out results/pinn_prefit --freeze c,v --w-data 1e-2   # theta only (frozen_cv_wd1e-2)
    python -m pinn.prefit_sweep --out results/pinn_prefit --freeze c --w-data 1e-2     # v and theta (frozen_c_wd1e-2)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def row(d: Path):
    """Medians over the materials of one folder: D error, t+ rmse, c and v misfit (and at the end of stage 1)."""
    from pinn.jax_pinn import summarize
    if not list(d.glob("m*.json")):
        return None
    s = summarize(d)
    R = [json.loads(p.read_text()) for p in d.glob("m*.json")]

    def med(key):
        v = [r.get(key) for r in R]
        v = [x for x in v if x is not None and np.isfinite(x)]
        return float(np.median(v)) if v else float("nan")
    return dict(D=s["median_fit"]["D_relL2_pct"], D0=s["median_start"]["D_relL2_pct"], tp=s["median_fit"]["tplus_rmse"],
                better=s["D_err_improved"], n=s["n"], c=s["median_rel_rmse"], v=med("v_rel_rmse"),
                c_fit=med("fit_end_rel_rmse"), v_fit=med("fit_end_v_rel_rmse"), wall=s["median_wall_s"])


def table(out: Path, weights):
    runs = [("staged w100 (earlier)", Path("results/pinn_staged/w100")), ("staged w1 (earlier)", Path("results/pinn_staged/w1"))]
    runs += [(f"data-first joint w_data={w}", d) for w, d in sorted((p.name[2:], p) for p in out.glob("wd*") if p.is_dir())]
    runs += [("data-first, c and v frozen (theta only)", d) for d in sorted(out.glob("frozen_cv_wd*"))]
    runs += [(f"data-first, c frozen (v, theta) w_data={d.name.split('wd')[1]}", d) for d in sorted(out.glob("frozen_c_wd*"))]
    lines = ["| run | input | D err % (start -> fit) | better | t+ rmse | c misfit % | v misfit % | c / v misfit % after stage 1 |",
             "|---|---|---|---|---|---|---|---|"]
    for name, d in runs:
        for st in ("A_fno_ttt", "B_center"):
            r = row(d / st / "cv")
            if r is None:
                continue
            lines.append(f"| {name} | {st[0]} | {r['D0']:.1f} -> {r['D']:.1f} | {r['better']}/{r['n']} | {r['tp']:.3f} | "
                         f"{100 * r['c']:.3f} | {100 * r['v']:.2f} | {100 * r['c_fit']:.4f} / {100 * r['v_fit']:.3f} |")
    txt = "\n".join(lines)
    (out / "table.md").write_text(txt + "\n")
    print(txt, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("results/pinn_prefit"))
    p.add_argument("--w-data", default="1e-2,1e-3,1e-4")
    p.add_argument("--materials", default="0:8")
    p.add_argument("--fit-steps", type=int, default=30000)
    p.add_argument("--fit-lbfgs-iters", type=int, default=3000)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--lbfgs-iters", type=int, default=300)
    p.add_argument("--w-phys", type=float, default=1.0, help="weight on every physics term (pde, vol, bc)")
    p.add_argument("--table-only", action="store_true")
    p.add_argument("--freeze", default="", help="networks held fixed after stage 1 (c,v = theta only; c = v and theta)")
    a = p.parse_args()
    weights = a.w_data.split(",")
    a.out.mkdir(parents=True, exist_ok=True)
    if not a.table_only:
        pre = f"frozen_{a.freeze.replace(',', '')}_" if a.freeze else ""
        for w in weights:
            cmd = [sys.executable, "-m", "pinn.torch_pinn", "--build", "cv", "--starts", "fno_ttt,center",
                   "--materials", a.materials, "--out", str(a.out / f"{pre}wd{w}"), "--fit-v",
                   "--fit-steps", str(a.fit_steps), "--fit-lbfgs-iters", str(a.fit_lbfgs_iters),
                   "--prefit-cache", str(a.out / "cache"), "--steps", str(a.steps), "--lr", "1e-4",
                   "--phi-lr-mult", "10", "--lbfgs-iters", str(a.lbfgs_iters), "--w-data", w,
                   "--w-pde", str(a.w_phys), "--w-vol", str(a.w_phys), "--w-bc", str(a.w_phys)]
            if a.freeze:
                cmd += ["--freeze", a.freeze]
            print(" ".join(cmd), flush=True)
            with open(a.out / f"run_{pre}wd{w}.txt", "w") as log:
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False)
            table(a.out, weights)
    table(a.out, weights)


if __name__ == "__main__":
    main()
