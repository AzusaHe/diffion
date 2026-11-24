"""Table of the PINN tuning tests in results/pinn_tune (run by results/pinn_tune/run_all.sh), next to the
data-first baselines of results/pinn_prefit.  Same 8 test materials, w_data 1e-2, physics weights 1.

    python -m pinn.tune_table            # writes results/pinn_tune/table.md

Test 1 (start at the true theta) also reports the physics loss (w_pde L_pde + w_vol L_vol + w_bc L_bc) at the
truth, at the first logged step of stage 2, and at the end: if the fit lowers it while moving away from the
truth, the minimum of the PINN loss is not at the true D and t+.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pinn.prefit_sweep import row

OUT = Path("results/pinn_tune")
PRE = Path("results/pinn_prefit")
RUNS = [("baseline: data-first joint", PRE / "wd1e-2"),
        ("baseline: c, v frozen (theta only)", PRE / "frozen_cv_wd1e-2"),
        ("1 truth start, c, v frozen", OUT / "truth_frozen_cv"),
        ("1 truth start, joint", OUT / "truth_joint"),
        ("2 smoothness w 1e-2, joint", OUT / "smooth_w1e-2"),
        ("3 FNO prior w 0.1, joint", OUT / "prior_w0.1"),
        ("3 FNO prior w 1, joint", OUT / "prior_w1"),
        ("3 FNO prior w 10, joint", OUT / "prior_w10"),
        ("4 hard wall-flux BC, joint", OUT / "hardbc")]


def phys(e, w):
    return sum(w[k] * e[f"L_{k}"] for k in ("pde", "vol", "bc") if f"L_{k}" in e)


def truth_losses(d: Path):
    """Median physics loss at the start of stage 2 (theta = truth) and at the end."""
    a, b = [], []
    for f in sorted(d.glob("m*.json")):
        r = json.loads(f.read_text())
        L = [e for e in r["log"] if "L_pde" in e and e["stage"] in ("joint", "lbfgs", "end")]
        if L:
            a.append(phys(L[0], r["weights"]))
            b.append(phys(L[-1], r["weights"]))
    return (float(np.median(a)), float(np.median(b))) if a else (float("nan"), float("nan"))


def main():
    lines = ["| test | input | D err % (start -> fit) | better | t+ rmse | c misfit % | v misfit % |",
             "|---|---|---|---|---|---|---|"]
    for name, d in RUNS:
        for st in sorted(p.name for p in d.glob("[AB]_*")) if d.exists() else []:
            r = row(d / st / "cv")
            if r is None:
                continue
            inp = "truth" if st == "A_truth" else st[0]
            lines.append(f"| {name} | {inp} | {r['D0']:.1f} -> {r['D']:.1f} | {r['better']}/{r['n']} | {r['tp']:.3f} | "
                         f"{100 * r['c']:.3f} | {100 * r['v']:.2f} |")
    lines += ["", "Test 1, physics loss (w_pde L_pde + w_vol L_vol + w_bc L_bc), median over materials:", "",
              "| run | at the true theta (stage-2 start) | at the end |", "|---|---|---|"]
    for name, d in RUNS[2:4]:
        a, b = truth_losses(d / "A_truth" / "cv")
        lines.append(f"| {name} | {a:.4g} | {b:.4g} |")
    txt = "\n".join(lines)
    (OUT / "table.md").write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
