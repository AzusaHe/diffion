"""Figure and table: total-loss PINNs (Build 1 'cv', Build 2 'c') from input A (fno_ttt) and input B (center),
with L-BFGS and Nelder-Mead on the same materials for reference.

    python -m analysis.pinn_figures                 -> results/pinn/figures/pinn_results.png + markdown table
    python -m analysis.pinn_figures --root <dir>    (another results folder laid out as results/pinn)
"""
import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analysis.metrics import score  # noqa: E402
from inverse.lbfgs import load_test  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
BLUE, ORANGE, GRAY, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#8a8a85", "#1f1f1e", "#6b6a64", "#e6e5e0"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})
V = [("A_fno_ttt/cv", "Build 1, input A", BLUE, "-"), ("A_fno_ttt/c", "Build 2, input A", ORANGE, "-"),
     ("B_center/cv", "Build 1, input B", BLUE, ":"), ("B_center/c", "Build 2, input B", ORANGE, ":")]


def load(root):
    return {d: sorted([json.loads(p.read_text()) for p in (root / d).glob("m*.json")], key=lambda r: r["k"])
            for d, *_ in V}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=REPO / "results" / "pinn")
    a = p.parse_args()
    R = load(a.root)
    T = load_test()
    ks = sorted({r["k"] for rows in R.values() for r in rows})
    D = lambda th, k: score(th, T["theta"][k], T["window"][k])  # noqa: E731

    def ref(path):
        rows = [json.loads((REPO / path / f"m{k:03d}.json").read_text()) for k in ks]
        return [D(r["theta"], r["k"]) for r in rows], [r["wall_s"] for r in rows], rows

    refs = {"L-BFGS, input A": ref("results/lbfgs/A_fno_ttt"), "L-BFGS, input B": ref("results/lbfgs/B_center"),
            "NM, input A": ref("results/gradfree/A_fno_ttt/NM"), "NM, input B": ref("results/gradfree/B_center/NM")}

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw=dict(width_ratios=[1.15, 1.15, 1.2]))
    table = []
    for d, lab, col, ls in V:
        rows = R[d]
        if not rows:
            continue
        steps = np.array([l["step"] for l in rows[0]["log"]])
        Dt = np.array([[D(l["theta"], r["k"])["D_relL2_pct"] for l in r["log"]] for r in rows])
        E = np.array([[l["rel_rmse"] for l in r["log"]] for r in rows])
        for y, axi in ((Dt, ax[0]), (E, ax[1])):
            for row in y:
                axi.plot(steps, row, color=col, ls=ls, lw=0.8, alpha=0.18)
            axi.plot(steps, np.median(y, 0), color=col, ls=ls, lw=2, label=lab)
        end = [D(r["theta"], r["k"]) for r in rows]
        start = [D(r["theta0"], r["k"]) for r in rows]
        wall = [r["wall_s"] / r.get("batch", 1) for r in rows]       # GPU runs train `batch` fits at once
        table.append((lab, start, end, [r["rel_rmse"] for r in rows], wall, rows[0]["steps"]))

    a0, a1, a2 = ax
    a0.set(xlabel="Adam step", ylabel="D(c) error (%)", title="D(c) error during training (median, thin: each material)")
    a0.legend(frameon=False, fontsize=8.5)
    for t in (1e-2, 1e-3):
        a1.axhline(t, color=GRID, lw=1, zorder=0)
        a1.text(a1.get_xlim()[1], t * 1.12, f"{t * 100:g} %", color=MUTED, fontsize=8.5, ha="right")
    a1.set(yscale="log", xlabel="Adam step", ylabel="RMSE of binned c / mean c", title="Data misfit during training")

    colors = {v[1]: v[2] for v in V}
    groups = [(lab, [e["D_relL2_pct"] for e in end], colors[lab]) for lab, start, end, *_ in table]
    groups += [(k, [e["D_relL2_pct"] for e in v[0]], GRAY) for k, v in refs.items()]
    for i, (lab, vals, col) in enumerate(groups):
        x = i + np.linspace(-0.18, 0.18, len(vals))
        a2.scatter(x, vals, s=22, color=col, edgecolor="#fcfcfb", linewidth=1, zorder=3)
        a2.plot([i - 0.3, i + 0.3], [np.median(vals)] * 2, color=INK, lw=1.5, zorder=4)
    for inp in ("input A", "input B"):
        start = next((t[1] for t in table if t[0].endswith(inp)), None)
        if start:
            m = np.median([s["D_relL2_pct"] for s in start])
            a2.axhline(m, color=GRID, lw=1, ls="--", zorder=0)
            a2.text(len(groups) - 0.5, m, f"{inp} start (median)", color=MUTED, fontsize=8, ha="right", va="bottom")
    a2.set_xticks(range(len(groups)), [g[0].replace(", ", "\n") for g in groups], fontsize=8)
    a2.set(ylabel="D(c) error at the end (%)", title=f"Final D(c) error, {len(ks)} materials (bar: median)")
    for axi in ax:
        axi.grid(axis="y", color=GRID, lw=0.8)
        axi.set_axisbelow(True)
    fig.tight_layout()
    out = a.root / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "pinn_results.png", dpi=160)

    med = lambda x: float(np.median(x))  # noqa: E731
    print(f"Materials k = {ks}\n")
    print("| Method | D(c) error at start (%) | D(c) error at end (%) | t+ RMSE at end | D better than start | "
          "data RMSE / mean c | wall time per fit (s)* |")
    print("|---|---|---|---|---|---|---|")
    for lab, start, end, rmse, wall, steps in table:
        better = sum(e["D_relL2_pct"] < s["D_relL2_pct"] for e, s in zip(end, start))
        print(f"| PINN {lab} ({steps} steps) | {med([s['D_relL2_pct'] for s in start]):.2f} | "
              f"{med([e['D_relL2_pct'] for e in end]):.2f} | {med([e['tplus_rmse'] for e in end]):.3f} | "
              f"{better}/{len(end)} | {med(rmse):.1e} | {med(wall):.0f} |")
    for lab, (sc, wall, rows) in refs.items():
        start = [D(r["theta0"], r["k"]) for r in rows]
        better = sum(e["D_relL2_pct"] < s["D_relL2_pct"] for e, s in zip(sc, start))
        print(f"| {lab} | {med([s['D_relL2_pct'] for s in start]):.2f} | {med([e['D_relL2_pct'] for e in sc]):.2f} | "
              f"{med([e['tplus_rmse'] for e in sc]):.3f} | {better}/{len(sc)} | "
              f"{med([r['rmse'] / r['mean_c'] for r in rows]):.1e} | {med(wall):.1f} |")
    print("\n*GPU PINN runs: batch wall time / fits in the batch.  CPU runs: wall time of one fit while all fits ran "
          "in parallel, one thread each.")


if __name__ == "__main__":
    main()
