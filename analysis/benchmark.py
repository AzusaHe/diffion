"""Summaries and publication figures for optimizer benchmark results.

A results folder holds one `<METHOD>_start<i>.json` summary and one `<METHOD>_start<i>.csv`
per-evaluation log per run, as written by inverse/run_baselines.py.
"""
import csv
import json
from pathlib import Path

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter

from inverse.baselines import LABELS, METHODS
from solver import transport_1d as model

C_EVAL = jnp.linspace(0.70, 1.35, 200)        # concentration range covered by the data (M)
COLORS = {"NM": "#2a78d6", "CMAES": "#eb6834", "BO": "#1baf7a"}

STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 14, "axes.labelsize": 14, "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 12,
    "axes.linewidth": 1.0, "xtick.direction": "in", "ytick.direction": "in", "ytick.right": True,
    "xtick.major.size": 5, "ytick.major.size": 5, "ytick.minor.size": 3,
    "xtick.major.width": 1.0, "ytick.major.width": 1.0, "ytick.minor.width": 0.8,
    "svg.fonttype": "none", "pdf.fonttype": 42,
}


def load_runs(folder):
    """Run summaries grouped by method, each list sorted by start index."""
    folder = Path(folder)
    return {m: [json.loads(p.read_text()) for p in sorted(folder.glob(f"{m}_start*.json"))] for m in METHODS}


def property_mse(theta):
    """Mean-squared error of D(c) and t+(c) against the true curves over C_EVAL."""
    D, tp = model.properties(C_EVAL, jnp.asarray(theta))
    D_true, tp_true = model.properties(C_EVAL, jnp.asarray(model.THETA_TRUE))
    return float(jnp.mean((D - D_true) ** 2)), float(jnp.mean((tp - tp_true) ** 2))


def write_summary(folder):
    """Write summary.csv (one row per run) into the results folder."""
    folder = Path(folder)
    fields = ["method", "start", "status", "evals_to_target", "wall_s_to_target", "total_evals",
              "total_wall_s", "total_solver_s", "best_rmse_M", "mse_D", "mse_tplus"]
    with open(folder / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for runs in load_runs(folder).values():
            for r in runs:
                mse_D, mse_t = property_mse(r["theta_best"])
                w.writerow({**r, "mse_D": mse_D, "mse_tplus": mse_t})


def _cost_panel(ax, runs_by_method, to_target_key, total_key):
    for x, m in enumerate(METHODS):
        runs = runs_by_method[m]
        reached = np.array([r[to_target_key] for r in runs if r["reached_target"]], float)
        full = np.array([r[total_key] for r in runs if not r["reached_target"]], float)
        if reached.size:
            ax.bar(x, reached.mean(), width=0.62, color=COLORS[m], edgecolor="black", linewidth=1.0, zorder=2)
            if reached.size > 1:
                ax.errorbar(x, reached.mean(), yerr=reached.std(ddof=1), fmt="none", ecolor="black",
                            elinewidth=1.2, capsize=5, capthick=1.2, zorder=3)
        else:
            ax.bar(x, full.mean(), width=0.62, facecolor="white", edgecolor=COLORS[m], hatch="///",
                   linewidth=1.2, zorder=2)
        _scatter_runs(ax, x, reached, full)
        ax.text(x, 1.015, f"{reached.size}/{len(runs)}", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=12)


def _error_panel(ax, runs_by_method, index):
    for x, m in enumerate(METHODS):
        runs = runs_by_method[m]
        values = np.array([property_mse(r["theta_best"])[index] for r in runs])
        mask = np.array([r["reached_target"] for r in runs])
        ax.bar(x, values.mean(), width=0.62, color=COLORS[m], edgecolor="black", linewidth=1.0, zorder=2)
        _scatter_runs(ax, x, values[mask], values[~mask])
    ax.set_yscale("log")
    lo, hi = _log_limits([property_mse(r["theta_best"])[index] for runs in runs_by_method.values() for r in runs])
    ax.set_ylim(lo, hi)
    _log_minor(ax)


def _scatter_runs(ax, x, reached, full):
    n = reached.size + full.size
    xs = x + (np.linspace(-0.13, 0.13, n) if n > 1 else np.zeros(n))
    ax.scatter(xs[:reached.size], reached, s=30, facecolor="white", edgecolor="black", linewidth=1.0, zorder=4)
    ax.scatter(xs[reached.size:], full, s=46, marker="x", color="black", linewidth=1.3, zorder=4)


def _log_limits(values):
    values = np.asarray(values, float)
    return 10 ** np.floor(np.log10(values.min() / 2)), 10 ** np.ceil(np.log10(values.max() * 2))


def _log_minor(ax):
    ax.yaxis.set_minor_locator(LogLocator(subs=np.arange(2, 10)))
    ax.yaxis.set_minor_formatter(NullFormatter())


def plot_benchmark(folder, out_stem, log_walltime):
    """Four-panel figure: (a) wall time and (b) evaluations to target, (c) MSE in D and (d) MSE in t+.

    Bars in (a)-(b) are mean +/- standard deviation over runs that reached the target; a hatched bar is
    the mean budget used when every run used its full budget. Bars in (c)-(d) average all runs.
    Circles mark runs that reached the target, crosses mark runs that used their full budget.
    Writes out_stem + .png (600 dpi), .pdf and .svg.
    """
    runs = load_runs(folder)
    with plt.rc_context(STYLE):
        fig, ((a, b), (c, d)) = plt.subplots(2, 2, figsize=(8.0, 7.0))
        _cost_panel(a, runs, "wall_s_to_target", "total_wall_s")
        _cost_panel(b, runs, "evals_to_target", "total_evals")
        _error_panel(c, runs, 0)
        _error_panel(d, runs, 1)

        walls = [r["wall_s_to_target"] if r["reached_target"] else r["total_wall_s"] for rs in runs.values() for r in rs]
        if log_walltime:
            a.set_yscale("log")
            a.set_ylim(*_log_limits(walls))
            _log_minor(a)
        else:
            a.set_ylim(0, 1.15 * max(walls))
            a.yaxis.set_minor_locator(AutoMinorLocator())
        evals = [r["evals_to_target"] if r["reached_target"] else r["total_evals"] for rs in runs.values() for r in rs]
        b.set_ylim(0, 1.15 * max(evals))
        b.yaxis.set_minor_locator(AutoMinorLocator())

        a.set_ylabel("Wall time to target (s)")
        b.set_ylabel("Function evaluations to target")
        c.set_ylabel(r"MSE in $D$ at best fit")
        d.set_ylabel(r"MSE in $t_+$ at best fit")
        for ax, label in zip((a, b, c, d), ("(a)", "(b)", "(c)", "(d)")):
            ax.set_xticks(range(len(METHODS)), [LABELS[m] for m in METHODS])
            ax.set_xlim(-0.6, len(METHODS) - 0.4)
            ax.tick_params(axis="x", length=0)
            ax.text(-0.25, 1.0, label, transform=ax.transAxes, va="bottom", ha="left")

        handles = [Line2D([], [], ls="none", marker="o", mfc="white", mec="black", ms=6, label="Run reached target")]
        if any(not r["reached_target"] for rs in runs.values() for r in rs):
            handles.append(Line2D([], [], ls="none", marker="x", color="black", ms=7, mew=1.3,
                                  label="Run used full budget"))
        if any(not any(r["reached_target"] for r in rs) for rs in runs.values()):
            handles.append(Patch(facecolor="white", edgecolor="black", hatch="///", label="Mean budget used"))
        fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False,
                   bbox_to_anchor=(0.5, 1.04), handletextpad=0.3, columnspacing=1.2)
        fig.tight_layout(w_pad=2.5, h_pad=2.0)
        out_stem = Path(out_stem)
        out_stem.parent.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf", "svg"):
            fig.savefig(out_stem.parent / f"{out_stem.name}.{ext}", dpi=600, bbox_inches="tight")
        plt.close(fig)
