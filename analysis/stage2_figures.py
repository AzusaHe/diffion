"""Progress-report figures A1-A6 (one or two messages each).

    python -m analysis.stage2_figures

  A1 errors           MSE of ln D and t+ per material, supervised vs after TTT
  A2 error_reduction  RMS error ratio supervised / TTT for ln D, t+, ln[(1 - t+)/D]
  A3 per_material     ln D error of every material, after TTT vs before
  A4 training_curves  training and validation coefficient MSE vs epoch
  A5 wall_time        validation coefficient MSE vs training wall time
  A6 ttt_steps        ln D error and PDE loss vs TTT step

Style: Times New Roman (Liberation Serif fallback), 14 pt, ticks in on all four sides; PDF + 300-dpi PNG
in results/stage2/figures/.
"""
from __future__ import annotations

import matplotlib
import matplotlib.ticker as mt
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .stage2_load import ARCHS, NAME, RESULTS, SEEDS, Results  # noqa: E402

OUT = RESULTS / "figures"
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#6b6b6b"
COL = {"mlp": BLUE, "fno": ORANGE}
MODELS = [("mlp", False), ("mlp", True), ("fno", False), ("fno", True)]
XPOS = np.array([0, 0.85, 2.15, 3.0])

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "TeX Gyre Termes", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 14, "axes.labelsize": 14, "xtick.labelsize": 13, "ytick.labelsize": 13,
    "legend.fontsize": 12, "axes.linewidth": 1.0, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True, "xtick.major.size": 5, "ytick.major.size": 5, "xtick.minor.size": 2.5,
    "ytick.minor.size": 2.5, "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    "pdf.fonttype": 42})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf"); fig.savefig(OUT / f"{name}.png", dpi=300); plt.close(fig)


def log_y(ax, subs=None):
    ax.set_yscale("log")
    if subs is not None:
        ax.yaxis.set_major_locator(mt.LogLocator(base=10, subs=subs))
    ax.yaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(mt.NullFormatter())


def panel_label(ax, letter):
    ax.text(-0.2, 1.02, f"({letter})", transform=ax.transAxes, fontsize=15, fontweight="bold", va="bottom")


def boxes(ax, R, metric, ylabel):
    """Box = 25-75 %, whiskers = 10-90 % of all seeds x materials; thick line = mean of per-seed medians."""
    for x, (a, ttt) in zip(XPOS, MODELS):
        q10, q25, q75, q90 = np.quantile(R.pooled(a, ttt, metric), [0.1, 0.25, 0.75, 0.9])
        ax.add_patch(plt.Rectangle((x - 0.25, q25), 0.5, q75 - q25, facecolor=COL[a] if ttt else "white",
                                   alpha=0.25 if ttt else 1, edgecolor=COL[a], lw=1.2))
        ax.vlines(x, q10, q25, color=COL[a], lw=1.2); ax.vlines(x, q75, q90, color=COL[a], lw=1.2)
        ax.hlines([q10, q90], x - 0.1, x + 0.1, color=COL[a], lw=1.2)
        ax.hlines(R.seed_stat(a, ttt, metric)[0], x - 0.25, x + 0.25, color=COL[a], lw=2.4)
    ax.set_xticks(XPOS); ax.set_xticklabels(["Sup.", "+TTT", "Sup.", "+TTT"])
    ax.set_xlim(-0.55, 3.55); ax.tick_params(axis="x", which="both", top=False)
    for a, xc in (("mlp", 0.425), ("fno", 2.575)):
        ax.text(xc, -0.13, NAME[a], transform=ax.get_xaxis_transform(), ha="center", va="top", color=COL[a],
                fontsize=14, fontweight="bold")
    log_y(ax); ax.set_ylabel(ylabel)


def fig_errors(R):
    fig, axs = plt.subplots(1, 2, figsize=(9.2, 4.0))
    boxes(axs[0], R, "mse_lnD", r"MSE of $\ln D$")
    boxes(axs[1], R, "mse_tplus", r"MSE of $t_+$")
    for ax, l in zip(axs, "ab"):
        panel_label(ax, l)
    fig.subplots_adjust(wspace=0.38, bottom=0.2)
    save(fig, "A1_errors")


REDUCTION = [("mse_lnD", r"$\ln D$", np.sqrt), ("mse_tplus", r"$t_+$", np.sqrt),
             ("group_rmse", r"$\ln[(1-t_+)/D]$", lambda v: v)]


def error_reduction(R, arch, metric, f):
    """Per seed: median RMS error supervised / median RMS error after TTT."""
    return [np.median(f(R.per_material(arch, False, s, metric))) / np.median(f(R.per_material(arch, True, s, metric)))
            for s in SEEDS]


def fig_error_reduction(R):
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    w = 0.34
    for i, (met, lab, f) in enumerate(REDUCTION):
        for j, a in enumerate(ARCHS):
            r = error_reduction(R, a, met, f)
            x = i + (j - 0.5) * w
            ax.bar(x, np.mean(r), w * 0.9, color=COL[a], label=NAME[a] if i == 0 else None)
            ax.vlines(x, min(r), max(r), color="k", lw=1.2)
    ax.axhline(1, color=GREY, lw=1, ls="--")
    ax.set_xticks(range(len(REDUCTION))); ax.set_xticklabels([q[1] for q in REDUCTION])
    ax.tick_params(axis="x", which="both", top=False, length=0)
    ax.set_ylabel("Error reduction by TTT\n(RMS error, supervised / TTT)")
    ax.set_ylim(0, 6); ax.legend(loc="upper left")
    save(fig, "A2_error_reduction")


def fig_per_material(R):
    fig, axs = plt.subplots(1, 2, figsize=(9.2, 4.4), sharex=True, sharey=True)
    lim = (1e-4, 3e-1)
    for ax, a, l in zip(axs, ARCHS, "ab"):
        b, t = R.pooled(a, False, "mse_lnD"), R.pooled(a, True, "mse_lnD")
        ax.scatter(b, t, s=9, color=COL[a], alpha=0.45, lw=0)
        ax.plot(lim, lim, color="k", lw=1); ax.plot(lim, np.array(lim) / 2, color=GREY, lw=1, ls="--")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
        ax.text(0.04, 0.96, f"{NAME[a]}\nimproved: {100 * np.mean(t < b):.0f} %\nhalved: {100 * np.mean(t < b / 2):.0f} %",
                transform=ax.transAxes, va="top", fontsize=13, color=COL[a])
        ax.text(2.2e-4, 2.2e-4 * 1.35, "no change", fontsize=12, ha="left", va="bottom", rotation=45, rotation_mode="anchor")
        ax.text(4.4e-4, 2.2e-4 / 1.35, "half", color=GREY, fontsize=12, ha="left", va="top", rotation=45, rotation_mode="anchor")
        ax.set_xlabel(r"MSE of $\ln D$, supervised")
        panel_label(ax, l)
    axs[0].set_ylabel(r"MSE of $\ln D$, after TTT")
    fig.subplots_adjust(wspace=0.12)
    save(fig, "A3_per_material")


def fig_training_curves(R):
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    for a in ARCHS:
        H = [R.sup[(a, s)]["history"][1:] for s in SEEDS]
        ep = np.array([h["epoch"] for h in H[0]])
        for key, ls in (("train_coef_mse", "--"), ("val_coef_mse", "-")):
            y = np.array([[h[key] for h in hs] for hs in H])
            ax.fill_between(ep, y.min(0), y.max(0), color=COL[a], alpha=0.2, lw=0)
            ax.plot(ep, y.mean(0), color=COL[a], ls=ls, lw=1.8)
    log_y(ax, subs=(1, 2, 3, 4, 5, 6, 7))
    ax.set_xlim(0, ep[-1]); ax.set_ylim(0.018, 0.08)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Coefficient MSE")
    h = [plt.Line2D([], [], color=COL[a], lw=2, label=NAME[a]) for a in ARCHS] + \
        [plt.Line2D([], [], color="k", lw=1.8, ls="--", label="training"), plt.Line2D([], [], color="k", lw=1.8, label="validation")]
    ax.legend(handles=h, loc="upper right", ncol=2, columnspacing=1.2)
    save(fig, "A4_training_curves")


def fig_wall_time(R):
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    for a in ARCHS:
        per_step = R.timing[a]["step_s_median"]
        for s in SEEDS:
            h = R.sup[(a, s)]["history"][1:]
            ax.plot([e["step"] * per_step / 60 for e in h], [e["val_coef_mse"] for e in h], color=COL[a], lw=1.3,
                    label=NAME[a] if s == 0 else None)
    ax.set_xscale("log"); log_y(ax, subs=(1, 2, 3, 4, 5, 6, 7))
    ax.xaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_ylim(0.02, 0.07)
    ax.set_xlabel("Training wall time (min, one CPU thread)"); ax.set_ylabel("Validation coefficient MSE")
    ax.legend(loc="upper right")
    save(fig, "A5_wall_time")


def fig_ttt_steps(R):
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 4.0))
    steps = np.array(R.checkpoints, float)
    for ax, (met, lab), l in zip(axs, (("mse_lnD", r"MSE of $\ln D$"), ("pde_loss", r"PDE loss $\mathcal{L}_{\mathrm{PDE}}$")), "ab"):
        for a in ARCHS:
            st = np.array([R.seed_stat(a, True, met, k) for k in R.checkpoints])
            ax.fill_between(steps, st[:, 1], st[:, 2], color=COL[a], alpha=0.2, lw=0)
            ax.plot(steps, st[:, 0], color=COL[a], lw=2, marker="o", ms=5, label=NAME[a])
        if met == "pde_loss":
            ax.axhline(R.truth_pde_loss, color="k", ls="--", lw=1.2)
            ax.text(155, R.truth_pde_loss * 1.08, "true coefficients", fontsize=12, va="bottom", ha="right")
        ax.set_xscale("function", functions=(lambda v: np.sign(v) * np.sqrt(np.abs(v)), lambda v: np.sign(v) * v ** 2))
        ax.set_xticks(steps); ax.set_xticklabels([str(int(s)) for s in steps])
        ax.xaxis.set_minor_locator(mt.NullLocator()); ax.set_xlim(-3, 165)
        log_y(ax, subs=(1, 2, 5)); ax.set_xlabel("TTT steps per material"); ax.set_ylabel(lab)
        panel_label(ax, l)
    axs[0].legend(loc="lower left")
    fig.subplots_adjust(wspace=0.38)
    save(fig, "A6_ttt_steps")


def main():
    R = Results()
    for f in (fig_errors, fig_error_reduction, fig_per_material, fig_training_curves, fig_wall_time, fig_ttt_steps):
        f(R)
    print("figures in", OUT)


if __name__ == "__main__":
    main()
