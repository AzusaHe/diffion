"""Plot the paired single-current and three-current diffusivity results."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent / "random"


def main():
    results = json.loads((ROOT / "learned/comparison.json").read_text())["results"]
    colors = ["#16736e", "#ac5328"]
    fig, (overall, by_rank) = plt.subplots(1, 2, figsize=(9.0, 3.1), layout="constrained")

    def pair_plot(ax, labels, left, right, legend):
        y = np.arange(len(labels))
        ax.hlines(y, left, right, color="#a7b7b8", linewidth=1.8, zorder=2)
        ax.scatter(left, y, s=46, color=colors[0], marker="o", label=legend[0], zorder=3)
        ax.scatter(right, y, s=51, facecolors="white", edgecolors=colors[1],
                   linewidths=1.8, marker="s", label=legend[1], zorder=3)
        ax.set_yticks(y, labels)
        ax.set_ylim(len(labels)-0.4, -0.6)
        ax.set_xlim(0, 10)
        ax.set_xticks([0, 2, 4, 6, 8, 10])
        ax.set_xlabel("Median diffusivity error (%)")
        ax.spines[["top","left","right"]].set_visible(False)
        ax.spines["bottom"].set_color("#b7c4c4")
        ax.grid(axis="x", color="#e1e9e7")
        ax.tick_params(labelsize=9, length=0, pad=4)
        ax.set_axisbelow(True)
        ax.legend(loc="upper left", bbox_to_anchor=(-0.01,-0.28), ncol=2,
                  frameon=False, fontsize=8.5, handletextpad=0.2, columnspacing=0.7)

    methods = ["mlp_supervised","mlp_ttt","fno_supervised","fno_ttt"]
    series = [results[m]["D_relL2_pct"] for m in methods]
    pair_plot(overall, ["MLP", "MLP + TTT", "FNO", "FNO + TTT"],
              [r["three_current_original_window"] for r in series],
              [r["one_current_observed_window"] for r in series],
              ["Three currents", "One current"])
    overall.set_title("A    Each protocol’s observed range", loc="left", fontsize=11, fontweight="bold")

    fno = results["fno_ttt"]["D_relL2_pct"]
    pair_plot(by_rank, ["Low (87)", "Middle (112)", "High (101)"],
              [fno["three_current_observed_window_by_rank"][str(i)] for i in range(3)],
              [fno["one_current_observed_window_by_rank"][str(i)] for i in range(3)],
              ["Three currents", "One current"])
    by_rank.set_title("B    FNO + TTT by assigned current", loc="left", fontsize=11, fontweight="bold")
    fig.savefig(ROOT / "report_data/diffusivity_comparison.png", dpi=220, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
