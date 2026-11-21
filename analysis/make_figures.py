"""Rebuild summary.csv and the four-panel benchmark figure for every target in results/baselines/.

Run from the repository root:
    python -m analysis.make_figures
"""
from pathlib import Path

from analysis.benchmark import plot_benchmark, write_summary

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results" / "baselines"
FIGURES = RESULTS / "figures"
TARGETS = {                     # results folder -> wall-time axis on a log scale
    "target_0.01pct": True,
    "target_0.1pct": True,
    "target_1pct": False,
}

if __name__ == "__main__":
    for name, log_walltime in TARGETS.items():
        write_summary(RESULTS / name)
        plot_benchmark(RESULTS / name, FIGURES / f"benchmark_{name}", log_walltime)
        print("wrote", name)
