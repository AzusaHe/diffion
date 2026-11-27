"""Summarize predictions on each protocol's observed concentration interval.

Use --run-root one_current/random from the diffion-ms directory. A reference-window
score uses the original three-current concentration interval for both protocols.
The one-current networks never receive that reference window at inference.
"""
from pathlib import Path
import argparse
import json

import numpy as np

from run import REPO, score_all, write_json

ARCHIVE = REPO / "results" / "vanilla pinn"
METRICS = ("D_relL2_pct", "tplus_rmse", "group_rmse", "mse_lnD", "mse_tplus")


def summarize(arrays, case_mask):
    values = np.asarray(arrays)[:, case_mask]
    return float(np.median(values, axis=1).mean())


def main(protocol, architectures):
    learned = protocol / "learned"
    data_dir = protocol / "data"
    split = np.load(data_dir / "train_split.npy")
    test = np.flatnonzero(split == "id_test")[:300]
    with np.load(data_dir / "train.npz") as data:
        indices = data["index"][test]
        truth = data["theta"][test]
        windows = data["window"][test]
    with np.load(REPO / "data" / "train.npz") as original:
        if not np.array_equal(original["index"][test], indices):
            raise ValueError("Material IDs differ")
        reference_windows = original["window"][test]
    ranks_path = data_dir / "current_rank.npy"
    ranks = np.load(ranks_path)[test] if ranks_path.exists() else np.full(len(test), 2)
    draws = np.random.default_rng(819).integers(0, len(test), size=(2000, len(test)))
    all_cases = np.ones(len(test), dtype=bool)
    results = {}
    for arch in architectures:
        for phase in ("supervised", "ttt"):
            observed = []
            reference = []
            old = []
            for seed in range(3):
                with np.load(learned / f"{arch}_s{seed}" / f"{phase}.npz") as saved:
                    if not np.array_equal(saved["index"], indices):
                        raise ValueError("Prediction IDs differ")
                    observed.append(score_all(saved["theta"], truth, windows))
                    reference.append(score_all(saved["theta"], truth, reference_windows))
                if phase == "supervised":
                    old.append(json.loads((ARCHIVE / f"{arch}_s{seed}" / "result.json").read_text())["test"])
                else:
                    old.append(json.loads((ARCHIVE / "ttt" / f"{arch}_s{seed}.json").read_text())["scores"]["150"])
            row = {}
            for metric in METRICS:
                o = np.asarray([r[metric] for r in observed])
                a = np.asarray([r[metric] for r in reference])
                b = np.asarray([r[metric] for r in old])
                historical_medians = np.median(b[:, draws], axis=2)
                delta = (np.median(a[:, draws], axis=2) - historical_medians).mean(0)
                observed_delta = (np.median(o[:, draws], axis=2) - historical_medians).mean(0)
                row[metric] = {
                    "one_current_observed_window": summarize(o, all_cases),
                    "one_current_original_window": summarize(a, all_cases),
                    "three_current_original_window": summarize(b, all_cases),
                    "one_minus_three_observed_window_95pct_material_bootstrap": np.quantile(observed_delta,[0.025,0.975]).tolist(),
                    "one_minus_three_original_window_95pct_material_bootstrap": np.quantile(delta,[0.025,0.975]).tolist(),
                    "three_current_observed_window_by_rank": {
                        str(rank): summarize(b, ranks == rank) for rank in np.unique(ranks)
                    },
                    "one_current_observed_window_by_rank": {
                        str(rank): summarize(o, ranks == rank) for rank in np.unique(ranks)
                    },
                    "one_current_original_window_by_rank": {
                        str(rank): summarize(a, ranks == rank) for rank in np.unique(ranks)
                    },
                }
            results[f"{arch}_{phase}"] = row
    filename = "comparison.json" if set(architectures) == {"fno", "mlp"} else f"comparison_{'_'.join(architectures)}.json"
    write_json(learned / filename, {
        "n_materials": len(indices), "n_seeds": 3, "aggregation": "mean of the three seed-specific medians",
        "bootstrap_draws": 2000, "bootstrap_seed": 819, "matched_indices": True,
        "different_windows": int(np.any(windows != reference_windows, axis=1).sum()),
        "median_observed_span": {"one":float(np.median(windows[:,1]-windows[:,0])),
                                 "three":float(np.median(reference_windows[:,1]-reference_windows[:,0]))},
        "rank_counts": {str(rank): int((ranks == rank).sum()) for rank in np.unique(ranks)}, "results": results,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--arch", nargs="+", choices=["mlp","fno"], default=["mlp","fno"])
    args=parser.parse_args()
    main(args.run_root.resolve(), args.arch)
