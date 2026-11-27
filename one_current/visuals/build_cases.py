"""Export and verify paired material scores. Run from diffion-ms with
python -m one_current.visuals.build_cases. Only NumPy and the saved runs are needed.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from one_current.learned.historical_refinements import cohorts
from one_current.score_results import curves, save_csv


REPO = Path(__file__).resolve().parents[2]
RANDOM = REPO / "one_current" / "random"
METRICS = ("D_relL2_pct", "tplus_rmse")


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def historical_predictions(ids, truth, common_bounds):
    """Return per-case historical scores in the original test-loader order."""
    result = {}
    models = REPO / "results" / "vanilla pinn"
    for arch in ("mlp", "fno"):
        for seed in range(3):
            model = f"{arch}_s{seed}"
            sup = read_json(models / model / "result.json")["test"]
            # The historical loader uses the first 300 rows of the ID test split.
            # Match the saved theta against the original truth and windows to
            # independently verify this order against both reported error arrays.
            recomputed = curves(np.asarray(sup["theta"]), truth, common_bounds)
            for key, values in zip(METRICS, recomputed):
                np.testing.assert_allclose(values, sup[key], rtol=1e-8, atol=1e-10)
            ttt = read_json(models / "ttt" / f"{model}.json")["scores"]
            for key in METRICS:
                # The TTT archive has ordinal case records; step zero is a second
                # per-case signature of the supervised checkpoint/order.
                np.testing.assert_allclose(ttt["0"][key], sup[key], rtol=5e-4, atol=1e-5)
            for name, group in (("supervised", sup), ("ttt", ttt["150"])):
                result[f"{arch}_{name}", seed] = dict(zip(ids, zip(*(group[k] for k in METRICS))))

    for method, directory, n, theta_key in cohorts():
        if method == "fno_ttt_pilot":
            continue
        files = sorted(directory.glob("m[0-9][0-9][0-9].json"))
        records = [read_json(path) for path in files]
        indexed = {row["index"]: row for row in records}
        if len(records) != n or set(indexed) != set(ids[:n]):
            raise ValueError(f"{method} has a different historical material cohort.")
        ordered = [indexed[i] for i in ids[:n]]
        np.testing.assert_array_equal([r["k"] for r in ordered], np.arange(n))
        estimates = np.asarray([r[theta_key] for r in ordered])
        d, t = curves(estimates, truth[:n], common_bounds[:n])
        result[method, None] = dict(zip(ids[:n], zip(d, t)))
    return result


def check_medians(rows):
    refs = read_csv(RANDOM / "report_data" / "comparison.csv")
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        groups[row["method"]][row["seed"]].append(row)
    if len(groups) != 34 or set(groups) != {r["method"] for r in refs} or len(rows) != 6512:
        raise ValueError("Unexpected method or material coverage.")
    metrics = {
        "D_one_observed_pct": "one_current_D_observed_pct",
        "D_one_common_pct": "one_current_D_common_pct",
        "D_three_common_pct": "three_current_D_common_pct",
        "t_one_observed_rmse": "one_current_t_observed_rmse",
        "t_one_common_rmse": "one_current_t_common_rmse",
        "t_three_common_rmse": "three_current_t_common_rmse",
    }
    for ref in refs:
        by_seed = groups[ref["method"]]
        assert len(by_seed) == int(ref["runs_per_material"])
        for cohort in by_seed.values():
            assert len({r["material_id"] for r in cohort}) == len(cohort) == int(ref["n_materials"])
        for col, source in metrics.items():
            per_seed = [np.median([r[col] for r in cohort]) for cohort in by_seed.values()]
            np.testing.assert_allclose(np.mean(per_seed), float(ref[source]), rtol=1e-8, atol=1e-12)


def main():
    split = np.load(REPO / "data/train_split.npy")
    np.testing.assert_array_equal(split, np.load(RANDOM / "data/train_split.npy"))
    test = np.flatnonzero(split == "id_test")[:300]
    with np.load(REPO / "data/train.npz") as base, np.load(RANDOM / "data/train.npz") as single:
        ids = base["index"][test]
        truth = base["theta"][test]
        orig_iotas = base["iotas"][test]
        three_bounds = base["window"][test]
        one_bounds = single["window"][test]
        np.testing.assert_array_equal(ids, single["index"][test])
        np.testing.assert_array_equal(truth, single["theta"][test])
        ranks = np.load(RANDOM / "data/current_rank.npy")[test]
        currents = single["iotas"][test, 0]
        np.testing.assert_array_equal(orig_iotas[np.arange(len(ids)), ranks], currents)
    orig = historical_predictions(ids, truth, three_bounds)
    index = {int(mid): i for i, mid in enumerate(ids)}
    out = []
    for row in read_csv(RANDOM / "report_data/materials.csv"):
        label = row["method"]
        seed = None
        if label[-3:] in {"_s0", "_s1", "_s2"}:
            seed = int(label[-1])
            label = label[:-3]
        mid = int(row["material_id"])
        i = index[mid]
        if int(row["rank"]) != ranks[i] or float(row["iota"]) != currents[i]:
            raise ValueError(f"Current mismatch for material {mid}")
        old_d, old_t = orig[label, seed][mid]
        out.append(dict(method=label, seed=seed, material_id=mid, rank=int(ranks[i]), iota=currents[i],
                        iota_low=orig_iotas[i, 0], iota_middle=orig_iotas[i, 1], iota_high=orig_iotas[i, 2],
                        one_window_min=one_bounds[i, 0], one_window_max=one_bounds[i, 1],
                        three_window_min=three_bounds[i, 0], three_window_max=three_bounds[i, 1],
                        D_one_common_pct=float(row["D_common_pct"]), D_three_common_pct=old_d,
                        t_one_common_rmse=float(row["t_common_rmse"]), t_three_common_rmse=old_t,
                        D_one_observed_pct=float(row["D_observed_pct"]), t_one_observed_rmse=float(row["t_observed_rmse"])))
    check_medians(out)
    folder = RANDOM / "visuals"
    folder.mkdir(parents=True, exist_ok=True)
    save_csv(folder / "paired_cases.csv", out)
    print(f"Verified {len(out)} paired estimates and all six published metrics for all 34 configurations.")


if __name__ == "__main__":
    main()
