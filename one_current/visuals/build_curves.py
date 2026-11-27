"""Export saved transport-property coefficients for the offline case explorer.

Run after build_cases with python -m one_current.visuals.build_curves.
The coefficients are stored at full precision so even the direct solver's
small differences from truth are retained.
"""
import json
from collections import defaultdict

import numpy as np

from one_current.score_results import curves, groups
from one_current.visuals.build_cases import RANDOM, REPO, cohorts, read_csv, read_json


def original_coefficients(ids):
    history = {}
    networks = REPO / "results/vanilla pinn"
    for arch in ("mlp", "fno"):
        for seed in range(3):
            test = read_json(networks / f"{arch}_s{seed}/result.json")["test"]
            history[f"{arch}_supervised", str(seed)] = np.asarray(test["theta"])
    for method, directory, n, coefficient_key in cohorts():
        if method == "fno_ttt_pilot":
            continue
        records = [read_json(path) for path in sorted(directory.glob("m[0-9][0-9][0-9].json"))]
        np.testing.assert_array_equal([r["index"] for r in records], ids[:n])
        history[method, "fixed"] = np.asarray([r[coefficient_key] for r in records])
        if method in ("lbfgs_mlp_ttt", "lbfgs_fno_ttt"):
            # The fit records preserve the seed-0 TTT estimate as their start.
            # TTT archives for seeds 1 and 2 contain metrics without coefficients.
            history[method.removeprefix("lbfgs_"), "0"] = np.asarray([r["theta0"] for r in records])
    return history


def validate(z, truth, bounds, targets, prefix, window):
    if z.shape != truth.shape or np.any(~np.isfinite(z)) or np.any((z < 0) | (z > 1)):
        raise ValueError("Expected a bounded, finite eight-coefficient estimate per case.")
    observed = curves(z, truth, bounds)
    expected = [[float(r[key]) for r in targets] for key in (f"D_{prefix}_{window}_pct", f"t_{prefix}_{window}_rmse")]
    np.testing.assert_allclose(observed, expected, rtol=1e-8, atol=1e-12)


def main():
    out = RANDOM / "visuals"
    paired = defaultdict(dict)
    for r in read_csv(out / "paired_cases.csv"):
        paired[r["method"], r["seed"] or "fixed"][int(r["material_id"])] = r
    selection = np.flatnonzero(np.load(REPO / "data/train_split.npy") == "id_test")[:300]
    with np.load(REPO / "data/train.npz") as original, np.load(RANDOM / "data/train.npz") as one:
        ids = original["index"][selection]
        truth = original["theta"][selection]
        common = original["window"][selection]
        iotas = one["iotas"][selection, 0]
    old = original_coefficients(ids)
    fits = defaultdict(dict)
    one_count = three_count = 0
    for label, case_ids, one_z in groups(iotas):
        method, marker, seed = label.rpartition("_s")
        if not marker or seed not in ("0", "1", "2"):
            method, seed = label, "fixed"
        n = len(case_ids)
        np.testing.assert_array_equal(case_ids, ids[:n])
        targets = [paired[method, seed][i] for i in case_ids]
        one_bounds = np.array([[r["one_window_min"],r["one_window_max"]] for r in targets],dtype=float)
        three_bounds = np.array([[r["three_window_min"],r["three_window_max"]] for r in targets],dtype=float)
        validate(one_z, truth[:n], one_bounds, targets, "one", "observed")
        validate(one_z, truth[:n], common[:n], targets, "one", "common")
        three_z = old.get((method, seed))
        if three_z is not None:
            validate(three_z, truth[:n], three_bounds, targets, "three", "observed")
            other = three_z.tolist()
            three_count += n
        else:
            if method not in ("fno_ttt", "mlp_ttt") or seed not in ("1", "2"):
                raise ValueError(f"Missing archived coefficients for {method}, seed {seed}")
            other = [None] * n
        fits[method][seed] = {int(i): [z1, z3] for i, z1, z3 in zip(case_ids, one_z.tolist(), other)}
        one_count += n
    assert (len(fits), one_count, three_count) == (34, 6512, 5312)
    payload = {"truth": {int(i): z for i, z in zip(ids, truth.tolist())}, "fits": fits}
    path = out / "curve_coefficients.json"
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    temp.replace(path)
    print(f"Verified saved property curves for {one_count} single-current and {three_count} three-current estimates.")


if __name__ == "__main__":
    main()
