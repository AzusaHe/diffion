"""Join slope-loss fits to their matched controls on the same measured interval."""
import csv
import json

import numpy as np


PAIRED = [
    f"pinn_{method}_{start}"
    for method in ("standard_cv", "staged_cv_w1", "staged_cv_w100", "data_first", "frozen", "fine")
    for start in ("fno_ttt", "center")
] + [f"pinn_{m}_fno_ttt" for m in ("data_correction", "solver_correction", "solver_correction_weight1")]


def values(bundle, key, count):
    arr = np.asarray(bundle[key], dtype=float)
    assert arr.shape == (count,) and np.isfinite(arr).all() and np.all(arr >= 0), key
    return arr.tolist()


def build_comparison(root, materials, runs, errors, label, preview=False):
    """Use the primary benchmark's scoring and verify cohort and interval pairing."""
    result_dir = root / "derivative_results"
    if not result_dir.exists():
        return None
    materials_by_id = {m["id"]: m for m in materials}
    expected_ids = [m["id"] for m in materials[:8]]
    rows = []
    summary = []
    for domain in ("clean", "gaussian_noisy"):
        for count in (1, 3):
            for slug in PAIRED:
                stem = f"{domain}__{count}__{slug}"
                fit = result_dir / stem
                ref = root / "derivative_controls" / stem
                if preview and not (fit.with_suffix(".json").exists() and ref.with_suffix(".json").exists()):
                    continue
                control_metadata = json.loads(ref.with_suffix(".json").read_text())
                fit_metadata = json.loads(fit.with_suffix(".json").read_text())
                schedules = []
                for metadata, weight in ((control_metadata, 0), (fit_metadata, 1)):
                    schedule = metadata["schedule"].copy()
                    assert schedule.pop("spatial_derivative_weight") == metadata["spatial_derivative_weight"] == weight
                    schedules.append(schedule)
                assert schedules[0] == schedules[1] and control_metadata["weights"] == fit_metadata["weights"], stem
                selected_currents = [materials_by_id[i]["chosen"][:count] for i in expected_ids]
                assert control_metadata["current_indices"] == fit_metadata["current_indices"] == selected_currents, stem
                elapsed = [metadata["test_s"] for metadata in (control_metadata, fit_metadata)]
                assert all(np.isfinite(t) and t > 0 for t in elapsed)
                with np.load(fit.with_suffix(".npz")) as exp, np.load(ref.with_suffix(".npz")) as base:
                    ids, low, high = (base[k] for k in ("ids", "lo", "hi"))
                    n = len(ids)
                    assert ids.tolist() == expected_ids, stem
                    for key in ("ids", "lo", "hi"):
                        np.testing.assert_array_equal(base[key], exp[key], err_msg=stem)
                    for source in (exp, base):
                        assert source["theta"].shape == (n, 8) and np.isfinite(source["theta"]).all(), stem
                        assert np.all((source["theta"] >= 0) & (source["theta"] <= 1)), stem
                    truth = [materials_by_id[int(i)]["truth"] for i in ids]
                    d0, t0 = errors(base["theta"], truth, low, high)
                    d1, t1 = errors(exp["theta"], truth, low, high)
                    c1, v1 = (values(exp, key, n) for key in ("c_slope_rmse", "v_slope_rmse"))
                    c0 = values(base, "c_slope_rmse", n)
                    v0 = values(base, "v_slope_rmse", n)
                    field_metrics = {
                        f"{key}{arm}": values(source, code, n)
                        for arm, source in enumerate((base, exp))
                        for key, code in (("c_value", "rel_rmse"), ("v_value", "v_rel_rmse"))
                    }
                    np.testing.assert_array_equal(base["steps"], exp["steps"], err_msg=stem)
                    archive_diff = None
                    if stem in runs:
                        previous = np.asarray(runs[stem]["observations"])
                        np.testing.assert_array_equal(previous[:, 0], ids)
                        np.testing.assert_array_equal(previous[:, 3], low)
                        np.testing.assert_array_equal(previous[:, 4], high)
                        archive_diff = [float(np.max(np.abs(previous[:, i] - err))) for i, err in ((1, d0), (2, t0))]
                    per_case = dict(lo=low, hi=high, d0=d0, d1=d1, t0=t0, t1=t1,
                                    c0=c0, c1=c1, v0=v0, v1=v1, **field_metrics)
                    cases = [dict(id=int(mid), **{key: float(metric[i]) for key, metric in per_case.items()})
                             for i, mid in enumerate(ids)]
                start = "fno_ttt" if slug.endswith("_fno_ttt") else "center"
                name = "Two-stage · frozen fields" if slug.startswith("pinn_frozen_") else label(slug).split(" (")[0]
                warm_time = 0.0
                theta_initial = np.full((n, 8), 0.5)
                if start == "fno_ttt":
                    warm = runs[f"{domain}__{count}__fno_ttt"]
                    warm_time = warm["per_s"]
                    warm_rows = np.asarray(warm["observations"][:n])
                    np.testing.assert_array_equal(warm_rows[:, 0], ids)
                    theta_initial = np.clip(warm_rows[:, 5:], 1e-5, 1 - 1e-5)
                d_initial, t_initial = errors(theta_initial, truth, low, high)
                perf = dict(
                    D_initial_pct=float(np.median(d_initial)), t_initial_rmse=float(np.median(t_initial)),
                    D_control_pct=float(np.median(d0)), D_derivative_pct=float(np.median(d1)),
                    D_paired_delta_pp=float(np.median(d1 - d0)), D_improved=int(sum(d1 < d0)),
                    t_control_rmse=float(np.median(t0)), t_derivative_rmse=float(np.median(t1)),
                    t_paired_delta_rmse=float(np.median(t1 - t0)), t_improved=int(sum(t1 < t0)),
                    control_s_per_material=elapsed[0] / n + warm_time,
                    derivative_s_per_material=elapsed[1] / n + warm_time,
                    c_slope_control_rmse=float(np.median(c0)),
                    c_slope_derivative_rmse=float(np.median(c1)),
                    v_slope_control_rmse=float(np.median(v0)),
                    v_slope_derivative_rmse=float(np.median(v1)),
                )
                perf.update({key + "_median": float(np.median(metric)) for key, metric in field_metrics.items()})
                setting = dict(domain=domain, currents=count, method=slug, start=start, n=n)
                rows.append(dict(**setting, name=name, cases=cases, archive_diff=archive_diff, **perf))
                summary.append(dict(**setting, **perf))
    physics_file = result_dir / "fixed_field_physics.json"
    physics = json.loads(physics_file.read_text())["rows"] if physics_file.exists() else []
    if not preview:
        assert len(rows) == 4 * len(PAIRED)
        assert len(physics) == 8 * len(expected_ids)
        for domain in ("clean", "gaussian_noisy"):
            for count in (1, 3):
                for weight in (0, 1):
                    selected = [r for r in physics if (r["domain"], r["currents"], r["derivative_weight"]) == (domain, count, weight)]
                    assert sorted(r["material_id"] for r in selected) == expected_ids
                    assert all(np.isfinite(r[k]) and r[k] >= 0 for r in selected for k in ("fitted", "truth"))
        with (root / "derivative_accuracy_cost.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
    return dict(conditions=rows, physics=physics)
