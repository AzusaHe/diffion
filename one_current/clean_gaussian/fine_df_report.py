"""Score saved 290-bin forward-only inversions on their own observed ranges."""
import csv
import io
import json
from collections import Counter

import numpy as np


DOMAINS = ("clean", "gaussian_noisy")
METHODS = ("nm", "cmaes", "bo")
STARTS = ("center", "fno", "fno_ttt", "mlp", "mlp_ttt")
STOPS = ("reached_target", "used_budget", "used_walltime", "optimizer_converged")


def build_fine_df(root, materials, bank, errors, equal_budget, preview):
    directory = root / "fine_df_results"
    conditions = []
    all_accuracy = io.StringIO(newline="")
    accuracy_writer = csv.DictWriter(all_accuracy, fieldnames=next(csv.reader(io.StringIO(equal_budget["csv"]))))
    cohorts = {}
    before = {}
    after = {}
    ids = np.array([m["id"] for m in materials[:300]])
    matched = np.isin(ids, equal_budget["shared_ids"])
    true_theta = np.array([m["truth"] for m in materials[:300]])
    chosen = np.array([m["chosen"] for m in materials[:300]])
    if not directory.exists():
        return None
    for domain in DOMAINS:
        for count in (1, 3):
            cohort = f"{domain}__{count}"
            for method in METHODS:
                for start in STARTS:
                    stem = directory / f"{cohort}__{method}_{start}"
                    if not (stem.with_suffix(".npz").exists() and stem.with_suffix(".json").exists()):
                        continue
                    key = stem.name
                    prior_key = f"{cohort}__{start}"
                    prior = bank.get(prior_key)
                    md = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
                    with np.load(stem.with_suffix(".npz"), allow_pickle=False) as z:
                        np.testing.assert_array_equal(z["ids"], ids)
                        np.testing.assert_array_equal(z["current_indices"], chosen[:, :count])
                        theta, theta0 = z["theta"], z["theta0"]
                        expected = np.full_like(true_theta, 0.5) if start == "center" else np.clip(
                            np.array([row[5:] for row in prior["observations"][:len(ids)]]), 1.e-6, 1.-1.e-6)
                        np.testing.assert_allclose(theta0, expected, rtol=0., atol=1.e-10)
                        if theta.shape != true_theta.shape or not np.all(np.isfinite(theta)) or np.any((theta < 0) | (theta > 1)):
                            raise ValueError(f"Invalid final coefficients for {key}")
                        lo, hi = z["lo"], z["hi"]
                        initial_d, initial_t = errors(theta0, true_theta, lo, hi)
                        final_d, final_t = errors(theta, true_theta, lo, hi)
                        initial_c, final_c = z["initial_c_rmse"], z["concentration_rmse"]
                        elapsed, evals = z["elapsed_s"], z["evals"]
                        solver_time, invalid_evals = z["solver_s"], z["invalid_evals"]
                        statuses = z["status"].astype(str).tolist()
                    n = len(ids)
                    for values in (lo, hi, initial_d, final_d, initial_t, final_t, initial_c, final_c, elapsed, evals, solver_time, invalid_evals):
                        if values.shape != (n,) or not np.isfinite(values).all():
                            raise ValueError(f"Missing or invalid material value for {key}")
                    counts = Counter(statuses)
                    budget = 35 if method == "bo" else 500
                    if (len(statuses) != n or counts.keys() - set(STOPS) or np.any(elapsed <= 0)
                            or np.any((evals < 1) | (evals > budget)) or np.any(initial_c < 0)
                            or np.any((solver_time < 0) | (solver_time > elapsed)) or np.any((invalid_evals < 0) | (invalid_evals > evals))
                            or np.any(final_c < 0) or np.any(final_c > initial_c + 1.e-12)):
                        raise ValueError(f"Inconsistent optimizer diagnostics for {key}")
                    if "statuses" in md and md["statuses"] != dict(counts):
                        raise ValueError(f"Metadata status counts disagree for {key}")
                    bounds = np.column_stack((lo, hi))
                    init_scores = np.column_stack((initial_d, initial_t, initial_c))
                    if cohort in cohorts:
                        np.testing.assert_array_equal(bounds, cohorts[cohort])
                    else:
                        cohorts[cohort] = bounds
                    if prior_key in before:
                        np.testing.assert_allclose(init_scores, before[prior_key], rtol=1.e-10, atol=1.e-13)
                    else:
                        before[prior_key] = init_scores
                    after[key] = [[float(d), float(t), float(c), float(s), int(e), STOPS.index(stop)]
                                  for d, t, c, s, e, stop in zip(final_d, final_t, final_c, elapsed, evals, statuses)]
                    init_s = prior["per_s"] if prior else 0.
                    fit_s = float(np.mean(elapsed))
                    span = np.minimum(hi, 2) - np.maximum(lo, 0)
                    conditions.append(dict(
                        domain=domain, currents=count, optimizer=method, start=start, n_test=n,
                        spatial_bins=290, max_frames_per_current=151, solver_cells=580,
                        initial_D_median_pct=float(np.median(initial_d)), refined_D_median_pct=float(np.median(final_d)),
                        initial_t_median_rmse=float(np.median(initial_t)), refined_t_median_rmse=float(np.median(final_t)),
                        initial_c_median_rmse=float(np.median(initial_c)), refined_c_median_rmse=float(np.median(final_c)),
                        D_improved_materials=int(np.count_nonzero(final_d < initial_d)),
                        t_improved_materials=int(np.count_nonzero(final_t < initial_t)),
                        c_improved_materials=int(np.count_nonzero(final_c < initial_c)),
                        median_observed_c_span=float(np.median(span)),
                        initialization_s_per_material=init_s, fit_s_per_material=fit_s,
                        pipeline_s_per_material=fit_s + init_s, fit_worker_seconds_sum=float(np.sum(elapsed)),
                        forward_objective_s_per_material=float(np.mean(solver_time)),
                        optimizer_overhead_s_per_material=float(np.mean(elapsed - solver_time)),
                        invalid_forward_evaluations=int(np.sum(invalid_evals)),
                        shared_prior_training_s=prior["training"] if prior else 0.,
                        prior_validation_search_s=prior["tuning"] if prior else 0.,
                        evaluation_limit=budget, median_evaluations=float(np.median(evals)),
                        **{stop: counts[stop] for stop in STOPS}))
                    summary = conditions[-1]
                    accuracy_writer.writerow(dict(domain=domain, currents=count, method=f"grid290_{method}_{start}",
                        family="Numerical optimization", n_test=n, D_median_pct=summary["refined_D_median_pct"],
                        t_median_rmse=summary["refined_t_median_rmse"], median_observed_c_span=summary["median_observed_c_span"],
                        matched6_D_median_pct=np.median(final_d[matched]), matched6_t_median_rmse=np.median(final_t[matched]),
                        matched6_median_observed_c_span=np.median(span[matched]), device="CPU",
                        pipeline_s_per_material=init_s + fit_s, initialization_s_per_material=init_s,
                        fit_s_per_material=fit_s, shared_training_s=summary["shared_prior_training_s"],
                        validation_search_s=summary["prior_validation_search_s"]))
    if not conditions or (not preview and len(conditions) != len(DOMAINS) * 2 * len(METHODS) * len(STARTS)):
        return None
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=conditions[0].keys())
    writer.writeheader()
    writer.writerows(conditions)
    summary_csv = stream.getvalue()
    if not preview:
        (root / "fine_df_accuracy_cost.csv").write_bytes(summary_csv.encode("utf-8"))
    checks = directory / "forward_validation.json"
    validation = json.loads(checks.read_text(encoding="utf-8")) if checks.exists() else None
    return dict(conditions=conditions, csv=summary_csv, accuracy_csv_append=all_accuracy.getvalue(),
                ids=ids.tolist(), stops=STOPS, validation=validation,
                cohorts={k: v.tolist() for k, v in cohorts.items()},
                before={k: v.tolist() for k, v in before.items()}, after=after)
