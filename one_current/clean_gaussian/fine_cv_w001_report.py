"""Compare additive c/v corrections from the same saved 375-update base."""
import hashlib
import json

import numpy as np

if __package__:
    from .fine_cv_report import as_csv, build_fine_cv
    from .fine_cv_cohort_report import METRICS, RESULTS
else:
    from fine_cv_report import as_csv, build_fine_cv
    from fine_cv_cohort_report import METRICS, RESULTS

FOLDER = "fine_cv_correction_w001_results"
METHOD = "correction_w001"
STEM = "fine_cv_300_w001"
MATCHED_FIELDS = (
    "ids", "current_indices", "valid_frames", "lo", "hi", "theta0", "theta375",
    "c_rmse0", "v_rmse0", "c_rmse375", "v_rmse375", "base375_hash",
    "correction_initial_hash", "base_execution", "correction_setup_s", "reused_from_eight",
)


def merge_weight_csv(original, added):
    """Join the original and paired numeric CSVs while preserving their stored precision."""
    tables = [text.rstrip("\r\n").splitlines() for text in (original, added)]
    headers = [table[0].split(",") for table in tables]
    fields = list(dict.fromkeys(headers[0] + headers[1]))
    lines = [",".join(fields)]
    weight = {"standard": "1", "correction_w1": "1", "correction_w100": "100", METHOD: "0.01"}
    for table, header in zip(tables, headers):
        for line in table[1:]:
            cells = line.split(",")
            if len(cells) != len(header) or '"' in line:
                raise ValueError("Unexpected quoting in numeric c/v export.")
            row = dict(zip(header, cells))
            row.setdefault("observation_weight", weight[row["method"]])
            row.setdefault("total_PINN_updates", row["new_PINN_updates"])
            lines.append(",".join(row.get(field, "") for field in fields))
    return "\r\n".join(lines) + "\r\n"


def native_audit(folder):
    path = folder / "raw_field_validation.json"
    if not path.is_file():
        return None
    audit = json.loads(path.read_text(encoding="utf-8"))
    assert audit["domains"] == ["clean"]
    assert [audit[k] for k in ("materials", "contexts", "pipelines", "endpoint_records")] == [300, 1200, 1200, 2400]
    expected = {row["path"]: row["sha256"] for row in audit["scored_archives"]}
    current = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.glob("clean__*__*.npz")}
    if len(current) != 4 or expected != current:
        raise ValueError("The new correction audit does not match the four clean archives.")
    return {k: audit[k] for k in ("contexts", "pipelines", "endpoint_records", "max_c_rmse_abs", "max_v_rmse_abs")}


def build_w001_report(root, payload, errors, preview=False):
    previous = payload.get("fine_cv_300")
    if not previous:
        return None
    study = build_fine_cv(root, payload["materials"], payload["runs"], errors, payload["equal_budget"],
                          preview=True, result_dir=FOLDER, cohort_size=300, methods=(METHOD,), domains=("clean",))
    if not study:
        return None
    assert study["protocol"]["checkpoint_steps"] == [0, 375, 500]
    assert study["protocol"]["field_widths"] == [3, 44, 47, 46, 1]
    assert study["protocol"]["correction_weights"] == [0.01]
    assert study["ids"] == previous["ids"] and len(study["rows"]) == 4
    truth = np.array([m["truth"] for m in payload["materials"][:300]])
    material_rows = []
    for row in study["rows"]:
        d, n, start = row["domain"], row["currents"], row["start"]
        key = f"{d}__{n}__{METHOD}_{start}"
        cases = study["cases"][key]
        with np.load(root / FOLDER / (key + ".npz")) as z:
            assert np.all(z["updates375"] == 375) and np.all(z["total_updates"] == 500)
            for field, value in (("correction_weight", 0.01), ("new_updates", 125), ("reused_base_updates", 375)):
                np.testing.assert_array_equal(z[field], np.full(300, value))
            assert (row["field_prefit_updates"], row["joint_updates"], row["correction_updates"]) == (236, 139, 125)
            shared, setup, branch = (z[k] for k in ("shared_stage1_s", "correction_setup_s", "stage2_s"))
            np.testing.assert_allclose(z["stage1_s"], shared + setup, rtol=1.e-7)
            np.testing.assert_allclose(z["elapsed_s"], shared + setup + branch, rtol=1.e-7)
            row.update(observation_weight=0.01, new_PINN_updates=125, total_PINN_updates=500,
                       reused_base_updates=375, new_material_branches=300, reused_final_branches=0,
                       stage1_mean_s=float(shared.mean()), stage2_mean_s=float((setup + branch).mean()),
                       new_125_mean_s=float(branch.mean()), new_125_total_s=float(branch.sum()))
            modes = {k: z[k].astype(str) for k in ("base_execution", "final_execution")}
            base_hashes, init_hashes, reused = (z[k] for k in ("base375_hash", "correction_initial_hash", "reused_from_eight"))
            for phase, values in modes.items():
                assert set(values) <= {"separate_graph", "reused_graph"}
                for graph in ("separate_graph", "reused_graph"):
                    row[f"n_{phase.split('_')[0]}_{graph}"] = int((values == graph).sum())
            for i, v in enumerate(cases):
                v.update(domain=d, currents=n, method=METHOD, start=start, observation_weight=0.01,
                         fit_origin="new_correction_existing_base", base_origin="prior_eight" if reused[i] else "new_292",
                         base_execution=modes["base_execution"][i], final_execution=modes["final_execution"][i],
                         total_PINN_updates=500, new_PINN_updates=125, reused_base_updates=375,
                         field_prefit_updates=236, joint_updates=139, correction_updates=125,
                         stage1_s=shared[i], stage2_s=setup[i]+branch[i], new_125_s=branch[i],
                         shared_once_PINN_s_contribution=branch[i],
                         base375_hash=base_hashes[i], correction_initial_hash=init_hashes[i])
            for m in METRICS:
                end = np.array([v[f"{m}_500"] for v in cases])
                row[f"p25_{m}_500"], row[f"p75_{m}_500"] = np.percentile(end, [25, 75])
                for phase in (0, 375):
                    delta = end - [v[f"{m}_{phase}"] for v in cases]
                    row[f"n_lower_{m}_500_vs_{phase}"] = int((delta < 0).sum())
                    row[f"median_paired_delta_{m}_500_vs_{phase}"] = float(np.median(delta))
            for reference in ("w1", "w100"):
                old_key = f"{d}__{n}__correction_{reference}_{start}"
                with np.load(root / RESULTS / (old_key + ".npz")) as old:
                    for field in MATCHED_FIELDS:
                        np.testing.assert_array_equal(z[field], old[field], err_msg=f"{key} vs {reference}: {field}")
                    np.testing.assert_array_equal(shared, old["stage1_s"])
                    de, te = errors(old["theta"], truth, z["lo"], z["hi"])
                    old_branch = old["stage2_s"] - old["correction_setup_s"]
                    old_elapsed = old["elapsed_s"]
                    assert (old_branch > 0).all()
                    row[f"{reference}_125_mean_s"] = float(old_branch.mean())
                    row[f"{reference}_pipeline_mean_s"] = row["archived_start_mean_s"] + float(old_elapsed.mean())
                    for i, v in enumerate(cases):
                        v[f"{reference}_125_s"] = old_branch[i]
                        v[f"{reference}_pipeline_s"] = row["archived_start_mean_s"] + old_elapsed[i]
                    for metric, historical in zip(METRICS, (de, te, old["c_rmse"], old["v_rmse"])):
                        delta = np.array([v[f"{metric}_500"] for v in cases]) - historical
                        row[f"{reference}_median_{metric}_500"] = float(np.median(historical))
                        row[f"median_paired_delta_{metric}_vs_{reference}"] = float(np.median(delta))
                        row[f"n_lower_{metric}_vs_{reference}"] = int((delta < 0).sum())
                        row[f"n_equal_{metric}_vs_{reference}"] = int((delta == 0).sum())
                        for v, value, diff in zip(cases, historical, delta):
                            v[f"{reference}_{metric}_500"] = value
                            v[f"paired_delta_{metric}_vs_{reference}"] = diff
        material_rows.extend(cases)
    assert len(material_rows) == 1200
    summary, individual = as_csv(study["rows"]), as_csv(material_rows)
    if '"' in individual:
        raise ValueError("Material CSV contains an unexpected quoted field.")
    verified = native_audit(root / FOLDER)
    if not preview:
        for suffix, prior, exported in (("accuracy_cost", previous["summary_csv"], summary),
                                        ("materials", previous["case_csv"], individual)):
            (root / f"{STEM}_{suffix}.csv").write_bytes(exported.encode("utf-8"))
            (root / f"fine_cv_300_all_weights_{suffix}.csv").write_bytes(merge_weight_csv(prior, exported).encode("utf-8"))
    return dict(rows=study["rows"], ids=study["ids"], complete_domains=["clean"], protocol=study["protocol"], native_audit=verified,
                summary_csv=summary, case_csv=individual,
                accuracy_csv_append=study["accuracy_csv_append"].replace(",grid290_cv_", ",grid290_cv_wide500_n300_"))
