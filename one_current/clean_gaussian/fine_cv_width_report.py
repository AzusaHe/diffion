"""Pair saved 400-update c/v PINN fits by material and observation condition."""
import numpy as np

if __package__:
    from .fine_cv_report import STEPS, as_csv, build_fine_cv
else:
    from fine_cv_report import STEPS, as_csv, build_fine_cv

PROPERTIES = ("D_pct", "t_rmse", "c_rmse", "v_rmse")
KEY = ("domain", "currents", "method", "start")
COSTS = ("archived_start_mean_s", "stage1_s", "stage2_s", "correction_setup_s", "PINN_s", "pipeline_s")
MATCH = ("material_id", "current_indices", "dimensionless_currents", "valid_frames_per_current",
         "c_min", "c_max", "score_min", "score_max", "stage1_updates", "stage2_updates", "new_PINN_updates",
         "archived_TTT_updates", "archived_start_mean_s", "D_pct_0", "t_rmse_0")


def add_width_view(root, html):
    start, end = "/* BEGIN CV WIDTH */", "/* END CV WIDTH */"
    script = start + "\n" + (root / "fine_cv_width_view.js").read_text(encoding="utf-8") + "\n" + end
    if start in html:
        return html[:html.index(start)] + script + html[html.index(end)+len(end):]
    def insert(anchor, replacement):
        nonlocal html
        if html.count(anchor) != 1:
            raise ValueError(f"Missing c/v width report anchor: {anchor}")
        html = html.replace(anchor, replacement)
    nav = '<button data-tab="fine-cv">290 × 151 c/v PINNs</button>'
    insert(nav, nav+'<button data-tab="cv-width">c/v network width</button>')
    insert("function groupTable(", script + "\nfunction groupTable(")
    insert('const earlier=tab==="fine-cv"', 'const earlier=(tab==="fine-cv"||tab==="cv-width")')
    insert('({accuracy:showAccuracy,','({accuracy:showAccuracy,"cv-width":showCvWidth,')
    insert('    if(tab==="fine-cv")cvBind();','    if(tab==="fine-cv")cvBind();\n    if(tab==="cv-width")widthBind();')
    csv_anchor = "DB.fine_cv.accuracy_csv_append]"
    insert(csv_anchor, "DB.fine_cv.accuracy_csv_append,DB.fine_cv_width.accuracy_csv_append]")
    return html


def build_width_comparison(root, payload, errors, preview=False):
    base = payload.get("fine_cv")
    if not base:
        return None
    wide = build_fine_cv(root, payload["materials"], payload["runs"], errors, payload["equal_budget"],
                         preview=True, result_dir="fine_cv_double_results")
    if not wide:
        return None
    assert base["protocol"]["field_widths"] == [3,32,32,32,1]
    assert wide["protocol"]["field_widths"] == [3,44,47,46,1] and wide["protocol"]["per_field_parameters"] == 4546
    np.testing.assert_array_equal(base["ids"], wide["ids"])
    if set(base["cases"]) != set(wide["cases"]):
        raise ValueError("Widths have different fit groups")
    summaries, pairs, by_case = [], [], {}
    for fit in wide["rows"]:
        config = {k: fit[k] for k in KEY}
        domain, count, method, start = (config[k] for k in KEY)
        key = f"{domain}__{count}__{method}_{start}"
        narrow_cases, wide_cases = base["cases"][key],wide["cases"][key]
        if len(narrow_cases) != 8 or len(wide_cases) != 8:
            raise ValueError(f"Expected eight matched materials in {key}")
        rows = []
        for a, b in zip(narrow_cases,wide_cases):
            for field in MATCH:
                if a[field] != b[field]:
                    raise ValueError(f"Pair differs in {field} for {key}/{a['material_id']}")
            row = {k:a[k] for k in MATCH if k not in ("D_pct_0","t_rmse_0")}
            for k in (*COSTS, *(f"{p}_{t}" for t in STEPS for p in PROPERTIES)):
                row["base_"+k], row["wide_"+k], row["delta_"+k] = a[k], b[k], b[k]-a[k]
            rows.append(row)
            pairs.append(config | row)
        by_case[key] = rows
        nets = 2 if method=="standard" else 4
        summary = config | dict(n_test=len(rows), new_PINN_updates=400, archived_TTT_updates=fit["archived_TTT_updates"],
                                base_parameters_per_field=2273, wide_parameters_per_field=4546,
                                base_final_total_NN_parameters=nets*2273, wide_final_total_NN_parameters=nets*4546,
                                base_final_trainable_including_8_logits=4554, wide_final_trainable_including_8_logits=9100)
        for t in STEPS:
            for prop in PROPERTIES:
                k = f"{prop}_{t}"
                for kind in ("base","wide","delta"):
                    summary[f"median_{kind}_{k}"] = float(np.median([v[f"{kind}_{k}"] for v in rows]))
                summary[f"delta_of_medians_{k}"] = summary[f"median_wide_{k}"] - summary[f"median_base_{k}"]
                summary[f"wider_lower_n_{k}"] = int(np.count_nonzero([v["delta_"+k] < 0 for v in rows]))
        for cost in COSTS:
            for kind in ("base","wide","delta"):
                summary[f"mean_{kind}_{cost}"] = float(np.mean([r[f"{kind}_{cost}"] for r in rows]))
        summary["wide_over_base_PINN_time"] = summary["mean_wide_PINN_s"] / summary["mean_base_PINN_s"]
        summaries.append(summary)
    csv_summary, csv_cases = as_csv(summaries), as_csv(pairs)
    if not preview:
        for filename, content in (("fine_cv_width_accuracy_cost.csv",csv_summary),("fine_cv_width_materials.csv",csv_cases)):
            (root / filename).write_bytes(content.encode("utf-8"))
    accuracy_csv = wide["accuracy_csv_append"].replace(",grid290_cv_", ",grid290_cv_wide_")
    return dict(rows=summaries,cases=by_case,ids=base["ids"],summary_csv=csv_summary,case_csv=csv_cases,
                accuracy_csv_append=accuracy_csv,protocol=wide["protocol"])
