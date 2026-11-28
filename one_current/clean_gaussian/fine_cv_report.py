"""Summarize the 400-update c/v PINNs using their saved fine-grid fits."""
import csv
import io
import json

import numpy as np


METHODS = ("standard", "correction_w1", "correction_w100")
STARTS = ("center", "fno_ttt")
STEPS = (0, 300, 400)


def as_csv(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=rows[0])
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def add_fine_cv_view(root, html):
    """Extend the shipped HTML in place so its historical view code stays paired with its data."""
    begin, end = "/* BEGIN FINE CV VIEW */", "/* END FINE CV VIEW */"
    js = begin + "\n" + (root / "fine_cv_view.js").read_text(encoding="utf-8") + "\n" + end
    if begin in html:
        i, j = html.index(begin), html.index(end) + len(end)
        return html[:i] + js + html[j:]
    def replace_once(old, new):
        nonlocal html
        if html.count(old) != 1:
            raise ValueError(f"Unrecognized report template anchor: {old}")
        html = html.replace(old, new)
    nav = '<button data-tab="accuracy">Accuracy</button>'
    replace_once(nav, nav + '<button data-tab="fine-cv">290 × 151 c/v PINNs</button>')
    replace_once("function groupTable(", js + "\nfunction groupTable(")
    replace_once("const family=byId(\"family\").value;", 'const family=byId("family").value;\n'+
                 'if(!family||family===groupNames[2])h+=\'<p><a href="#fine-cv">290 × 151 c/v PINNs</a> have 400 new updates per fit. Compare the 300- and 400-update estimates in the new study.</p>\';')
    location = "if(includeFine) h+=fineAccuracy()" if "if(includeFine) h+=fineAccuracy()" in html else "h+=accuracyTable();"
    replace_once(location, "if(!family||family===groupNames[2])h+=cvAccuracy();\n" + location)
    # Keep every pre-existing CSV component whatever the historical report revision.
    a=html.index("const blob=new Blob([DB.equal_budget.csv")
    b=html.index('],{type:', a)
    html=html[:b] + ",DB.fine_cv.accuracy_csv_append" + html[b:]
    replace_once('const earlier=[', 'const earlier=tab==="fine-cv"?"":[')
    replace_once('({accuracy:showAccuracy,', '({accuracy:showAccuracy,"fine-cv":showFineCv,')
    replace_once('    if(tab==="accuracy"&&byId("equalMethod"))', '    if(tab==="fine-cv")cvBind();\n    if(tab==="accuracy"&&byId("equalMethod"))')
    # The older Accuracy section and masthead used universal wording before the new 400-update cohort existed.
    html=html.replace("Every PINN uses 1,000 optimizer updates per material.", "The earlier PINN comparison uses 1,000 optimizer updates per material.")
    html=html.replace("updates per material for each PINN on Accuracy", "updates per fit in the earlier PINN comparison")
    return html


def build_fine_cv(root, materials, runs, errors, equal, preview=False, *, result_dir="fine_cv_results", cohort_size=8):
    folder = root / result_dir
    keys = [(d, n, m, s) for d in ("clean", "gaussian_noisy") for n in (1, 3) for m in METHODS for s in STARTS]
    stems = [folder / f"{d}__{n}__{m}_{s}" for d, n, m, s in keys]
    if any(not p.with_suffix(ext).is_file() for p in stems for ext in (".npz", ".json")):
        return None
    protocol = json.loads((folder / "protocol.json").read_text(encoding="utf-8"))
    steps = protocol["checkpoint_steps"]
    first, final = steps[1:]
    second = final - first
    ids = np.array([m["id"] for m in materials[:cohort_size]])
    truth = np.array([m["truth"] for m in materials[:cohort_size]])
    chosen = np.array([m["chosen"] for m in materials[:cohort_size]])
    assert len(ids) == cohort_size
    np.testing.assert_array_equal(protocol["material_ids"], ids)
    summary_rows, material_rows, cases, saved_bases = [], [], {}, {}
    all_accuracy = io.StringIO(newline="")
    writer = csv.DictWriter(all_accuracy, fieldnames=next(csv.reader(io.StringIO(equal["csv"]))))
    for (domain, count, method, start), stem in zip(keys, stems):
        with np.load(stem.with_suffix(".npz")) as z:
            arrays = {k: z[k] for k in z.files}
        stages = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))["stages"]
        assert sum(stages.values()) == final and steps[0] == 0
        n = len(ids)
        np.testing.assert_array_equal(arrays["ids"], ids)
        np.testing.assert_array_equal(arrays["current_indices"], chosen[:, :count])
        prior = runs[f"{domain}__{count}__{start}"] if start != "center" else None
        expected = np.clip([r[5:] for r in prior["observations"][:n]], 1.e-5, 1-1.e-5) if prior else np.full((n, 8), .5)
        for k in ("theta0", f"theta{first}", "theta"):
            assert arrays[k].shape == (n, 8) and np.isfinite(arrays[k]).all() and np.all((arrays[k] >= 0) & (arrays[k] <= 1)), (stem.name, k)
        np.testing.assert_allclose(arrays["theta0"], expected, atol=2.e-6, rtol=0.)
        for k in ("lo", "hi", "c_rmse0", "v_rmse0", f"c_rmse{first}", f"v_rmse{first}", "c_rmse", "v_rmse",
                  "stage1_s", "stage2_s", "correction_setup_s", "elapsed_s", f"updates{first}", "total_updates"):
            assert arrays[k].shape == (n,) and np.isfinite(arrays[k]).all(), (stem.name, k)
        assert np.all(arrays["stage1_s"] > 0) and np.all(arrays["stage2_s"] > 0)
        np.testing.assert_allclose(arrays["stage1_s"] + arrays["stage2_s"], arrays["elapsed_s"], rtol=1.e-6)
        assert np.all(arrays[f"updates{first}"] == first) and np.all(arrays["total_updates"] == final)
        if method.startswith("correction"):
            state = tuple(arrays[k] for k in (f"theta{first}", f"c_rmse{first}", f"v_rmse{first}", "stage1_s", "correction_setup_s",
                                                f"base{first}_hash", "correction_initial_hash"))
            base_key = domain, count, start
            if base_key in saved_bases:
                for actual, original in zip(state, saved_bases[base_key]):
                    np.testing.assert_array_equal(actual, original)
            else:
                saved_bases[base_key] = state
        scored_lo, scored_hi = np.maximum(arrays["lo"], 0), np.minimum(arrays["hi"], 2)
        init_s = prior["per_s"] if prior else 0.
        config = dict(domain=domain, currents=count, method=method, start=start)
        result = config | dict(n_test=n, stage1_updates=first, stage2_updates=second, new_PINN_updates=final,
                               field_prefit_updates=stages.get("field_prefit",0), joint_updates=stages["joint"],
                               correction_updates=stages.get("correction",0), correction_setup_mean_s=np.mean(arrays["correction_setup_s"]),
                               archived_TTT_updates=150 if prior else 0, median_observed_c_span=np.median(scored_hi - scored_lo),
                               archived_start_mean_s=init_s, stage1_mean_s=np.mean(arrays["stage1_s"]),
                               stage2_mean_s=np.mean(arrays["stage2_s"]), PINN_mean_s=np.mean(arrays["elapsed_s"]),
                               pipeline_mean_s=init_s + np.mean(arrays["elapsed_s"]))
        scores = {}
        for step, th, suffix in zip(steps, ("theta0", f"theta{first}", "theta"), ("0", str(first), "")):
            ds, ts = errors(arrays[th], truth, arrays["lo"], arrays["hi"])
            for name, values in (("D_pct", ds), ("t_rmse", ts), ("c_rmse", arrays["c_rmse"+suffix]), ("v_rmse", arrays["v_rmse"+suffix])):
                assert np.all(values >= 0) and np.isfinite(values).all()
                scores[f"{name}_{step}"] = values
                result[f"median_{name}_{step}"] = np.median(values)
        summary_rows.append(result)
        items = []
        for i, mid in enumerate(ids):
            detail = dict(material_id=int(mid), current_indices=" ".join(map(str,chosen[i,:count])),
                          dimensionless_currents=" ".join(map(str,materials[i]["iota"][:count])),
                          valid_frames_per_current=" ".join(map(str, arrays["valid_frames"][i])),
                          c_min=arrays["lo"][i], c_max=arrays["hi"][i], score_min=scored_lo[i], score_max=scored_hi[i],
                          stage1_s=arrays["stage1_s"][i], stage2_s=arrays["stage2_s"][i], correction_setup_s=arrays["correction_setup_s"][i], PINN_s=arrays["elapsed_s"][i],
                          archived_start_mean_s=init_s, pipeline_s=init_s+arrays["elapsed_s"][i],
                          archived_TTT_updates=150 if prior else 0, stage1_updates=first, stage2_updates=second, new_PINN_updates=final)
            detail.update({k: v[i] for k, v in scores.items()})
            material_rows.append(config | detail)
            items.append(detail)
        cases[stem.name] = items
        shared = np.isin(ids, equal["shared_ids"])
        writer.writerow(dict(domain=domain, currents=count, method=f"grid290_cv_{method}_{start}", family="PINN refinement",
            budget=final+result["archived_TTT_updates"], ttt_updates=result["archived_TTT_updates"],
            field_prefit_updates=result["field_prefit_updates"], joint_updates=result["joint_updates"], correction_updates=result["correction_updates"],
            arm="values", n_test=n, D_median_pct=result[f"median_D_pct_{final}"],
            t_median_rmse=result[f"median_t_rmse_{final}"], median_observed_c_span=result["median_observed_c_span"],
            matched6_D_median_pct=np.median(scores[f"D_pct_{final}"][shared]), matched6_t_median_rmse=np.median(scores[f"t_rmse_{final}"][shared]),
            matched6_median_observed_c_span=np.median((scored_hi-scored_lo)[shared]), device="NVIDIA GeForce RTX 4090",
            pipeline_s_per_material=result["pipeline_mean_s"], initialization_s_per_material=init_s, fit_s_per_material=result["PINN_mean_s"],
            shared_training_s=prior["training"] if prior else 0., validation_search_s=prior["tuning"] if prior else 0.))
    summary_csv, case_csv = as_csv(summary_rows), as_csv(material_rows)
    if not preview and result_dir == "fine_cv_results":
        for name, csv_text in (("fine_cv_accuracy_cost.csv", summary_csv), ("fine_cv_materials.csv", case_csv)):
            (root / name).write_bytes(csv_text.encode("utf-8"))
    return dict(rows=summary_rows, cases=cases, ids=ids.tolist(), summary_csv=summary_csv,
                case_csv=case_csv, accuracy_csv_append=all_accuracy.getvalue(), protocol=protocol)
