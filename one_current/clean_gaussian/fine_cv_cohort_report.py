"""Summarize the first 300 test materials for the wide, 500-update c/v study."""
import hashlib
import json

import numpy as np

if __package__:
    from .fine_cv_report import as_csv, build_fine_cv
else:
    from fine_cv_report import as_csv, build_fine_cv

RESULTS = "fine_cv_double_500_300_results"
EARLIER = "fine_cv_double_500_results"
METRICS = ("D_pct","t_rmse","c_rmse","v_rmse")


def add_cohort_view(root, html):
    begin, end = "/* BEGIN CV 300 MATERIALS */", "/* END CV 300 MATERIALS */"
    js = begin + "\n" + (root / "fine_cv_cohort_view.js").read_text(encoding="utf-8") + "\n" + end
    if begin in html:
        return html[:html.index(begin)] + js + html[html.index(end)+len(end):]
    def replace(old, new):
        nonlocal html
        if html.count(old) != 1:
            raise ValueError(f"Missing 300-material report anchor: {old}")
        html = html.replace(old,new)
    nav = '<button data-tab="cv-budget">400 vs. 500 updates</button>'
    replace(nav,nav+'<button data-tab="cv-300">300-material c/v PINNs</button>')
    replace("function groupTable(",js+"\nfunction groupTable(")
    replace('||tab==="cv-budget")?','||tab==="cv-budget"||tab==="cv-300")?')
    replace('({accuracy:showAccuracy,','({accuracy:showAccuracy,"cv-300":showCv300,')
    replace('    if(tab==="cv-budget")budgetBind();','    if(tab==="cv-budget")budgetBind();\n    if(tab==="cv-300")cv300Bind();')
    replace("if(!family||family===groupNames[2])h+=cvAccuracy();","if(!family||family===groupNames[2])h+=cv300Accuracy()+cvAccuracy();")
    replace("DB.fine_cv_budget.accuracy_csv_append]","DB.fine_cv_budget.accuracy_csv_append,DB.fine_cv_300.accuracy_csv_append]")
    return html


def matched_solvers(root, payload, result, errors):
    """Rescore previously fitted derivative-free coefficients on the exact PINN scoring intervals."""
    earlier = payload.get("fine_df")
    if not earlier:
        return []
    ids = np.asarray(result["ids"])
    np.testing.assert_array_equal(ids, earlier["ids"])
    truth = np.array([m["truth"] for m in payload["materials"][:len(ids)]])
    chosen = np.array([m["chosen"] for m in payload["materials"][:len(ids)]])
    comparison = []
    for entry in earlier["conditions"]:
        d,n,method,start = (entry[k] for k in ("domain","currents","optimizer","start"))
        if start not in ("center","fno_ttt"):
            continue
        src = root / "fine_df_results" / f"{d}__{n}__{method}_{start}.npz"
        if not src.is_file():
            return []
        bounds = result["cases"][f"{d}__{n}__standard_{start}"]
        lo,hi = (np.array([v[k] for v in bounds]) for k in ("c_min","c_max"))
        with np.load(src) as z:
            np.testing.assert_array_equal(z["ids"],ids)
            np.testing.assert_array_equal(z["current_indices"],chosen[:,:n])
            np.testing.assert_allclose(z["lo"],lo,rtol=0.,atol=2.e-7)
            np.testing.assert_allclose(z["hi"],hi,rtol=0.,atol=2.e-7)
            ds,ts=errors(z["theta"],truth,lo,hi)
        comparison.append(dict(domain=d,currents=n,method=method,start=start,n_test=len(ids),
                               median_D_pct=float(np.median(ds)),median_t_rmse=float(np.median(ts)),
                               pipeline_mean_s=entry["pipeline_s_per_material"]))
    return comparison


def matched_graph_timings(cases):
    """Aggregate the interleaved two-material comparison from the actual study fits."""
    groups={}
    for row in cases:
        if row["material_id"] in (1275,1355):
            key=tuple(row[k] for k in ("material_id","domain","currents","start"))
            groups.setdefault(key,[]).append(row)
    totals={mode:dict(contexts=0,seconds=0.) for mode in ("separate_graph","reused_graph")}
    assert len(groups)==16
    for members in groups.values():
        assert len(members)==3 and len({r["method"] for r in members})==3
        modes={r[k] for r in members for k in ("base_execution","final_execution")}
        assert len(modes)==1
        group=totals[modes.pop()]
        group["contexts"]+=1
        group["seconds"]+=sum(r["shared_once_PINN_s_contribution"] for r in members)
    assert all(v["contexts"]==8 for v in totals.values())
    return totals


def verified_native_audit(folder):
    path=folder/"raw_field_validation.json"
    if not path.is_file():
        return None
    audit=json.loads(path.read_text(encoding="utf-8"))
    assert [audit[k] for k in ("materials","contexts","pipelines","endpoint_records")]==[300,2400,7200,14400]
    assert audit["field_evaluations_at_updates"]==[0,375,500] and audit["reused_eight_checkpoint_files_byte_verified"]==64
    records={v["path"]:v["sha256"] for v in audit["scored_archives"]}
    actual={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.glob("*__*__*.npz")}
    if len(actual)!=24 or actual!=records:
        raise ValueError("Native-data validation hashes differ from the current coefficient and score archives.")
    return {k:audit[k] for k in ("pipelines","endpoint_records","max_c_rmse_abs","max_v_rmse_abs",
                                    "min_observed_frames_per_current","max_observed_frames_per_current")}


def build_cohort_report(root, payload, errors, preview=False):
    if not payload.get("fine_cv_budget"):
        return None
    study=build_fine_cv(root,payload["materials"],payload["runs"],errors,payload["equal_budget"],
                        preview=True,result_dir=RESULTS,cohort_size=300)
    if not study:
        return None
    assert len(study["rows"])==24 and study["protocol"]["checkpoint_steps"]==[0,375,500]
    assert study["protocol"]["field_widths"]==[3,44,47,46,1]
    assert len(set(study["ids"]))==300
    cases=[]
    for row in study["rows"]:
        d,n,m,s=(row[k] for k in ("domain","currents","method","start"))
        key=f"{d}__{n}__{m}_{s}"
        config=dict(domain=d,currents=n,method=m,start=s)
        mask=np.arange(300)<8
        with np.load(root/RESULTS/(key+".npz")) as current, np.load(root/EARLIER/(key+".npz")) as old:
            np.testing.assert_array_equal(current["reused_from_eight"],mask)
            # Exact comparison includes original field/transport metrics, budgets, times and checkpoint hashes.
            for field in old.files:
                np.testing.assert_array_equal(current[field][:8],old[field],err_msg=key+"/"+field)
            elapsed=current["elapsed_s"]
            unique=elapsed.copy()
            if m=="correction_w100":
                unique -= current["stage1_s"]+current["correction_setup_s"]
            assert np.isfinite(unique).all() and np.all(unique>0)
            modes=("separate_graph","reused_graph")
            base_mode,final_mode=(current[k].astype(str) for k in ("base_execution","final_execution"))
            assert base_mode.shape==final_mode.shape==(300,)
            for part,values in (("base",base_mode),("final",final_mode)):
                assert set(values)<=set(modes) and all(values[mask]=="separate_graph")
                for mode in modes:
                    row[f"n_{part}_{mode}"]=int(np.count_nonzero(values==mode))
            row.update(reused_prior_materials=8,new_materials=292,
                       PINN_s_reused_standalone=float(elapsed[mask].sum()),PINN_s_new_standalone=float(elapsed[~mask].sum()),
                       PINN_s_reused_counting_shared_once=float(unique[mask].sum()),
                       PINN_s_new_counting_shared_once=float(unique[~mask].sum()),
                       PINN_s_all_counting_shared_once=float(unique.sum()))
        expected=(0,500,0) if m=="standard" else (236,139,125)
        assert tuple(row[k] for k in ("field_prefit_updates","joint_updates","correction_updates"))==expected
        points=study["cases"][key]
        for k in METRICS:
            start,intermediate,final=(np.array([v[f"{k}_{t}"] for v in points]) for t in (0,375,500))
            q25,q75=np.percentile(final,[25,75])
            row.update({f"p25_{k}_500":float(q25),f"p75_{k}_500":float(q75),
                        f"n_lower_{k}_500_vs_0":int((final<start).sum()),
                        f"n_lower_{k}_500_vs_375":int((final<intermediate).sum()),
                        f"median_paired_delta_{k}_500_vs_0":float(np.median(final-start)),
                        f"median_paired_delta_{k}_500_vs_375":float(np.median(final-intermediate))})
        for i,v in enumerate(points):
            cases.append(config|dict(fit_origin="prior_eight" if mask[i] else "new_292",
                                     base_execution=base_mode[i],final_execution=final_mode[i],
                                     shared_once_PINN_s_contribution=float(unique[i]),
                                     field_prefit_updates=expected[0],joint_updates=expected[1],correction_updates=expected[2])|v)
    assert len(cases)==7200
    # The portable view reads this numeric CSV once to populate its per-material selector.
    # Text columns are identifiers and space-separated index/current lists.
    summary_csv,case_csv=as_csv(study["rows"]),as_csv(cases)
    if '"' in case_csv:
        raise ValueError("Unexpected quoted text in the embedded numeric material CSV.")
    reference=matched_solvers(root,payload,study,errors)
    graph_timing=matched_graph_timings(cases)
    validation=json.loads((root/RESULTS/"acceleration_validation.json").read_text(encoding="utf-8"))
    combined,=[r for r in validation["timing_summary"] if r["currents"]=="both"]
    for mode,k in (("separate_graph","fresh_graph_training_s"),("reused_graph","reused_graph_training_s")):
        np.testing.assert_allclose(graph_timing[mode]["seconds"],combined[k],atol=1.e-7,rtol=0.)
    graph_timing["equal_checkpoints"]=sum(v["checkpoints_with_bitwise_equal_fields_logits_and_rmse"] for v in validation["reference_results"])
    graph_timing["equal_tensors"]=sum(v["tensors_compared_bitwise"] for v in validation["reference_results"])
    native_audit=verified_native_audit(root/RESULTS)
    if not preview:
        for name,text in (("fine_cv_300_accuracy_cost.csv",summary_csv),("fine_cv_300_materials.csv",case_csv)):
            (root/name).write_bytes(text.encode("utf-8"))
    return dict(rows=study["rows"],ids=study["ids"],summary_csv=summary_csv,case_csv=case_csv,
                accuracy_csv_append=study["accuracy_csv_append"].replace(",grid290_cv_",",grid290_cv_wide500_n300_"),
                reference=reference,graph_timing=graph_timing,native_audit=native_audit,protocol=study["protocol"])
