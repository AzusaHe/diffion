"""Compare the separately trained 400- and 500-update schedules with field width fixed."""
import numpy as np

if __package__:
    from .fine_cv_report import as_csv, build_fine_cv
    from .fine_cv_width_report import COSTS, KEY, PROPERTIES
else:
    from fine_cv_report import as_csv, build_fine_cv
    from fine_cv_width_report import COSTS, KEY, PROPERTIES

CHECKPOINTS = (("initial",0,0),("stage1",300,375),("final",400,500))
OBS = ("material_id","current_indices","dimensionless_currents","valid_frames_per_current","c_min","c_max","score_min","score_max",
       "archived_TTT_updates","archived_start_mean_s")


def add_budget_view(root, html):
    begin, end = "/* BEGIN CV BUDGET */", "/* END CV BUDGET */"
    js = begin+"\n"+(root/"fine_cv_budget_view.js").read_text(encoding="utf-8")+"\n"+end
    if begin in html:
        return html[:html.index(begin)]+js+html[html.index(end)+len(end):]
    def replace(old, new):
        nonlocal html
        if html.count(old) != 1:
            raise ValueError(f"Unknown c/v budget report anchor: {old}")
        html = html.replace(old,new)
    nav='<button data-tab="cv-width">c/v network width</button>'
    replace(nav,nav+'<button data-tab="cv-budget">400 vs. 500 updates</button>')
    replace("function groupTable(",js+"\nfunction groupTable(")
    replace('||tab==="cv-width")?', '||tab==="cv-width"||tab==="cv-budget")?')
    replace('({accuracy:showAccuracy,','({accuracy:showAccuracy,"cv-budget":showCvBudget,')
    replace('    if(tab==="cv-width")widthBind();','    if(tab==="cv-width")widthBind();\n    if(tab==="cv-budget")budgetBind();')
    replace("DB.fine_cv_width.accuracy_csv_append]","DB.fine_cv_width.accuracy_csv_append,DB.fine_cv_budget.accuracy_csv_append]")
    return html


def build_budget_comparison(root, payload, errors, preview=False):
    shorter=payload.get("fine_cv_width")
    if not shorter:
        return None
    longer=build_fine_cv(root,payload["materials"],payload["runs"],errors,payload["equal_budget"],True,
                         result_dir="fine_cv_double_500_results")
    if not longer:
        return None
    assert shorter["protocol"]["checkpoint_steps"] == [0,300,400]
    assert longer["protocol"]["checkpoint_steps"] == [0,375,500]
    assert longer["protocol"]["field_widths"] == shorter["protocol"]["field_widths"] == [3,44,47,46,1]
    assert longer["ids"] == shorter["ids"] and longer["cases"].keys() == shorter["cases"].keys()
    summaries, flat, cases = [], [], {}
    for fit in longer["rows"]:
        config={k:fit[k] for k in KEY}
        d,c,m,s = (config[k] for k in KEY)
        key=f"{d}__{c}__{m}_{s}"
        expected=(0,500,0) if m=="standard" else (236,139,125)
        assert tuple(fit[k] for k in ("field_prefit_updates","joint_updates","correction_updates")) == expected
        schedule=dict(run400_new_PINN_updates=400,run500_new_PINN_updates=500,
                      run400_field_prefit_updates=0 if m=="standard" else 189,run400_joint_updates=400 if m=="standard" else 111,
                      run400_correction_updates=0 if m=="standard" else 100,run500_field_prefit_updates=expected[0],
                      run500_joint_updates=expected[1],run500_correction_updates=expected[2])
        pairs=[]
        for a,b in zip(shorter["cases"][key],longer["cases"][key]):
            for field in OBS:
                if a[field] != b[field]:
                    raise ValueError(f"Different {field} for {key}, material {a['material_id']}")
            row={k:a[k] for k in OBS}
            for stage, step_a, step_b in CHECKPOINTS:
                for prop in PROPERTIES:
                    v_a,v_b=a[f"wide_{prop}_{step_a}"],b[f"{prop}_{step_b}"]
                    if stage=="initial":
                        np.testing.assert_allclose(v_a,v_b,atol=1.e-7,rtol=1.e-5)
                    row[f"run400_{prop}_{step_a}"]=v_a
                    row[f"run500_{prop}_{step_b}"]=v_b
                    row[f"delta_{prop}_{stage}"]=v_b-v_a
            for cost in COSTS:
                a_cost,b_cost=a["wide_"+cost],b[cost]
                row["run400_"+cost],row["run500_"+cost],row["delta_"+cost]=a_cost,b_cost,b_cost-a_cost
            pairs.append(row)
            flat.append(config|row|schedule)
        assert len(pairs)==8
        cases[key]=pairs
        summary=config|dict(n_test=len(pairs),archived_TTT_updates=fit["archived_TTT_updates"],params_per_field=4546)|schedule
        for stage, step_a, step_b in CHECKPOINTS:
            for prop in PROPERTIES:
                av,bv,change=f"run400_{prop}_{step_a}",f"run500_{prop}_{step_b}",f"delta_{prop}_{stage}"
                for name in (av,bv,change):
                    summary["median_"+name]=float(np.median([r[name] for r in pairs]))
                summary[f"delta_of_medians_{prop}_{stage}"]=summary["median_"+bv]-summary["median_"+av]
                summary[f"n_lower_with_500_{prop}_{stage}"]=int(np.count_nonzero([r[change]<0 for r in pairs]))
        for cost in COSTS:
            for prefix in ("run400","run500","delta"):
                name=prefix+"_"+cost
                summary["mean_"+name]=float(np.mean([r[name] for r in pairs]))
        summary["PINN_time_ratio_500_over_400"]=summary["mean_run500_PINN_s"]/summary["mean_run400_PINN_s"]
        summaries.append(summary)
    summary_csv,case_csv=as_csv(summaries),as_csv(flat)
    if not preview:
        for fname,data in (("fine_cv_budget_accuracy_cost.csv",summary_csv),("fine_cv_budget_materials.csv",case_csv)):
            (root/fname).write_bytes(data.encode("utf-8"))
    accuracy_csv=longer["accuracy_csv_append"].replace(",grid290_cv_",",grid290_cv_wide500_")
    return dict(rows=summaries,cases=cases,ids=longer["ids"],summary_csv=summary_csv,case_csv=case_csv,
                accuracy_csv_append=accuracy_csv,protocol=longer["protocol"])
