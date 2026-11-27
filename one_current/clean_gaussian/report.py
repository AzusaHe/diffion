"""Assemble a portable benchmark report from predictions and recorded wall times."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

if __package__:
    from .derivative_report import PAIRED, build_comparison
    from .equal_budget_report import build_equal_budget
else:
    from derivative_report import PAIRED, build_comparison
    from equal_budget_report import build_equal_budget

HERE = Path(__file__).resolve().parent
DOMAINS = ("clean", "gaussian_noisy")
BASE = ["mlp","mlp_ttt","fno","fno_ttt","cnn"]
SOLVER = ["lbfgs_"+s for s in ("center","mlp","mlp_ttt","fno","fno_ttt")]
SOLVER += [m+"_"+s for m in ("nm","cmaes","bo") for s in ("center","fno_ttt")]
PILOT = ["pinn_"+m+"_"+s for m in ("standard_c","standard_cv","staged_c_w1","staged_c_w100",
         "staged_cv_w1","staged_cv_w100","data_first","fine") for s in ("center","fno_ttt")]
PILOT += ["pinn_"+m+"_fno_ttt" for m in ("data_correction","solver_correction","solver_correction_weight1")]
ORDER = BASE + SOLVER + PILOT + ["cnn_pinn"]
NAMES = dict(mlp="MLP",fno="FNO",cnn="CNN",mlp_ttt="MLP + TTT",fno_ttt="FNO + TTT",
             cnn_pinn="CNN → staged PINN",center="Center")
PINN_NAMES = dict(standard_c="Standard · c",standard_cv="Standard · c and v",staged_c_w1="Staged · c · w = 1",
         staged_c_w100="Staged · c · w = 100",staged_cv_w1="Staged · c and v · w = 1",
         staged_cv_w100="Staged · c and v · w = 100",data_first="Data-first",fine="Data-first · fine grid",
         data_correction="Data-first + correction",solver_correction="Solver + correction · wᴅ = 0.01",
         solver_correction_weight1="Solver + correction · wᴅ = 1",frozen="Two-stage · frozen fields")


def label(s):
    if s in NAMES: return NAMES[s]
    start=next(k for k in ("fno_ttt","mlp_ttt","mlp","fno","center") if s.endswith("_"+k))
    method=s[:-len(start)-1]
    if method.startswith("pinn_"): name=PINN_NAMES[method[5:]]
    else: name={"lbfgs":"L-BFGS-B","bo":"Bayesian optimization","cmaes":"CMA-ES","nm":"Nelder–Mead"}[method]
    return f"{name} ({NAMES[start]} start)"


def errors(estimate,truth,lo,hi):
    estimate,truth,lo,hi = (np.asarray(x,dtype=np.float64) for x in (estimate,truth,lo,hi))
    low,high=np.maximum(lo,0),np.minimum(hi,2)
    if not np.all(high>low): raise ValueError("Empty property interval")
    u=np.linspace(low/2,high/2,301,axis=1)
    b=np.stack(((1-u)**3,3*u*(1-u)**2,3*u**2*(1-u),u**3),axis=2)
    def curves(theta):
        d=1e-12*np.exp(np.log(20)*np.einsum("ntk,nk->nt",b,theta[:,:4]))
        t=-.25+1.25*np.einsum("ntk,nk->nt",b,theta[:,4:])
        return d,t
    d,t=curves(estimate);dt,tt=curves(truth)
    return 100*np.linalg.norm(d-dt,axis=1)/np.linalg.norm(dt,axis=1),np.sqrt(np.mean((t-tt)**2,axis=1))


def read(path): return json.loads(path.read_text(encoding="utf-8"))

def format_schedule(metadata):
    text=metadata["description"]
    if "Budgeted rerun: " not in text:
        return
    header,encoded=text.split("Budgeted rerun: ",1)
    schedule,end=json.JSONDecoder().raw_decode(encoded)
    stages=[(schedule.get(k,0),name) for k,name in (
        ("fit_steps","field-prefit Adam steps"),("fit_lbfgs_iters","field-prefit L-BFGS iterations"),
        ("warmup_steps","physics warmup steps"),("steps","joint Adam steps"),("lbfgs_iters","joint L-BFGS iterations"))]
    pretty=", ".join(f"{count:,} {name}" for count,name in stages if count)
    weights=schedule.get("w",dict(data=1.,pde=100.,vol=100.,bc=100.))
    loss=", ".join(f"{k} = {v:g}" for k,v in weights.items())
    metadata["description"]=header+f"Schedule includes {pretty}. Loss weights are {loss}. "+encoded[end:].lstrip(". ")


def main(preview=False):
    cache=HERE/"cache"
    with np.load(cache/"binned.npz") as z:
        ti=np.flatnonzero(z["split"]=="test")
        ids=z["ids"][ti]; truth=z["theta"][ti]
        materials=[dict(id=int(mid),truth=theta.tolist(),chosen=p.tolist(),iota=io.tolist()) for mid,theta,p,io
                   in zip(ids,truth,z["chosen"][ti],z["iota"][ti])]
        coarse={dom:z[dom][ti] for dom in DOMAINS}
    cases=dict(zip(ids,range(len(ids))))
    bank={}; seeds={}; summaries=[]
    for domain in DOMAINS:
        for count in (1,3):
            marks={}
            for slug in ORDER:
                stem=HERE/"results"/f"{domain}__{count}__{slug}"
                if preview and not stem.with_suffix(".json").exists(): continue
                md=read(stem.with_suffix(".json"))
                format_schedule(md)
                with np.load(stem.with_suffix(".npz")) as z:
                    mids,theta,lo,hi=(z[k] for k in ("ids","theta","lo","hi"))
                    individual_times=z["elapsed_s"].tolist() if "elapsed_s" in z else None
                rows=[cases[m] for m in mids]
                expected_n=(389 if slug in BASE else 30 if slug.startswith("bo_")
                            else 300 if slug in SOLVER else 8 if slug in PILOT else 1)
                assert len(rows)==expected_n,(stem.name,len(rows),expected_n)
                assert len(set(mids))==len(rows) and theta.shape==(len(mids),8)
                assert np.isfinite(theta).all() and np.all((theta>=-1e-7)&(theta<=1.0000001))
                assert md["train_s"]>=0 and md["test_s"]>0
                assert mids.tolist()==ids[:len(mids)].tolist()
                if slug=="cnn_pinn":
                    assert md["gate_status"] in ("accepted","rejected")
                    assert md["checkpoint"]==("final_checkpoint.pt" if md["gate_status"]=="accepted" else "latest_checkpoint.pt")
                if slug in BASE or slug in SOLVER or slug in PILOT:
                    if slug!="cnn" and not slug.startswith("pinn_fine_"):
                        obs=coarse[domain][rows,:count]
                        np.testing.assert_allclose(lo,np.nanmin(obs,axis=(1,2,3)),atol=1e-6)
                        np.testing.assert_allclose(hi,np.nanmax(obs,axis=(1,2,3)),atol=1e-6)
                de,te=errors(theta,truth[rows],lo,hi)
                if md.get("warm_start"):
                    parent=marks[md["warm_start"]]
                    train=parent["training"]
                    tune=parent["tuning"]
                    init=md.get("cnn_init_s",parent["per_s"])
                else:
                    train=md["train_s"]; tune=md.get("tune_s",0.); init=0.
                local=md["test_s"]/len(rows)
                item=dict(name=label(slug),slug=slug,domain=domain,count=count,n=len(rows),meta=md,
                          family="Supervised and TTT" if slug in BASE else "Numerical optimization" if slug in SOLVER else "PINN refinement",
                          training=train,tuning=tune,stage_s=local,init_s=init,per_s=local+init,
                          individual_times=individual_times,
                          # Columns are ID, D error, t+ error, observed c bounds, then coefficients.
                          observations=[[int(mid),float(d),float(t),float(l),float(h),*x.tolist()]
                                         for mid,d,t,l,h,x in zip(mids,de,te,lo,hi,theta)])
                marks[slug]=item; bank[stem.name]=item
                summaries.append(dict(domain=domain,currents=count,method=slug,n_test=len(rows),
                                      D_median_pct=np.median(de),t_median_rmse=np.median(te),
                                      median_observed_c_span=np.median(np.minimum(hi,2)-np.maximum(lo,0)),
                                      train_s=train,validation_search_s=tune,stage_test_s=md["test_s"],
                                      pipeline_mean_s_per_material=local+init,
                                      gate_status=md.get("gate_status",""),checkpoint=md.get("checkpoint","")))
                if slug in ("mlp","fno","mlp_ttt","fno_ttt"):
                    rep=[dict(seed=0,d=float(np.median(de)),t=float(np.median(te)),train=train,test=local)]
                    for seed in (1,2):
                        rf=HERE/"results_seeds"/str(seed)/stem.name
                        if preview and not rf.with_suffix(".json").exists(): continue
                        with np.load(rf.with_suffix(".npz")) as zz:
                            np.testing.assert_array_equal(zz["ids"],mids)
                            np.testing.assert_allclose(zz["lo"],lo);np.testing.assert_allclose(zz["hi"],hi)
                            ds,ts=errors(zz["theta"],truth[rows],lo,hi)
                        cost=read(rf.with_suffix(".json"))
                        rep.append(dict(seed=seed,d=float(np.median(ds)),t=float(np.median(ts)),train=cost["train_s"],test=cost["test_s"]/len(rows)))
                    seeds[stem.name]=rep
    if not preview:
        assert len(bank)==4*len(ORDER) and len(ORDER)==36
        for slug in ORDER:
            cohort={tuple(r[0] for r in bank[f"{d}__{c}__{slug}"]["observations"]) for d in DOMAINS for c in (1,3)}
            assert len(cohort)==1
    summary=read(cache/"preparation.json")
    costs={d:read(cache/f"cnn_{d}_prep.json")["seconds"] for d in DOMAINS}
    payload=dict(runs=bank,materials=materials,order=ORDER,names={s:label(s) for s in ORDER},
                 preparation=summary,cnn_prep=costs,seeds=seeds)
    payload["derivatives"] = build_comparison(HERE, materials, bank, errors, label, preview)
    payload["equal_budget"] = build_equal_budget(HERE, materials, bank, BASE+SOLVER, PILOT, PAIRED, errors, label, preview)
    payload["architectures"] = [
        model for filename in ("arch_learned.json", "arch_native.json", "arch_pinn.json")
        for model in read(HERE/filename)
    ]
    diag=HERE/"learn_residual_diagnostic.json"
    if diag.exists(): payload["residuals"]=read(diag)
    html=(HERE/"report_template.html").read_text(encoding="utf-8")
    data=json.dumps(payload,allow_nan=False,ensure_ascii=False,separators=(",",":")).replace("</","<\\/")
    report=HERE/("cache/report_preview.html" if preview else "Clean_Gaussian_Benchmark.html")
    derivative_view = (HERE / "derivative_view.js").read_text(encoding="utf-8")
    report.write_text(html.replace("__DERIVATIVE_VIEW__", derivative_view).replace("__BENCHMARK_DATA__",data),encoding="utf-8")
    if not preview:
        with (HERE/"accuracy_cost.csv").open("w",encoding="utf-8",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
    print(report, len(bank), "conditions",sum(row["n_test"] for row in summaries),"material predictions")


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--preview",action="store_true")
    main(p.parse_args().preview)
