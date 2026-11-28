"""Publish complete supervised results from independently noised observation grids."""
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np

if __package__:
    from .equal_budget_report import summary
    from .fine_df_report import build_fine_df, METHODS
    from .grid_comparison_report import GRIDS, grid_figures, model_name, result_folder, slug_for
else:
    from equal_budget_report import summary
    from fine_df_report import build_fine_df, METHODS
    from grid_comparison_report import GRIDS, grid_figures, model_name, result_folder, slug_for


REVISION = "corrected_grid_native_v1"
FOLDER = "corrected_noisy_results"
DOMAIN = "gaussian_noisy"
BASE_SLUGS = {slug_for(w, m) for w, _ in GRIDS for m in ("mlp", "fno")}
SOLVER_SLUGS = {f"grid290_{method}_{start}" for method in METHODS for start in ("center","fno")}


def write_rows(rows, fields=None, header=True):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields or list(dict.fromkeys(k for row in rows for k in row)))
    if header:
        writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def complete_grids(root):
    widths = []
    for w,t in GRIDS:
        folder = root/FOLDER/f"grid{w}"
        stems = [folder/f"seed{s}"/f"{DOMAIN}__{n}__{model}"
                 for s in range(3) for n in (1,3) for model in ("fno","mlp" if w==20 else "conv_mlp")]
        if ((folder/"preparation.json").is_file()
            and all(stem.with_suffix(ext).is_file() for stem in stems for ext in (".json",".npz"))
            and all(qualified_metadata(json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))) for stem in stems)):
            widths.append((w,t))
    return widths


def qualified_metadata(meta):
    return (meta.get("epoch_metrics") == "same_fit" and meta.get("training_precision") == "float32_tf32_off"
            and meta.get("cpu_threads") == 1 and Path(meta.get("checkpoint_root","")).name == "same_fit_fp32")


def same_fit_weights(root,w,t,n,seed,model,meta):
    local=root/meta["checkpoint_root"]/f"seed{seed}"/f"{DOMAIN}__{n}__{model}"
    portable=root/"trained_models"/f"grid_{w}x{t}"/DOMAIN/f"{n}_current"/f"seed_{seed}"/model
    if (local/"epoch_metrics.csv").is_file():
        trace=local/"epoch_metrics.csv"
        weights=local/"best.pt"
        trace_sha=hashlib.sha256(trace.read_bytes()).hexdigest()
    else:
        trace=portable/"epochs.csv";weights=portable/"best.pt"
        manifest=json.loads((portable/"manifest.json").read_text(encoding="utf-8"))
        provenance=json.loads((portable/"loss_provenance.json").read_text(encoding="utf-8"))
        assert provenance["status"]=="same_fit"
        trace_sha=manifest["files"]["epoch_metrics_sha256"]
        # Git may check out CSVs with LF or CRLF; the separate trainer hash remains byte-exact.
        normalized=trace.read_bytes().replace(b"\r\n",b"\n")
        lf_hash=hashlib.sha256(normalized).hexdigest()
        canonical=manifest["files"].get("packaged_epoch_history_lf_sha256")
        if canonical:
            assert canonical==lf_hash,trace
        else:
            crlf_hash=hashlib.sha256(normalized.replace(b"\n",b"\r\n")).hexdigest()
            assert manifest["files"]["packaged_epoch_history_sha256"] in (lf_hash,crlf_hash),trace
    with trace.open(encoding="utf-8-sig",newline="") as f:epochs=list(csv.DictReader(f))
    assert len(epochs)==100 and [int(r["epoch"]) for r in epochs]==list(range(1,101))
    for name in ("train_mse","validation_mse","test_mse"):
        values=np.asarray([float(row[name]) for row in epochs])
        assert np.all(np.isfinite(values) & (values>=0)), (trace,name)
    best=int(np.argmin([float(row["validation_mse"]) for row in epochs]))
    assert best+1==meta["best_epoch"], (trace,best+1,meta["best_epoch"])
    np.testing.assert_allclose(float(epochs[best]["validation_mse"]),meta["best_val_coef_mse"],rtol=3e-6)
    weight_sha=hashlib.sha256(weights.read_bytes()).hexdigest()
    if not (local/"epoch_metrics.csv").is_file():assert manifest["checkpoint"]["sha256"]==weight_sha
    for k,value in (("checkpoint_sha256",weight_sha),("epoch_metrics_sha256",trace_sha)):
        if meta.get(k):assert meta[k]==value,(k,weights)
    return weight_sha,trace_sha


def clean_reference(root, ids, truth, errors):
    """Read or build the compact clean per-material scores used in the seed-matched comparison."""
    path=root/FOLDER/"clean_reference_scores.npz"
    if path.exists():
        with np.load(path) as archive:
            np.testing.assert_array_equal(archive["ids"],ids)
            np.testing.assert_array_equal(archive["truth"],truth)
            scores={k:archive[k] for k in archive if k not in ("ids","truth")}
    else:
        scores={}
        for w,_ in GRIDS:
            for n in (1,3):
                for family in ("mlp","fno"):
                    model="conv_mlp" if family=="mlp" and w>20 else family
                    for seed in range(3):
                        source=root/result_folder(w,seed)/f"clean__{n}__{model}.npz"
                        with np.load(source) as z:
                            np.testing.assert_array_equal(z["ids"],ids)
                            scores[f"{w}_{n}_{family}_{seed}"]=np.stack(errors(z["theta"],truth,z["lo"],z["hi"]))
        tmp=path.with_suffix(".building.npz")
        np.savez_compressed(tmp,ids=ids,truth=truth,**scores)
        tmp.replace(path)
    assert len(scores)==60 and all(value.shape==(2,len(ids)) and np.isfinite(value).all() for value in scores.values())
    return scores


def complete_solver_counts(root):
    directory=root/FOLDER/"derivative_free_290"
    return [n for n in (1,3) if all((directory/f"{DOMAIN}__{n}__{method}_{start}.{ext}").is_file()
                for method in METHODS for start in ("fno","center") for ext in ("npz","json"))]


def replace_solvers(root,payload,errors,widths):
    """Keep original clean fits and add each completed current cohort for the corrected solver."""
    historical=payload["fine_df"]
    counts=complete_solver_counts(root) if (290,151) in widths else []
    directory=root/FOLDER/"derivative_free_290"
    bank={}
    for n in counts:
        bank[f"{DOMAIN}__{n}__fno"]=payload["runs"][f"{DOMAIN}__{n}__fno_grid290"]
    fresh=build_fine_df(root,payload["materials"],bank,errors,payload["equal_budget"],False,directory=directory,
                        domains=(DOMAIN,),starts=("center","fno"),current_counts=counts,write_csv=False) if counts else None
    old_check=historical["validation"]
    clean=lambda r:r["domain"]=="clean"
    checks=dict(old_check,cases=[r for r in old_check["cases"] if clean(r)],paired_observation_noise=[]) if old_check else None
    rows=[dict(r,dataset_revision="original_clean",initialization_grid=("20x13" if r["start"]!="center" else ""))
          for r in historical["conditions"] if clean(r)]
    fields=next(csv.reader(io.StringIO(payload["equal_budget"]["csv"])))
    acc=[r for r in csv.DictReader(io.StringIO(historical["accuracy_csv_append"]),fieldnames=fields) if clean(r)]
    merged={**historical,**{k:{i:r for i,r in historical[k].items() if i.startswith("clean__")} for k in ("cohorts","before","after")}}
    if fresh:
        prior_sha={}
        for n in counts:
            warm=bank[f"{DOMAIN}__{n}__fno"];bounds=np.array(warm["observations"][:300])[:,3:5]
            np.testing.assert_allclose(fresh["cohorts"][f"{DOMAIN}__{n}"],bounds,rtol=0,atol=1.e-7)
            seed_file=root/FOLDER/"grid290/seed0"/f"{DOMAIN}__{n}__fno.npz"
            prior_sha[n]=hashlib.sha256(seed_file.read_bytes()).hexdigest()
        for r in fresh["conditions"]:
            warm=bank[f"{DOMAIN}__{r['currents']}__fno"];prior=warm["meta"]
            record=directory/f"{DOMAIN}__{r['currents']}__{r['optimizer']}_{r['start']}.json"
            md=json.loads(record.read_text(encoding="utf-8"))
            expected=dict(version="corrected_df_290_v1",dataset_revision=REVISION,source_grid=[290,151],domain=DOMAIN,
                input_sha256=prior["input_sha256"],source_sha256=prior["source_sha256"],initializer_sha256=prior_sha[r["currents"]],
                initializer_grid=[290,151],initializer_seed=0,n_test=300,currents=r["currents"],method=r["optimizer"],
                start=r["start"],solver_cells=580,budget_evals=r["evaluation_limit"],target_concentration_rmse=.0001,max_wall_s=300)
            for k,v in expected.items():
                if md.get(k)!=v:raise ValueError(f"{record}: {k} differs from the corrected solver protocol")
            tail=f"__{r['currents']}__{r['optimizer']}_{r['start']}"
            corrected_end=np.asarray(fresh["after"][DOMAIN+tail])[:,:3]
            clean_end=np.asarray(historical["after"]["clean"+tail])[:,:3]
            changes=np.median(corrected_end-clean_end,axis=0)
            rows.append(dict(r,dataset_revision=REVISION,initialization_grid=("290x151" if r["start"]=="fno" else ""),
                            paired_clean_initialization_grid=("20x13" if r["start"]=="fno" else ""),
                            paired_D_change_vs_clean_pctpt=float(changes[0]),paired_t_change_vs_clean=float(changes[1]),
                            paired_c_change_vs_clean=float(changes[2]),
                            initializer_checkpoint_sha256=(prior["checkpoint_sha256"] if r["start"]=="fno" else ""),
                            initializer_prediction_sha256=(prior_sha[r["currents"]] if r["start"]=="fno" else ""),
                            input_sha256=prior["input_sha256"]))
        acc.extend(csv.DictReader(io.StringIO(fresh["accuracy_csv_append"]),fieldnames=fields))
        for k in ("cohorts","before","after"):merged[k].update(fresh[k])
        incremental=[directory/f"forward_validation_{n}.json" for n in counts]
        checks_to_add=([fresh["validation"]] if fresh["validation"] else
                       [json.loads(path.read_text(encoding="utf-8")) for path in incremental if path.is_file()])
        for incoming in checks_to_add:
            cases=[r for r in incoming["cases"] if r["domain"]==DOMAIN and r["currents"] in counts]
            checks=checks or dict(incoming,cases=[],paired_observation_noise=[])
            checks["cases"].extend(cases)
            checks["paired_observation_noise"].extend(r for r in incoming.get("paired_observation_noise",[]) if r["currents"] in counts)
    merged.update(conditions=rows,csv=write_rows(rows),accuracy_csv_append=write_rows(acc,fields,header=False),validation=checks)
    payload["fine_df"]=merged
    return counts


def archive_runs(root,payload):
    """Recover coefficient starts for a rebuild of the historical derivative-free experiment."""
    if not payload.get("corrected_noisy"):return payload["runs"]
    runs=dict(payload["runs"])
    for n in (1,3):
        for slug in ("fno","mlp"):
            key=f"{DOMAIN}__{n}__{slug}"
            source=root/"results"/key
            meta=json.loads(source.with_suffix(".json").read_text(encoding="utf-8"))
            with np.load(source.with_suffix(".npz")) as z:
                ids,theta=z["ids"],z["theta"]
                starts=np.column_stack((ids,np.zeros((len(ids),4)),theta)).tolist()
            runs[key]=dict(observations=starts,training=meta["train_s"],tuning=0.,per_s=meta["test_s"]/len(ids))
    return runs


def read_fit(root, w, t, family, n, seed, prep, ids):
    model = "conv_mlp" if family=="mlp" and w>20 else family
    relative = Path(FOLDER)/f"grid{w}"/f"seed{seed}"/f"{DOMAIN}__{n}__{model}"
    meta = json.loads((root/relative).with_suffix(".json").read_text(encoding="utf-8"))
    for k,expected in (("dataset_revision",REVISION),("source_grid",[w,t]),("architecture",model),
                       ("currents",n),("seed",seed),("domain",DOMAIN),("n_train",1402),("n_val",156),
                       ("n_test",len(ids)),("epochs",100),("source_folder",prep["source_folder"]),
                       ("input_sha256",prep["input_sha256"]),("source_sha256",prep["source_sha256"]),
                       ("epoch_metrics","same_fit"),("training_precision","float32_tf32_off"),("cpu_threads",1)):
        if meta.get(k)!=expected:
            raise ValueError(f"{relative}: {k}={meta.get(k)!r}, expected {expected!r}")
    assert meta["train_s"]>0 and meta["test_s"]>0 and meta["test_repeats"]==10 and meta["test_batch"]==128
    expected_params=(65741 if n==1 else 65851) if family=="fno" else (23888 if n==1 else 67648)+111*(w>20)
    assert meta["parameters"]==expected_params and 1<=meta["best_epoch"]<=100
    weights_sha,epochs_sha=same_fit_weights(root,w,t,n,seed,model,meta)
    meta=dict(meta,checkpoint_sha256=weights_sha,epoch_metrics_sha256=epochs_sha)
    with np.load((root/relative).with_suffix(".npz")) as z:
        np.testing.assert_array_equal(z["ids"],ids)
        theta,lo,hi=(z[k] for k in ("theta","lo","hi"))
    assert theta.shape==(len(ids),8) and np.isfinite(theta).all() and np.all((theta>=0)&(theta<=1))
    assert lo.shape==hi.shape==(len(ids),) and np.all(np.minimum(hi,2)>np.maximum(lo,0))
    return relative.as_posix(),meta,theta,lo,hi


def apply_corrected(root, payload, errors):
    """Replace published supervised records after all 12 runs at a resolution are present."""
    widths = complete_grids(root)
    if not widths:
        return None
    original_grid=payload["grid_comparison"]
    prior_equal_csv=payload["equal_budget"]["csv"]
    if payload.get("corrected_noisy") and not set(payload["corrected_noisy"]["completed_widths"])<={w for w,_ in widths}:
        raise ValueError("An already published corrected grid is missing its full set of predictions")
    ids=np.asarray([m["id"] for m in payload["materials"]])
    truth=np.asarray([m["truth"] for m in payload["materials"]])
    assert len(ids)==389 and len(set(ids))==389
    full=len(widths)==len(GRIDS)
    clean_scores=clean_reference(root,ids,truth,errors)
    clean_reference_rows=[r for r in original_grid["rows"] if r["domain"]=="clean"]
    if len(clean_reference_rows)!=60:raise ValueError("The full report must include all 60 original clean supervised fits")
    for row in clean_reference_rows:
        scores=clean_scores[f"{row['space_bins']}_{row['currents']}_{row['family']}_{row['seed']}"]
        np.testing.assert_allclose(np.median(scores,axis=1),[row["own_D_median_pct"],row["own_t_median_rmse"]],rtol=1.e-10)
        if row["seed"]==0:
            slug=slug_for(row["space_bins"],row["family"])
            own=np.asarray(payload["runs"][f"clean__{row['currents']}__{slug}"]["observations"])
            np.testing.assert_array_equal(own[:,0],ids)
            np.testing.assert_allclose(scores,own[:,1:3].T,rtol=1.e-10)
    saved, bounds, prep = {},{},{}
    for w,t in widths:
        meta=json.loads((root/FOLDER/f"grid{w}"/"preparation.json").read_text(encoding="utf-8"))
        assert meta["dataset_revision"]==REVISION and meta["source_grid"]==[w,t] and meta["eligible"]==1947
        assert meta["split"]=={"train":1402,"val":156,"test":389}
        prep[str(w)] = meta
        for n in (1,3):
            for family in ("mlp","fno"):
                for seed in range(3):
                    fit=read_fit(root,w,t,family,n,seed,meta,ids)
                    saved[w,n,family,seed]=fit
                    interval=bounds.setdefault((w,n),fit[3:])
                    for a,b in zip(fit[3:],interval):np.testing.assert_array_equal(a,b)
    for bank in (payload["runs"],payload["seeds"]):
        for key in list(bank):
            if key.startswith(DOMAIN+"__") and key.split("__",2)[-1] in BASE_SLUGS:
                del bank[key]
    rows, primary = [],{}
    for n in (1,3):
        if full:
            common_lo=np.maximum(0,np.maximum.reduce([bounds[w,n][0] for w,_ in widths]))
            common_hi=np.minimum(2,np.minimum.reduce([bounds[w,n][1] for w,_ in widths]))
            assert np.all(common_hi>common_lo)
        for family in ("mlp","fno"):
            for seed in range(3):
                for w,t in widths:
                    stem,meta,theta,lo,hi=saved[w,n,family,seed]
                    de,te=errors(theta,truth,lo,hi)
                    model=meta["architecture"];slug=slug_for(w,family);key=f"{DOMAIN}__{n}__{slug}"
                    per_s=meta["test_s"]/len(ids)
                    clean_de,clean_te=clean_scores[f"{w}_{n}_{family}_{seed}"]
                    if full:
                        dc,tc=errors(theta,truth,common_lo,common_hi)
                        if w==20:original_dc,original_tc=dc,tc
                    row=dict(domain=DOMAIN,currents=n,family=family,seed=seed,space_bins=w,max_frames=t,model=model,
                             n_train=1402,n_val=156,n_test=len(ids),parameters=meta["parameters"],
                             own_D_median_pct=float(np.median(de)),own_t_median_rmse=float(np.median(te)),
                             own_median_c_span=float(np.median(np.minimum(2,hi)-np.maximum(0,lo))),
                             common_grids_space_x_time="|".join(f"{x}x{y}" for x,y in GRIDS) if full else "",
                             common_D_median_pct=float(np.median(dc)) if full else None,
                             common_t_median_rmse=float(np.median(tc)) if full else None,
                             common_median_c_span=float(np.median(common_hi-common_lo)) if full else None,
                             n_common_bounds_equal_20=int(np.sum((common_lo==np.maximum(0,bounds[20,n][0])) & (common_hi==np.minimum(2,bounds[20,n][1])))) if full else None,
                             common_paired_D_change_vs_20_pctpt=float(np.median(dc-original_dc)) if full else None,
                             common_paired_t_change_vs_20=float(np.median(tc-original_tc)) if full else None,
                             train_s=meta["train_s"],test_s=meta["test_s"],mean_inference_s_per_material=per_s,
                             inference_batch=meta["test_batch"],inference_repeats=meta["test_repeats"],device=meta["device"],
                             result_path=stem,dataset_revision=REVISION,source_folder=meta["source_folder"],input_sha256=meta["input_sha256"],
                             checkpoint_sha256=meta["checkpoint_sha256"],epoch_metrics_sha256=meta["epoch_metrics_sha256"],
                             best_epoch=meta["best_epoch"],best_val_coef_mse=meta["best_val_coef_mse"],
                             epochs=meta["epochs"],optimizer=meta["optimizer"],batch_size=meta["batch_size"],
                             learning_rate=meta["learning_rate"],weight_decay=meta["weight_decay"],scheduler=meta["scheduler"],
                             training_precision=meta["training_precision"],cpu_threads=meta["cpu_threads"],
                             epoch_evaluation_s=meta["epoch_evaluation_s"],
                             paired_own_D_change_vs_clean_pctpt=float(np.median(de-clean_de)),
                             paired_own_t_change_vs_clean=float(np.median(te-clean_te)))
                    rows.append(row)
                    score=dict(seed=seed,d=row["own_D_median_pct"],t=row["own_t_median_rmse"],train=meta["train_s"],test=per_s)
                    payload["seeds"].setdefault(key,[]).append(score)
                    if seed==0:
                        obs=np.column_stack((ids,de,te,lo,hi,theta)).tolist()
                        for o,mid in zip(obs,ids):o[0]=int(mid)
                        run=dict(name=model_name(w,t,family),slug=slug,domain=DOMAIN,count=n,n=len(ids),family="Supervised and TTT",
                                 meta=meta,training=meta["train_s"],tuning=0.,stage_s=per_s,init_s=0.,per_s=per_s,
                                 individual_times=None,observations=obs)
                        payload["runs"][key]=run
                        primary[n,slug]=summary(run,np.array(obs),{},[],payload["equal_budget"]["shared_ids"])
    clean=[dict(r, dataset_revision="original_clean") for r in original_grid["rows"] if r["domain"]=="clean"]
    pieces=sorted(clean+rows,key=lambda r:(r["domain"],r["currents"],r["family"],r["seed"],r["space_bins"]))
    fields=list(csv.reader(io.StringIO(prior_equal_csv)))[0]
    retained=[r for r in csv.DictReader(io.StringIO(prior_equal_csv)) if r["domain"]=="clean"]
    retained.extend(r for (n,slug),r in primary.items() if slug in ("mlp","fno"))
    payload["equal_budget"]["csv"]=write_rows(retained,fields)
    old_add=list(csv.DictReader(io.StringIO(original_grid["accuracy_csv_append"]),fieldnames=fields))
    fine_add=[r for r in old_add if r["domain"]=="clean"]+[r for (n,slug),r in primary.items() if slug not in ("mlp","fno")]
    figures={k:v for k,v in original_grid["figures"].items() if k.startswith("clean__")}
    if full:figures.update(grid_figures(rows))
    payload["grid_comparison"]={**original_grid,"rows":pieces,"csv":write_rows(pieces),"figures":figures,
                               "accuracy_csv_append":write_rows(fine_add, fields, header=False)}
    solved=replace_solvers(root,payload,errors,widths)
    extras=[v.get("accuracy_csv_append","") for k in ("fine_df","fine_cv","fine_cv_width","fine_cv_budget","fine_cv_300","fine_cv_w001") if (v:=payload.get(k))]
    all_text=payload["equal_budget"]["csv"]+payload["grid_comparison"]["accuracy_csv_append"]+"".join(extras)
    all_rows=list(csv.DictReader(io.StringIO(all_text)))
    all_rows=[r for r in all_rows if r["domain"]=="clean" or r["method"] in BASE_SLUGS or
              (r["method"] in SOLVER_SLUGS and int(r["currents"]) in solved)]
    source_grids={(r["domain"],f"grid290_{r['optimizer']}_{r['start']}"):r["initialization_grid"]
                  for r in payload["fine_df"]["conditions"]}
    for r in all_rows:
        r["dataset_revision"]="original_clean" if r["domain"]=="clean" else REVISION
        r["initialization_grid"]=source_grids.get((r["domain"],r["method"]),"")
    corrected=dict(revision=REVISION,completed_widths=[w for w,_ in widths],all_shared_ranges=full,
                   preparation=prep,supervised_csv=write_rows(rows),all_accuracy_csv=write_rows(all_rows),
                   derivative_free_current_counts=solved)
    inventory=root/FOLDER/"source_inventory.json"
    if inventory.exists():
        verified=json.loads(inventory.read_text(encoding="utf-8"))
        assert verified["dataset_revision"]==REVISION
        for w,_ in widths:
            assert verified["grids"][str(w)]["indexed_source_sha256"]==prep[str(w)]["source_sha256"]
        corrected["source_inventory"]=verified
    payload["corrected_noisy"]=corrected
    return corrected


def corrected_exports(payload):
    grid=payload["grid_comparison"]
    revision=payload["corrected_noisy"]
    simple=[]
    for r in payload["runs"].values():
        md=r["meta"]
        if r["domain"]!= "clean" and md.get("dataset_revision")!=REVISION:continue
        o=np.asarray(r["observations"])
        simple.append(dict(domain=r["domain"],currents=r["count"],method=r["slug"],n_test=r["n"],
            D_median_pct=float(np.median(o[:,1])),t_median_rmse=float(np.median(o[:,2])),
            median_observed_c_span=float(np.median(np.minimum(2,o[:,4])-np.maximum(0,o[:,3]))),
            train_s=r["training"],validation_search_s=r["tuning"],stage_test_s=md["test_s"],
            pipeline_mean_s_per_material=r["per_s"],gate_status=md.get("gate_status",""),checkpoint=md.get("checkpoint",""),
            dataset_revision=REVISION if r["domain"]==DOMAIN else "original_clean"))
    for r in payload["fine_df"]["conditions"]:
        simple.append(dict(domain=r["domain"],currents=r["currents"],method=f"grid290_{r['optimizer']}_{r['start']}",
            n_test=r["n_test"],D_median_pct=r["refined_D_median_pct"],t_median_rmse=r["refined_t_median_rmse"],
            median_observed_c_span=r["median_observed_c_span"],train_s=r["shared_prior_training_s"],
            validation_search_s=r["prior_validation_search_s"],stage_test_s=r["fit_worker_seconds_sum"],
            pipeline_mean_s_per_material=r["pipeline_s_per_material"],gate_status="",
            checkpoint=r.get("initializer_checkpoint_sha256",""),dataset_revision=r["dataset_revision"],initialization_grid=r["initialization_grid"]))
    exports={"grid_accuracy_cost.csv":grid["csv"],"corrected_supervised_accuracy_cost.csv":revision["supervised_csv"],
            "all_accuracy_cost.csv":revision["all_accuracy_csv"],"equal_budget_accuracy_cost.csv":payload["equal_budget"]["csv"],
            "accuracy_cost.csv":write_rows(simple),"fine_df_accuracy_cost.csv":payload["fine_df"]["csv"]}
    fine=payload["fine_df"]
    materials={m["id"]:m for m in payload["materials"]}
    rows=[]
    for r in fine["conditions"]:
        prefix=f"{r['domain']}__{r['currents']}"
        for mid,b,begin,end in zip(fine["ids"],fine["cohorts"][prefix],fine["before"][prefix+"__"+r["start"]],
                                  fine["after"][prefix+"__"+r["optimizer"]+"_"+r["start"]]):
            d,t,c,s,e,stop=end
            m=materials[mid]
            rows.append(dict(domain=r["domain"],currents=r["currents"],optimizer=r["optimizer"],start=r["start"],
                material_id=mid,current_indices=" ".join(map(str,m["chosen"][:r["currents"]])),
                dimensionless_currents=" ".join(map(str,m["iota"][:r["currents"]])),observed_c_min=b[0],observed_c_max=b[1],
                scored_c_min=max(0,b[0]),scored_c_max=min(2,b[1]),initial_D_pct=begin[0],refined_D_pct=d,
                initial_t_rmse=begin[1],refined_t_rmse=t,initial_c_rmse=begin[2],refined_c_rmse=c,
                initialization_mean_s=r["initialization_s_per_material"],refinement_worker_s=s,
                pipeline_s_including_mean_init=s+r["initialization_s_per_material"],evaluations=e,status=fine["stops"][stop],
                dataset_revision=r["dataset_revision"],initialization_grid=r["initialization_grid"]))
    exports["fine_df_materials.csv"]=write_rows(rows)
    return exports


def add_corrected_view(root, html):
    start,end="/* BEGIN CORRECTED NOISE */","/* END CORRECTED NOISE */"
    script=start+"\n"+(root/"corrected_noise_view.js").read_text(encoding="utf-8")+"\n"+end
    if start in html:
        html=html[:html.index(start)]+script+html[html.index(end)+len(end):]
    def replace(old,new):
        nonlocal html
        if new in html:return
        if html.count(old)!=1:raise ValueError(f"Missing corrected-data view anchor: {old}")
        html=html.replace(old,new)
    replace('<option value="gaussian_noisy">Gaussian noise</option>',
            '<option value="gaussian_noisy">Gaussian noise · corrected</option>')
    replace('<div id="content"></div>', '<aside id="revisionNotice" class="summary" aria-live="polite"></aside><div id="content"></div>')
    replace('const DB=JSON.parse(document.querySelector("#benchmark").textContent),',
            'const PUBLISHED=JSON.parse(document.querySelector("#benchmark").textContent), DB=visibleNoiseStudies(PUBLISHED),')
    replace('const run=(s,c)=>runs[dom()+"__"+c+"__"+s]', 'const run=(s,c)=>activeNoiseRun(s,c)')
    replace('pipeline_s_including_mean_init,evaluations,status"];','pipeline_s_including_mean_init,evaluations,status,dataset_revision,initialization_grid"];')
    replace('e,fine.stops[stop]].join(",")', 'e,fine.stops[stop],r.dataset_revision,r.initialization_grid].join(",")')
    replace('let h=title("Property accuracy versus inference cost")','if(revisionCorrected())revisionSolverPoints(points,idx);\nlet h=title("Property accuracy versus inference cost")')
    replace("function finePlot(){", "function finePlot(){if(revisionCorrected()&&DB.corrected_noisy.derivative_free_current_counts.length===1)return revisionSingleCountPlot();")
    if start not in html:replace("function groupTable(", script+"\nfunction groupTable(")
    replace("function refreshMaterials(){","function refreshMaterials(){revisionMethodMenu();")
    replace('    const earlier=', '    if(renderNoiseRevision())return;\n    const earlier=')
    replace('    if(tab==="fine-grid" && byId("fineMaterial")){', '    revisionCleanLabels();\n    if(tab==="fine-grid" && byId("fineMaterial")){')
    replace("function downloadEqualBudget(){", "function downloadEqualBudget(){if(DB.corrected_noisy)return revisionDownload(DB.corrected_noisy.all_accuracy_csv,'all_accuracy_cost.csv');")
    return html
