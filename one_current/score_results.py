"""Recompute common-support property errors and a blinded, fixed-current forward check.

Invoke from diffion-ms with python -m one_current.score_results. Only complete
method cohorts are summarized. Material-level scores are indexed by source ID.
"""
import csv
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parent / "random"
OUT=ROOT / "report_data"


def predict(theta, current):
    from solver.dataset import simulate_batch
    values=[]; masks=[]
    for j in range(0,len(theta),32):
        ix=np.minimum(np.arange(j,j+32),len(theta)-1)
        pred,valid=simulate_batch(theta[ix],current[ix,None])
        n=min(32,len(theta)-j)
        values.append(np.asarray(pred)[:n,0]);masks.append(np.asarray(valid)[:n,0])
    return np.concatenate(values),np.concatenate(masks)


def curves(theta, truth, bounds):
    c=bounds[:,0,None]+np.linspace(0,1,301)[None,:]*np.diff(bounds,axis=1)
    u=np.clip(c/2,0,1)
    b=np.stack(((1-u)**3,3*u*(1-u)**2,3*u**2*(1-u),u**3),axis=-1)
    def props(z):
        return (0.1*np.exp(np.log(20)*np.einsum("npk,nk->np",b,z[:,:4])),
                -0.25+1.25*np.einsum("npk,nk->np",b,z[:,4:]))
    d,t=props(theta);dt,tt=props(truth)
    return 100*np.linalg.norm(d-dt,axis=1)/np.linalg.norm(dt,axis=1),np.sqrt(np.mean((t-tt)**2,axis=1))


def groups(currents):
    for arch in ("fno","mlp"):
        for seed in range(3):
            for label in ("supervised","ttt"):
                p=ROOT/"learned"/f"{arch}_s{seed}"/f"{label}.npz"
                if p.is_file():
                    with np.load(p) as f:
                        yield f"{arch}_{label}_s{seed}",f["index"],f["theta"]
    def read(label, folder, n):
        files=sorted(folder.glob("m[0-9][0-9][0-9].json"))
        if len(files)==n:
            rows=[json.loads(p.read_text()) for p in files]
            saved=np.asarray([row["iotas"] for row in rows])
            # Equality also checks shape: each fit must have exactly one
            # current, matching the per-material draw used by the networks.
            np.testing.assert_array_equal(saved, currents[:n,None])
            yield label,np.array([r["index"] for r in rows]),np.array([r["theta"] for r in rows])
    for p in sorted((ROOT/"solver_runs/results/lbfgs").glob("*")):
        yield from read("lbfgs_"+p.name,p,300)
    for p in sorted((ROOT/"solver_runs/results/gradfree").glob("*/*")):
        yield from read(p.name+"_"+p.parent.name,p,30 if p.name=="BO" else 300)
    for p in sorted((ROOT/"pinn_runs/results").glob("*/*")):
        yield from read("pinn_"+p.parent.name+"_"+p.name,p,8)


def save_csv(path,rows):
    if not rows: return
    tmp=path.with_suffix(".tmp")
    with tmp.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    tmp.replace(path)


def score():
    from solver.dataset import DATA as ORIGINAL_DATA
    OUT.mkdir(exist_ok=True)
    ix=np.flatnonzero(np.load(ROOT/"data/train_split.npy")=="id_test")[:300]
    with np.load(ROOT/"data/train.npz") as single,np.load(ORIGINAL_DATA/"train.npz") as triple:
        ids=single["index"][ix]; np.testing.assert_array_equal(ids,triple["index"][ix])
        truths=single["theta"][ix]; io=single["iotas"][ix,0]; single_w=single["window"][ix]; common_w=triple["window"][ix]
        measured=single["y"][ix,0]; prefix=single["K"][ix]
    rank=np.load(ROOT/"data/current_rank.npy")[ix]
    truth_pred,truth_valid=predict(truths,np.ones(len(ix)))
    valid=np.logical_and.accumulate(truth_valid,axis=1)
    reference=np.where(valid[:,:,None],truth_pred,np.nan)
    def frac_rmse(pred,ref):
        keep=np.isfinite(ref)
        numerator=np.sum(np.where(keep,(pred-np.nan_to_num(ref))**2,0),axis=(1,2))
        return 100*np.sqrt(numerator/np.sum(keep,axis=(1,2)))/np.nanmean(ref,axis=(1,2))
    summary=[]; by_case=[]
    for label,case_ids,z in groups(io):
        n=len(z)
        np.testing.assert_array_equal(case_ids,ids[:n])
        assert z.shape==(n,8) and np.isfinite(z).all()
        d,t=curves(z,truths[:n],single_w[:n])
        dc,tc=curves(z,truths[:n],common_w[:n])
        held,okh=predict(z,np.ones(n))
        obs,oko=predict(z,io[:n])
        held_error=frac_rmse(held,reference[:n]);obs_error=frac_rmse(obs,measured[:n])
        oh=np.all(okh|~valid[:n],axis=1)
        oo=np.all(oko|(np.arange(13)[None,:]>=prefix[:n,None]),axis=1)
        for i in range(n):
            by_case.append(dict(method=label,material_id=int(case_ids[i]),rank=int(rank[i]),iota=io[i],K=int(prefix[i]),
                D_observed_pct=d[i],t_observed_rmse=t[i],D_common_pct=dc[i],t_common_rmse=tc[i],
                c_replay_pct=obs_error[i],c_unseen1_pct=held_error[i],replay_admissible=bool(oo[i]),unseen_admissible=bool(oh[i])))
        summary.append(dict(method=label,n=n,D_observed_pct=np.median(d),t_observed_rmse=np.median(t),
            D_common_pct=np.median(dc),t_common_rmse=np.median(tc),p95_D_common_pct=np.percentile(dc,95),
            c_replay_pct=np.median(obs_error),c_unseen1_pct=np.median(held_error),
            n_replay_admissible=int(oo.sum()),n_unseen_admissible=int(oh.sum())))
    save_csv(OUT/"methods.csv",summary);save_csv(OUT/"materials.csv",by_case)
    print("Scored",len(summary),"method/seed/start groups,",len(by_case),"estimates",flush=True)


if __name__=="__main__":
    score()
