"""Paired variable-time PINN fits using the project's c-only and c,v PINN kernels."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from one_current.pinn_runs import solver_pinn as engine
from pinn.material import alpha_V

OUT = HERE/"results"
CACHE = HERE/"cache"
RAW = HERE.parents[2]/"latest_synthetic_ml"/"data"
METHODS = ("standard_c", "standard_cv", "staged_c_w1", "staged_cv_w1", "staged_c_w100",
           "staged_cv_w100", "data_first", "fine", "data_correction", "solver_correction", "solver_correction_weight1")
DTYPE = torch.float32


def statistics(y, ios, times):
    mask = np.isfinite(y[..., 0])
    result = np.full_like(y, np.nan)
    derivs = []
    for j, io in enumerate(ios):
        mj = mask[j]
        t, c = times[j,mj], y[j,mj].astype(float)
        assert len(t)>=2 and np.all(np.diff(t)>0)
        slopes = np.diff(c,axis=0)/np.diff(t)[:,None]
        derivs.append(slopes.ravel())
        ct = np.vstack((slopes[0], (slopes[1:]+slopes[:-1])/2, slopes[-1]))
        V = alpha_V(torch.from_numpy(c))[1].numpy()
        cf = np.column_stack((c[:,0], (c[:,1:]+c[:,:-1])/2, c[:,-1]))
        total = io - np.column_stack((np.zeros(len(c)), np.cumsum(ct,axis=-1)))/c.shape[-1]
        v = np.zeros_like(total)
        for k in range(c.shape[-1]):
            qprev = total[:,k]-cf[:,k]*v[:,k]
            v[:,k+1] = (v[:,k]-V[:,k]*(total[:,k+1]-qprev))/(1-V[:,k]*cf[:,k+1])
        result[j,mj] = (v[:,1:]+v[:,:-1])/2
    ends = np.max(np.where(mask,times,0.),axis=-1)
    return dict(y=np.nan_to_num(y), mask=mask, v_obs=np.nan_to_num(result), vmask=np.isfinite(result),
                iotas=ios,tmax=max(ends),end_times=ends,times=np.where(mask,times,0.),
                vscale=max(1e-5,np.nanmax(np.abs(result))),st=max(1e-5,np.sqrt(np.mean(np.concatenate(derivs)**2))),
                sigma=1e-3*np.nanmean(y),mean_c=np.nanmean(y),s_lo=0.0125)


def fine_inputs(ids, chosen, domain):
    y = np.full((*chosen.shape,104,40),np.nan,np.float32)
    times = np.zeros(y.shape[:-1],float)
    for b,mid in enumerate(ids):
        file=next((RAW/domain).glob(f"*/samples/material_{mid:06}.npz"))
        with np.load(file) as a:
            field=a["concentration_nd"]; valid=a["valid_time_mask"]; tau=a["tau"]
            for j,idx in enumerate(chosen[b]):
                avail=np.flatnonzero(valid[idx])
                k=min(104,len(avail))
                sample=avail[np.rint(np.linspace(0,len(avail)-1,k)).astype(int)]
                y[b,j,:k]=field[idx,sample].reshape(k,40,29).mean(-1,dtype=np.float64)
                times[b,j,:k]=tau[sample]
    return y,times


def make_corrections(params, stats, grid, device):
    """Freeze first-stage fields and add zero-initialized, residual-scaled sine networks."""
    ls = engine.Losses("cv",DTYPE,device,obs=grid)
    corrected=[]
    for i,st in enumerate(stats):
        base=engine._rebuild(params,iter([z[i].detach() for z in engine._leaves(params)]))
        state={k:torch.as_tensor(st[k],device=device,dtype=DTYPE) for k in ("tmax","vscale")}
        residual_c,residual_v=[],[]
        with torch.no_grad():
            for j,current in enumerate(st["iotas"]):
                mask=st["mask"][j]
                times=torch.tensor(st["times"][j,mask],dtype=DTYPE,device=device)
                io=torch.tensor(current,dtype=DTYPE,device=device)
                pred=torch.vmap(lambda t:ls.bin_average(base,t,io,state["tmax"]))(times).cpu().numpy()
                pv=torch.vmap(lambda t:torch.vmap(lambda x:engine.v_point(base,x,t,io,state["tmax"],state["vscale"]))(ls.XC))(times).cpu().numpy()
                residual_c.append(st["y"][j,mask]-pred)
                residual_v.append((st["v_obs"][j,mask]-pv)/st["vscale"])
        base={k:engine._rebuild(v,iter([z.cpu() for z in engine._leaves(v)])) for k,v in base.items() if k!="phi"}
        scale={}
        gen=torch.Generator().manual_seed(1000+i)
        for key,rs in (("c",residual_c),("v",residual_v)):
            eps=max(1e-6,np.sqrt(np.mean(np.concatenate(rs)**2)))
            kx=np.mean([np.sum(r[:,1:]*r[:,:-1]<0,axis=1).mean() for r in rs])
            kt=np.mean([np.sum(r[1:]*r[:-1]<0,axis=0).mean() for r in rs])
            scale["eps_"+key]=torch.tensor(eps,dtype=DTYPE)
            scale["k_"+key]=torch.tensor([np.clip(np.pi*kx/2,1,20),np.clip(np.pi*kt/2,1,20),1],dtype=DTYPE)
            net=engine.init_mlp(gen,[3,32,32,32,1],DTYPE)
            net[-1]=tuple(torch.zeros_like(q) for q in net[-1])
            base[key+"1"]=net
        base["ms"]=scale
        corrected.append(base)
    return corrected


def fit(domain,ncur,method,start,device,*,derivative_weight=None,force=False):
    slug="pinn_"+method+"_"+start
    out=OUT if derivative_weight is None else HERE/("derivative_results" if derivative_weight else "derivative_controls")
    out.mkdir(exist_ok=True)
    stem=out/f"{domain}__{ncur}__{slug}"
    if derivative_weight is None:
        weightsdir=CACHE/"pinn_fields"
    else:
        weightsdir=CACHE/("derivative_fields" if derivative_weight else "derivative_control_fields")
    needs_field=(derivative_weight is not None and method=="data_first" and start=="fno_ttt"
                 and not (weightsdir/f"{domain}__{ncur}__data__0.pt").exists())
    if stem.with_suffix(".json").exists() and not force and not needs_field: return True
    with np.load(CACHE/"binned.npz") as a:
        sel=np.flatnonzero(a["split"]=="test")[:8]
        ids=a["ids"][sel]; ions=a["iota"][sel,:ncur]; picks=a["chosen"][sel,:ncur]
        times=a["tau"][sel,:ncur]; y=a[domain][sel,:ncur]
    theta0=np.full((len(ids),8),.5)
    if start=="fno_ttt":
        fp=next((p for p in (OUT/f"{domain}__{ncur}__fno_ttt.npz",OUT/f"{domain}__{ncur}__fno_ttt_first8.npz") if p.exists()),None)
        if fp is None: return False
        with np.load(fp) as a:
            get=dict(zip(a["ids"],a["theta"]))
            theta0=np.array([get[mid] for mid in ids]).clip(1e-5,1-1e-5)
    solver=method.startswith("solver_correction")
    fieldfile=CACHE/f"{domain}__{ncur}__seedfield.npz"
    if solver and not fieldfile.exists(): return False
    if method=="fine": y,times=fine_inputs(ids,picks,domain)
    grid=engine.ObsGrid(np.arange(y.shape[2]),y.shape[-1],2)
    coarse=method!="fine"
    chunk=8 if coarse else (2 if ncur==1 else 1)
    kwargs=dict(width=32,depth=3,dtype=DTYPE,device=device,obs=grid,n_res=64,n_times=6,n_cells=12,
                steps=500,lbfgs_iters=0,log_every=5000,lbfgs_log_every=100)
    build="c" if method=="standard_c" or method.startswith("staged_c_") else "cv"
    if method.startswith("staged"):
        weight=float(method.split("_w")[1])
        kwargs.update(fit_steps=350,warmup_steps=100,steps=500,lr=1e-4,phi_lr_mult=10.,lbfgs_iters=16,
                      w=dict(vol=weight,bc=weight,pde=weight))
    if method in ("data_first","data_correction","fine","frozen"):
        kwargs.update(fit_steps=850,fit_lbfgs_iters=30,fit_v=True,steps=500,lr=1e-4,phi_lr_mult=10.,
                      lbfgs_iters=16,w=dict(data=.01,pde=1.,vol=1.,bc=1.))
    if method=="frozen":
        kwargs["freeze"]=("c","v")
    if derivative_weight is not None:
        kwargs.update(spatial_derivative_weight=derivative_weight,evaluate_spatial_derivatives=True)
    start_time=time.perf_counter()
    base_s=0.
    estimates=[]
    diagnostics={key:[] for key in ("c_slope_rmse","v_slope_rmse","c_slope_rel_rmse","v_slope_rel_rmse",
                                    "rel_rmse","v_rel_rmse","steps","evals")}
    weightsdir.mkdir(exist_ok=True)
    for k in range(0,len(ids),chunk):
        sl=slice(k,min(k+chunk,len(ids))); stats=[statistics(c,io,t) for c,io,t in zip(y[sl],ions[sl],times[sl])]
        def train(th, opts, sem=stats):
            return engine.train_batch(y[sl],ions[sl],th,build,ids[sl],observation_stats=sem,return_params=True,**opts)
        options=kwargs.copy()
        if solver or method=="data_correction":
            # The solver seed is the original fixed forward solution. Observed slopes
            # supervise the correction only. Data-seeded bases inherit their arm's loss.
            basedir=CACHE/"pinn_fields" if solver else weightsdir
            wfile=basedir/f"{domain}__{ncur}__{'solver' if solver else 'data'}__{k}.pt"
            if not solver and not wfile.exists() and derivative_weight is not None:
                if not fit(domain,ncur,"data_first",start,device,derivative_weight=derivative_weight,force=True):
                    return False
                start_time=time.perf_counter()  # the checkpoint records the shared base cost
            if not wfile.exists():
                if not solver: return False
                with np.load(fieldfile) as f:
                    np.testing.assert_array_equal(ids,f["ids"])
                    synth=f["field"][sl].copy()
                    synth[~np.isfinite(y[sl])]=np.nan
                    field_s=float(f["elapsed_s"])
                source_stats=[statistics(c,io,t) for c,io,t in zip(synth,ions[sl],times[sl])]
                pre=kwargs|dict(fit_steps=850,fit_lbfgs_iters=30,fit_v=True,steps=0)
                pre.pop("spatial_derivative_weight",None)
                pre.pop("evaluate_spatial_derivatives",None)
                tick=time.perf_counter()
                _,params=train(theta0[sl],pre,source_stats)
                # Base solver prefit is shared between the two correction weight settings.
                torch.save(dict(params=params,theta=theta0[sl],seconds=time.perf_counter()-tick+field_s),wfile)
                start_time=time.perf_counter()
            ck=torch.load(wfile,map_location=device)
            init=make_corrections(ck["params"],stats,grid,device)
            base_s+=ck["seconds"]
            options.update(fit_steps=0,fit_lbfgs_iters=0,steps=500,lbfgs_iters=16,lr=1e-3,phi_lr_mult=1.,fit_v=True,
                        freeze=("c","v","ms"),init_params=init,w=dict(data=1. if method.endswith("weight1") else .01,pde=1.,vol=1.,bc=1.))
            rows,_=train(ck["theta"],options)
        else:
            tick=time.perf_counter()
            rows,params=train(theta0[sl],options)
            if method=="data_first" and start=="fno_ttt":
                torch.save(dict(params=params,theta=np.array([r["theta"] for r in rows]),seconds=time.perf_counter()-tick),
                           weightsdir/f"{domain}__{ncur}__data__{k}.pt")
        estimates.extend(r["theta"] for r in rows)
        if derivative_weight is not None:
            schedule=rows[0]["schedule"]
            expected_steps=sum(schedule[key] for key in ("fit_steps","fit_lbfgs_iters","warmup_steps","joint_steps","lbfgs_iters"))
            if any(row["stopped_by_wall"] or row["steps"]!=expected_steps for row in rows):
                raise RuntimeError(f"Incomplete optimizer budget for {stem}")
            for key,values in diagnostics.items():
                values.extend(row[key] for row in rows)
    wall=time.perf_counter()-start_time+base_s
    assert np.isfinite(estimates).all()
    extra={}
    if derivative_weight is not None:
        extra.update(diagnostics)
        for key,value in extra.items():
            if not np.isfinite(value).all(): raise ValueError(f"{stem}: nonfinite {key}")
    np.savez(stem.with_suffix(".npz"),ids=ids,theta=estimates,lo=np.nanmin(y,axis=(1,2,3)),hi=np.nanmax(y,axis=(1,2,3)),**extra)
    stages={k:v for k,v in options.items() if k in ("fit_steps","fit_lbfgs_iters","warmup_steps","steps","lbfgs_iters","lr","phi_lr_mult","w")}
    md=dict(train_s=0.,test_s=wall,n_train=0,device=device,test_batch=chunk,
            description=f"Independent per-material {build} PINN; 32-unit, 3-hidden-layer fields, 2-point cell quadrature. "+
                        ("Up to 104 valid frames and 40 bins. " if not coarse else "Up to 13 valid frames and 20 bins. ")+
                        f"Budgeted rerun: {json.dumps(stages)}.",shared_field_fit_s=base_s)
    if start!="center": md["warm_start"]=start
    if derivative_weight is not None:
        md.update(spatial_derivative_weight=derivative_weight,weights=rows[0]["weights"],schedule=rows[0]["schedule"],
                  c_derivative_target="adjacent differences of observed concentration bin means",
                  v_derivative_target="adjacent differences of velocity reconstructed from observed concentration",
                  current_indices=picks.tolist(),material_ids=ids.tolist(),gpu=torch.cuda.get_device_name(device) if str(device).startswith("cuda") else "CPU")
        if method=="data_correction" or solver: md["field_source"]=wfile.relative_to(HERE).as_posix()
    if method=="data_correction": md["description"]+=" Includes a data-first PINN followed by a frozen-field correction."
    if solver: md["description"]+=" Includes forward solve, 850 Adam and 30 L-BFGS field-prefit steps, then a frozen-field correction."
    stem.with_suffix(".json").write_text(json.dumps(md,indent=2))
    print(domain,ncur,slug,round(wall,2),"s",flush=True)
    return True


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--device",default="cuda")
    p.add_argument("--methods",nargs="+",default=METHODS,choices=METHODS)
    p.add_argument("--domains",nargs="+",default=["clean","gaussian_noisy"])
    p.add_argument("--currents",nargs="+",type=int,default=[1,3])
    p.add_argument("--starts",nargs="+",default=["center","fno_ttt"])
    args=p.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(0)
    os.environ.setdefault("DIFFION_PINN_GRAPHS","1")
    OUT.mkdir(exist_ok=True)
    for method in args.methods:
        for domain in args.domains:
            for currents in args.currents:
                for start in args.starts:
                    if method in ("data_correction","solver_correction","solver_correction_weight1") and start=="center": continue
                    fit(domain,currents,method,start,args.device)
