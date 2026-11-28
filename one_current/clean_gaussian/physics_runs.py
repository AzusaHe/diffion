"""Fit numerical inverse methods on paired source observations; resume completed fits.

Run from diffion-ms with python -m one_current.clean_gaussian.physics_runs.
The optional --fields flag only regenerates the first-eight PINN seed fields.
Install numpy, scipy, numba, cma, scikit-optimize and scikit-learn.
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
import time
import warnings

for key in ("MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np
from scipy.optimize import minimize
from inverse import baselines
from .physics_fv import objective, value_grad, simulate

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
CACHE = HERE / "cache"
STORE = CACHE / "physics_fits"
DOMAINS = ("clean","gaussian_noisy")
METHODS = ("lbfgs","nm","cmaes","bo")

@lru_cache(maxsize=1)
def data():
    with np.load(CACHE/"binned.npz") as z:
        take = z["split"]=="test"
        return {k:np.ascontiguousarray(z[k][take]) for k in ("ids","iota","mask","tau","clean","gaussian_noisy")}

def observations(domain, count, row):
    d = data()
    return (np.ascontiguousarray(d["iota"][row,:count],dtype=float),
            np.ascontiguousarray(d["tau"][row,:count],dtype=float),
            np.ascontiguousarray(d["mask"][row,:count]),
            np.ascontiguousarray(d[domain][row,:count],dtype=float))

def starts(domain, count, kind, allow_first8=False):
    if kind == "center":
        return np.full((len(data()["ids"]),8), 0.5)
    path = RES/f"{domain}__{count}__{kind}.npz"
    if allow_first8 and not path.exists():
        path = path.with_stem(path.stem+"_first8")
    if not path.exists():
        return None
    with np.load(path) as z:
        ids,theta=z["ids"],z["theta"].astype(float)
    by_id=dict(zip(ids,theta))
    desired=data()["ids"][:len(ids)]
    if set(desired)!=set(ids):
        raise ValueError("Warm-start IDs do not match held-out materials")
    return np.clip(np.asarray([by_id[mid] for mid in desired]),1e-6,1-1e-6)

def seed_fields():
    for domain in DOMAINS:
        for count in (1,3):
            path = CACHE/f"{domain}__{count}__seedfield.npz"
            if path.exists():
                continue
            start=starts(domain,count,"fno_ttt",True)
            if start is None:
                continue
            fields=[]
            # Remove one-time Numba compilation from the per-material cost.
            p=observations(domain,count,0)
            simulate(start[0],*p[:3])
            clock=time.perf_counter()
            for row in range(8):
                p=observations(domain,count,row)
                fields.append(np.where(p[2][...,None],simulate(start[row],*p[:3]),np.nan))
            temp=path.with_suffix(".tmp.npz")
            np.savez(temp,ids=data()["ids"][:8],field=fields,elapsed_s=time.perf_counter()-clock)
            temp.replace(path)
            print("Seed field ready:",path.name,flush=True)

def init_worker():
    warnings.simplefilter("ignore",UserWarning)
    import cma, skopt
    p=observations("clean",1,0)
    value_grad(np.full(8,0.5),*p)

def fit(task):
    domain,count,method,start,row,theta0=task
    suffix=f"{domain}__{count}__{method}_{start}"
    dest=STORE/suffix/f"{row:04d}.json"
    if dest.exists():
        return
    p=observations(domain,count,row)
    # Penalty for a nonfinite trajectory at an infeasible box trial.
    def rmse(theta):
        try:
            cost=objective(theta,*p)
            return np.sqrt(cost) if np.isfinite(cost) and cost>=0 else 1.
        except (OverflowError,ZeroDivisionError):
            return 1.
    clock=time.perf_counter()
    if method=="lbfgs":
        best=[float("inf"),theta0.copy()]
        def loss(theta):
            try:
                value,gradient=value_grad(theta,*p)
            except (OverflowError,ZeroDivisionError):
                return 1e8,np.zeros(8)
            if not np.isfinite(value) or not np.isfinite(gradient).all():
                return 1e8,np.zeros(8)
            if value < best[0]:
                best[:]=[value,theta.copy()]
            return value*1e8,gradient*1e8
        result=minimize(loss,theta0,method="L-BFGS-B",jac=True,bounds=[(0.,1.)]*8,
                        options=dict(maxiter=150,maxfun=220,ftol=1e-11,gtol=1e-5,maxls=20))
        pred=best[1]
        cost=float(np.sqrt(best[0]))
        evals=int(result.nfev)
        status=str(result.message)
    else:
        limit=35 if method=="bo" else 500
        out,_=baselines.run(method.upper(),rmse,theta0,int(data()["ids"][row]),1e-4,limit,300.,lambda z:z)
        pred=np.asarray(out["theta_best"])
        cost=out["best_rmse_M"]
        evals=out["total_evals"]
        status=out["status"]
    elapsed=time.perf_counter()-clock
    dest.parent.mkdir(parents=True,exist_ok=True)
    value=dict(id=int(data()["ids"][row]),theta=pred.tolist(),lo=float(np.nanmin(p[3])),hi=float(np.nanmax(p[3])),
               elapsed_s=elapsed,evals=evals,c_rmse=cost,status=status)
    temp=dest.with_suffix(".tmp")
    temp.write_text(json.dumps(value,allow_nan=False),encoding="utf-8")
    temp.replace(dest)

def aggregate(domain,count,method,start,n):
    slug=f"{method}_{start}"
    prefix=f"{domain}__{count}__{slug}"
    files=[STORE/prefix/f"{row:04d}.json" for row in range(n)]
    if not all(f.exists() for f in files):
        return False
    records=[json.loads(p.read_text()) for p in files]
    np.testing.assert_array_equal([r["id"] for r in records],data()["ids"][:n])
    seconds=np.asarray([r["elapsed_s"] for r in records])
    numeval=[r["evals"] for r in records]
    bounds=lambda key:np.asarray([r[key] for r in records])
    temp=RES/f"{prefix}.tmp.npz"
    np.savez(temp,ids=bounds("id"),theta=bounds("theta"),lo=bounds("lo"),hi=bounds("hi"),
             elapsed_s=seconds,evals=numeval,concentration_rmse=bounds("c_rmse"))
    temp.replace(RES/f"{prefix}.npz")
    options={"lbfgs":"150 iterations, 220 value/gradient evaluations",
             "nm":"500 evaluations", "cmaes":"500 evaluations","bo":"35 evaluations, 16 initial random points, expected improvement"}
    detail=dict(train_s=0.,n_train=0,n_test=n,test_s=float(seconds.sum()),median_fit_s=float(np.median(seconds)),
                device="CPU",median_evals=float(np.median(numeval)),median_c_rmse=float(np.median(bounds("c_rmse"))),
                description=f"40-cell conservative finite-volume solver with an implicit midpoint corrector, fit to 20 bin averages. {options[method]}.",
                timing="Sum of elapsed per-material fitting times in concurrent CPU workers, excluding compilation and warm-start prediction.",
                warm_start=(None if start=="center" else start),budget=options[method],
                statuses={str(s):int(np.sum(bounds("status")==s)) for s in np.unique(bounds("status"))})
    (RES/f"{prefix}.json").write_text(json.dumps(detail,indent=2),encoding="utf-8")
    print("Wrote",prefix,n,round(seconds.sum(),1),flush=True)
    return True

def run(workers,cohort):
    STORE.mkdir(exist_ok=True,parents=True)
    RES.mkdir(exist_ok=True)
    pending={}
    for d in DOMAINS:
        for j in (1,3):
            for m in METHODS:
                for s in (("center","fno","mlp","fno_ttt","mlp_ttt") if m=="lbfgs" else ("center","fno_ttt")):
                    n=30 if m=="bo" else cohort
                    pending[d,j,m,s]=n
    with ProcessPoolExecutor(max_workers=workers,initializer=init_worker) as pool:
        from concurrent.futures import FIRST_COMPLETED,wait
        futures={}
        known=set()
        last=0
        while pending or futures:
            if time.monotonic()-last>5:
                seed_fields()
                for key,n in list(pending.items()):
                    result=RES/("__".join([key[0],str(key[1]),key[2]+"_"+key[3]])+".json")
                    if result.exists() and json.loads(result.read_text()).get("n_test")==n:
                        del pending[key]
                        continue
                    d,j,m,s=key
                    folder=STORE/f"{d}__{j}__{m}_{s}"
                    if len(list(folder.glob("*.json")))>=n and aggregate(*key,n):
                        del pending[key]
                last=time.monotonic()
            if len(futures)<workers*2:
                available=[]
                for key,n in pending.items():
                    d,j,m,s=key
                    z=starts(d,j,s)
                    if z is not None:
                        available.append((key,n,z))
                pushed=0
                for row in range(max(pending.values(),default=0)):
                    for key,n,z in available:
                        if (key,row) in known or row>=min(n,len(z)):
                            continue
                        known.add((key,row))
                        d,j,m,s=key
                        if (STORE/f"{d}__{j}__{m}_{s}"/f"{row:04d}.json").exists():
                            continue
                        futures[pool.submit(fit,(*key,row,z[row]))]=(key,row)
                        pushed+=1
                        if len(futures)>=workers*4:
                            break
                    if len(futures)>=workers*4:
                        break
            if futures:
                done,_=wait(futures,timeout=3,return_when=FIRST_COMPLETED)
                for f in done:
                    f.result()
                    del futures[f]
            elif pending:
                time.sleep(2)

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers",type=int,default=10)
    parser.add_argument("--cohort",type=int,default=300)
    parser.add_argument("--fields",action="store_true")
    args=parser.parse_args()
    if args.fields:
        seed_fields()
    else:
        run(args.workers,args.cohort)
