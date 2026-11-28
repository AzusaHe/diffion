"""Fit the project PINN families to the same one-current pilot.

Run from the diffion-ms root: python -m one_current.pinn_runs.run --method standard_cv
The per-material files and summary tables are written under pinn_runs/results.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch

from analysis.metrics import score
from pinn import multistage
from . import data_correction_pinn, solver_pinn

CODE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("DIFFION_ONE_CURRENT_ROOT", str(CODE.parent / "random"))).resolve()
HERE = ROOT / "pinn_runs"
DATA = ROOT / "data"
RESULT = HERE / "results"
PREFIT = dict(fit_steps=30000, fit_lbfgs_iters=3000, fit_v=True)
JOINT = dict(steps=5000, lr=1e-4, phi_lr_mult=10.0, lbfgs_iters=300,
             w=dict(data=0.01, pde=1., vol=1., bc=1.))
MS2 = dict(steps=4000, lbfgs=1000, joint_steps=2000, joint_lbfgs=500,
           lr=1e-3, joint_lr=1e-4, width=64, depth=4)
METHODS = ("standard_cv", "standard_c", "data_first", "data_correction",
           "solver_correction", "solver_correction_weight1", "fine",
           "staged_cv_w100", "staged_c_w100", "staged_cv_w1", "staged_c_w1")


def atomic_json(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(record, indent=2), encoding="utf-8")
    temp.replace(path)


def load_cases():
    kept = np.flatnonzero(np.load(DATA / "train_split.npy") == "id_test")[:8]
    with np.load(DATA / "train.npz") as f:
        cases = {k: f[k][kept] for k in ("y", "K", "iotas", "theta", "window", "index")}
    if cases["iotas"].shape != (8, 1) or cases["y"].shape != (8, 1, 13, 20):
        raise ValueError("Expected exactly one current on each of eight pilot materials")
    return cases


def load_start(kind, ids):
    if kind == "center":
        return np.full((len(ids), 8), 0.5)
    path = ROOT / "learned" / "fno_s0" / "ttt_first8.npz"
    if not path.exists():
        path = path.with_name("ttt.npz")
    with np.load(path) as f:
        np.testing.assert_array_equal(f["index"][:len(ids)], ids)
        theta = f["theta"][:len(ids)].copy()
    if theta.shape != (len(ids), 8) or not np.all(np.isfinite(theta)):
        raise ValueError("Missing or malformed single-current FNO + TTT initialization")
    return np.clip(theta, 1e-4, 1 - 1e-4)


def progress(step, wall, rel, stage):
    print(f"{stage:12s} step {step:6d} | elapsed {wall:7.1f} s | median relative c RMSE {np.median(rel):.5g}",flush=True)


def solver_seed(cases, theta0, ks):
    """Use the current, horizon and solver at the learned estimate to initialize the correction."""
    from solver.dataset import W_BINS
    from solver.dataset_fine import SUB
    ions, ys = cases["iotas"][ks], cases["y"][ks]
    native, times = multistage.solver_field(theta0, ions, cases["K"][ks])
    frame = native[:, :, SUB-1::SUB] @ W_BINS.T
    grids = solver_pinn.ObsGrid.fine(times)
    cache = HERE / "cache" / "solver_field"
    res, network = solver_pinn.train_batch(
        native, ions, theta0, "cv", ks, steps=0, prefit_cache=cache, obs=grids,
        return_params=True, progress=progress, **PREFIT)
    base = [solver_pinn._rebuild(network, iter([z[j].detach().cpu() for z in solver_pinn._leaves(network)]))
            for j in range(len(ks))]
    scales = multistage.correction_scales(base, ys, ions, "cuda", torch.float64)
    params, info = [], []
    for j, k in enumerate(ks):
        gen = torch.Generator().manual_seed(1000 + int(k))
        sizes = [3, 32, 32, 32, 1]
        c1 = solver_pinn.init_mlp(gen, sizes, torch.float64)
        v1 = solver_pinn.init_mlp(gen, sizes, torch.float64)
        for net in (c1, v1):
            net[-1] = tuple(torch.zeros_like(z) for z in net[-1])
        pars = {key: torch.as_tensor(scales[j][key], dtype=torch.float64)
                for key in ("eps_c", "eps_v", "k_c", "k_v")}
        params.append(dict(c=base[j]["c"], v=base[j]["v"], ms=pars, c1=c1, v1=v1))
        info.append(dict(**scales[j],
                         solver_vs_data_rel_rmse=float(np.sqrt(np.nanmean((frame[j]-ys[j])**2))/np.nanmean(ys[j])),
                         native_field_fit_rmse=res[j]["fit_end_rel_rmse"], solver_stage_wall_s=res[j]["wall_s"]))
    return params, info


def save_fit(method, start, ks, fit, cases, theta0, y, extra=None):
    sub = RESULT / method / start
    for j, (k, item) in enumerate(zip(ks, fit)):
        item.update(method=method, start=start, k=k, index=int(cases["index"][k]),
                    theta0=theta0[j].tolist(), iotas=cases["iotas"][k].tolist(),
                    K=int(cases["K"][k]), grid_shape=list(y[j].shape), window=cases["window"][k].tolist(),
                    property_scores=score(item["theta"], cases["theta"][k], cases["window"][k]))
        if extra:
            item["solver_seed"] = extra[j]
        atomic_json(sub / f"m{k:03d}.json", item)
    records = [sub / f"m{k:03d}.json" for k in range(8)]
    if not all(path.exists() for path in records):
        return
    rows = [json.loads(path.read_text()) for path in records]
    med = {metric: float(np.median([r["property_scores"][metric] for r in rows]))
           for metric in rows[0]["property_scores"]}
    summary = dict(n=8, method=method, start=start, median_fit=med,
                   median_relative_c_rmse=float(np.median([r["rel_rmse"] for r in rows])))
    atomic_json(sub / "summary.json", summary)
    print(summary, flush=True)


def architecture_schedule(method):
    """Original schedules for the two vanilla and two staged PINN builds."""
    if method.startswith("standard_"):
        return method.split("_")[-1], {}
    _, build, weight = method.split("_")
    w = float(weight[1:])
    return build, dict(fit_steps=3000, warmup_steps=1000, steps=5000, lr=1e-4,
                       phi_lr_mult=10., lbfgs_iters=300, w=dict(pde=w, vol=w, bc=w),
                       prefit_cache=HERE / "cache" / "staged")


def paired(method, batch_size):
    """Fit incomplete generic and learned starts in the same batch as the reference experiments."""
    if not method.startswith(("standard_", "staged_")):
        raise ValueError("Paired batching is supported for the standard and staged ensembles")
    cases = load_cases()
    jobs = []
    for start in ("center", "fno_ttt"):
        theta = load_start(start, cases["index"])
        for k in range(8):
            if not (RESULT / method / start / f"m{k:03d}.json").exists():
                jobs.append((start, k, theta[k]))
    if not jobs:
        return
    # Independent fits share no weights. Keep the batch below the full 16 on a busy GPU.
    free_bytes, _ = torch.cuda.mem_get_info()
    if free_bytes < 12 * 2**30:
        batch_size = min(batch_size, 8)
    build, opts = architecture_schedule(method)
    for i in range(0,len(jobs),batch_size):
        chunk=jobs[i:i+batch_size]
        ks = [k for _,k,_ in chunk]
        y, io = cases["y"][ks], cases["iotas"][ks]
        theta0 = np.stack([th for _,_,th in chunk])
        print(f"Starting {method}, {len(chunk)} independent fits, one current per fit",flush=True)
        rows = solver_pinn.train_batch(y, io, theta0, build, ks, progress=progress, **opts)
        for st in ("center", "fno_ttt"):
            jj = [j for j,(kind,_,_) in enumerate(chunk) if kind == st]
            if jj:
                save_fit(method, st, [ks[j] for j in jj], [rows[j] for j in jj],
                         cases, theta0[jj], y[jj])


def run(method, start, batch_size=16):
    if start == "paired":
        return paired(method, batch_size)
    cases = load_cases()
    sub = RESULT / method / start
    ks = [k for k in range(8) if not (sub / f"m{k:03d}.json").exists()]
    if not ks:
        return
    all_starts = load_start(start, cases["index"])
    for i in range(0,len(ks),batch_size):
        chunk = ks[i:i+batch_size]
        run_cases(method,start,cases,chunk,all_starts[chunk])


def run_cases(method,start,cases,ks,theta0):
    y = cases["y"][ks]
    ions = cases["iotas"][ks]
    engine = solver_pinn
    extra = None
    build = "cv"
    opts = {}
    if method.startswith(("standard_", "staged_")):
        build, opts = architecture_schedule(method)
    elif method.startswith("solver_correction"):
        params, extra = solver_seed(cases, theta0, ks)
        wdata = 1.0 if method.endswith("weight1") else 0.01
        opts.update(steps=5000, lr=1e-3, phi_lr_mult=1., lbfgs_iters=300, fit_v=True,
                    w=dict(data=wdata, pde=1., vol=1., bc=1.), freeze=("c","v","ms"), init_params=params)
    else:
        opts.update(PREFIT, **JOINT)
        cache = "data"
        if method == "fine":
            with np.load(DATA / "fine_test.npz") as f:
                np.testing.assert_array_equal(f["index"][:8], cases["index"])
                np.testing.assert_array_equal(f["iotas"][:8], cases["iotas"])
                y = f["y"][ks]
                opts["obs"] = solver_pinn.ObsGrid.fine(f["t"])
            cache = "fine"
        if method == "data_correction":
            engine = data_correction_pinn
            opts["stage2"] = MS2
            opts["first_stage_result"] = lambda rows: save_fit("data_first", start, ks, rows, cases, theta0, y)
        opts["prefit_cache"] = HERE / "cache" / cache
    print(f"Starting {method} from {start}; source IDs {cases['index'][ks].tolist()}; field shape {y.shape}",
          flush=True)
    fit = engine.train_batch(y, ions, theta0, build, ks, progress=progress, **opts)
    save_fit(method, start, ks, fit, cases, theta0, y, extra)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=METHODS, required=True)
    p.add_argument("--start", choices=["center", "fno_ttt", "paired"], default="center")
    p.add_argument("--batch-size",type=int,default=16)
    args = p.parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    # Independently scheduled lanes can reach the same method before its result is published.
    # The OS releases this lock if the worker exits during training; run() resumes completed cases.
    from filelock import FileLock
    lock_directory = HERE / 'locks'
    lock_directory.mkdir(exist_ok=True, parents=True)
    with FileLock(lock_directory / f'{args.method}_{args.start}.lock'):
        run(args.method, args.start, args.batch_size)
