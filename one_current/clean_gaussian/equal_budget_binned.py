"""Rerun binned PINNs under one end-to-end per-material optimizer update limit."""
import argparse
import json
import os
from time import perf_counter

import numpy as np
import torch

from pinn_benchmark import CACHE, DTYPE, HERE, OUT, engine, fine_inputs, make_corrections, statistics


BUDGET = 1000
DEST = HERE / "equal_budget_results"
METHODS = ("standard_c", "standard_cv", "staged_c_w1", "staged_cv_w1",
           "staged_c_w100", "staged_cv_w100", "data_first", "frozen", "data_correction",
           "solver_correction", "solver_correction_weight1", "fine")


def allocate(method, ttt):
    """Scale original Adam phase lengths using largest-remainder integer allocation."""
    if method.startswith("standard"):
        phases = {"joint": 500}
    elif method.startswith("staged"):
        phases = {"field_prefit": 350, "physics_warmup": 100, "joint": 500}
    elif method == "data_correction":
        phases = {"field_prefit": 850, "joint": 500, "correction": 500}
    elif method.startswith("solver_correction"):
        phases = {"field_prefit": 850, "correction": 500}
    else:
        phases = {"field_prefit": 850, "material_only" if method == "frozen" else "joint": 500}
    available = BUDGET - ttt
    den = sum(phases.values())
    counts = {key: available * length // den for key, length in phases.items()}
    priority = sorted(phases, key=lambda key: available * phases[key] % den, reverse=True)
    for key in priority[:available - sum(counts.values())]:
        counts[key] += 1
    return {"ttt": ttt, **counts}


def prepare(domain, count):
    with np.load(CACHE / "binned.npz") as source:
        rows = np.flatnonzero(source["split"] == "test")[:8]
        ids = source["ids"][rows]
        picks = source["chosen"][rows, :count]
        iota = source["iota"][rows, :count]
        times = source["tau"][rows, :count]
        y = source[domain][rows, :count]
    stem = OUT / f"{domain}__{count}__fno_ttt"
    assert json.loads(stem.with_suffix(".json").read_text())["ttt_steps"] == 150
    with np.load(stem.with_suffix(".npz")) as warm:
        np.testing.assert_array_equal(ids, warm["ids"][:len(ids)])
        starting = warm["theta"][:len(ids)].clip(1e-5, 1-1e-5)
    return ids, picks, iota, times, y, starting


def fit(domain, count, method, start, arm, data, device, force=False):
    ids, picks, iota, times, y, warm = data
    ttt = 150 if start == "fno_ttt" else 0
    stages = allocate(method, ttt)
    assert sum(stages.values()) == BUDGET
    result = DEST / f"{domain}__{count}__pinn_{method}_{start}__{arm}"
    metadata = result.with_suffix(".json")
    binary = result.with_suffix(".npz")
    if not force and metadata.exists() and binary.exists():
        existing = json.loads(metadata.read_text())
        if existing["stages"] == stages and existing["budget"] == BUDGET:
            with np.load(binary) as saved:
                np.testing.assert_array_equal(saved["ids"], ids)
                np.testing.assert_array_equal(saved["current_indices"], picks)
                assert np.all(saved["total_updates"] == BUDGET)
            return
    solver = method.startswith("solver_correction")
    correction = "correction" in method
    build = "c" if method == "standard_c" or method.startswith("staged_c_") else "cv"
    if arm == "slopes" and build != "cv":
        raise ValueError("The observed c/v slope objective requires both field networks")
    theta0 = warm if ttt else np.full((len(ids),8), .5)
    grid = engine.ObsGrid(np.arange(y.shape[2]), y.shape[-1], 2)
    chunk = (2 if count == 1 else 1) if method == "fine" else len(ids)
    defaults = dict(width=32, depth=3, dtype=DTYPE, device=device, obs=grid, n_res=64,
                    n_times=6, n_cells=12, log_every=5000, lbfgs_iters=0, fit_lbfgs_iters=0,
                    spatial_derivative_weight=float(arm == "slopes"), evaluate_spatial_derivatives=build == "cv")
    forward_s = 0.
    if solver:
        with np.load(CACHE / f"{domain}__{count}__seedfield.npz") as source:
            np.testing.assert_array_equal(source["ids"], ids)
            synthetic = source["field"].copy()
            synthetic[~np.isfinite(y)] = np.nan
            forward_s = float(source["elapsed_s"])
    all_rows, pinn_updates = [], []
    started = perf_counter()
    for k in range(0, len(ids), chunk):
        batch = slice(k, k+chunk)
        observed_stats = [statistics(c, io, t) for c, io, t in zip(y[batch], iota[batch], times[batch])]

        def train(th, stats, options):
            rows, params = engine.train_batch(y[batch], iota[batch], th, build, ids[batch],
                return_params=True, observation_stats=stats, **(defaults | options))
            expected = sum(options.get(key,0) for key in ("fit_steps","warmup_steps","steps"))
            if any(row["stopped_by_wall"] or row["steps"] != expected or row["evals"] != expected for row in rows):
                raise RuntimeError(f"Incomplete update budget at {result.name} in batch {k}: "
                                   f"{[(row['steps'],row['evals']) for row in rows]}, expected {expected}")
            return rows, params

        options = dict(fit_steps=stages.get("field_prefit",0), warmup_steps=stages.get("physics_warmup",0),
                       steps=stages.get("joint",stages.get("material_only",0)), fit_v=False)
        if method.startswith("staged"):
            w = float(method.split("_w")[1])
            options.update(lr=1e-4, phi_lr_mult=10., w=dict(pde=w,vol=w,bc=w))
        elif not method.startswith("standard"):
            options.update(lr=1e-4, phi_lr_mult=10., fit_v=True, w=dict(data=.01,pde=1.,vol=1.,bc=1.))
        if method == "frozen":
            options["freeze"] = ("c","v")
        base_stats = observed_stats
        if solver:
            base_stats = [statistics(c,io,t) for c,io,t in zip(synthetic[batch],iota[batch],times[batch])]
            options.update(spatial_derivative_weight=0., evaluate_spatial_derivatives=False)
        rows, params = train(theta0[batch], base_stats, options)
        performed = np.asarray([row["steps"] for row in rows])
        if correction:
            base_theta = np.asarray([row["theta"] for row in rows])
            if solver:
                np.testing.assert_allclose(base_theta, theta0[batch], atol=1e-6)
                base_theta = theta0[batch]
            base_params = make_corrections(params, observed_stats, grid, device)
            options = dict(steps=stages["correction"], fit_v=True, freeze=("c","v","ms"),
                           init_params=base_params, lr=1e-3, phi_lr_mult=1.,
                           w=dict(data=1. if method.endswith("weight1") else .01,pde=1.,vol=1.,bc=1.))
            rows, _ = train(base_theta, observed_stats, options)
            performed += [row["steps"] for row in rows]
        all_rows.extend(rows)
        pinn_updates.extend(performed)
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    wall = perf_counter()-started+forward_s
    pinn_updates = np.asarray(pinn_updates,dtype=int)
    assert np.all(pinn_updates+ttt == BUDGET)
    arrays = dict(ids=ids, current_indices=picks, theta=np.array([r["theta"] for r in all_rows]),
                  lo=np.nanmin(y,axis=(1,2,3)), hi=np.nanmax(y,axis=(1,2,3)), pinn_updates=pinn_updates,
                  ttt_updates=np.full(len(ids),ttt), total_updates=pinn_updates+ttt)
    keys = ["rel_rmse"]
    if build == "cv":
        keys += ["v_rel_rmse","c_slope_rmse","v_slope_rmse","c_slope_rel_rmse","v_slope_rel_rmse"]
    for key in keys:
        arrays[key] = np.array([r[key] for r in all_rows])
    for key, value in arrays.items():
        if not np.isfinite(value).all():
            raise FloatingPointError(f"Nonfinite {key}: {result}")
    md = dict(budget=BUDGET, stages=stages, test_s=wall, train_s=0., forward_s=forward_s, device=device,
              gpu=torch.cuda.get_device_name() if str(device).startswith("cuda") else None, test_batch=chunk,
              source_domain=domain, material_ids=ids.tolist(), current_indices=picks.tolist(),
              initialization=start, arm=arm, n_test=len(ids), solver_prefit=solver,
              spatial_derivative_weight=float(arm=="slopes"), lbfgs_steps=0,
              optimizer="Adam PINN; FNO + TTT starts include 150 per-material AdamW updates",
              allocation="Largest remainder using the original Adam phase proportions; ties in training order.",
              timing="Includes all PINN stages and the measured forward solve when used. FNO + TTT initialization is accounted separately.")
    temp_npz = binary.with_suffix(".tmp.npz")
    np.savez(temp_npz, **arrays)
    temp_npz.replace(binary)
    temp_json = metadata.with_suffix(".tmp.json")
    temp_json.write_text(json.dumps(md,indent=2)+"\n", encoding="utf-8")
    temp_json.replace(metadata)
    print(result.name,f"{wall:.2f}s",stages,flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda", choices=("cpu","cuda"))
    parser.add_argument("--domains", nargs="+", choices=("clean","gaussian_noisy"), default=("clean","gaussian_noisy"))
    parser.add_argument("--currents", nargs="+", type=int, choices=(1,3), default=(1,3))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(0)
    os.environ.setdefault("DIFFION_PINN_GRAPHS", "1")
    DEST.mkdir(exist_ok=True)
    for domain in args.domains:
        for count in args.currents:
            data = prepare(domain, count)
            for method in args.methods:
                inputs = data
                if method == "fine":
                    ids,picks,iota,_,_,warm = data
                    y,times = fine_inputs(ids,picks,domain)
                    inputs = ids,picks,iota,times,y,warm
                for start in (("fno_ttt",) if "correction" in method else ("center","fno_ttt")):
                    arms = ("control",) if method == "standard_c" or method.startswith("staged_c_") else ("control","slopes")
                    for arm in arms:
                        fit(domain,count,method,start,arm,inputs,args.device,args.force)


if __name__ == "__main__":
    main()
