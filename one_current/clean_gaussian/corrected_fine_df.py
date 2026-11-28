"""Derivative-free fits on the supplied corrected 290 × 151 observations. Run from the repo root."""
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import time
import warnings

for variable in ("MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np

from .fine_df_run import LIMITS, TARGET, WALL, optimize
from .fine_fv import objective
from .prepare import CACHE, DATA, HERE


ROOT = HERE / "corrected_noisy_results"
OUT = ROOT / "derivative_free_290"
STORE = CACHE / "corrected_df_290" / "checkpoints"
DOMAIN = "gaussian_noisy"
REVISION = "corrected_grid_native_v1"
STARTS = ("center", "fno")
SLUGS = [f"{method}_{start}" for method in LIMITS for start in STARTS]
STATE = None


def sha(data):
    return hashlib.sha256(data).hexdigest()


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as f:
        tmp = Path(f.name)
        if path.suffix == ".npz":
            np.savez(f, **value)
        else:
            f.write((json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8"))
    for attempt in range(10):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(.01 * (2**attempt))


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sources(folder):
    matches = sorted(folder.glob("*/samples/material_*.npz"))
    indexed = {int(f.stem[9:]): f for f in matches}
    if len(matches) != len(indexed):
        raise ValueError(f"Duplicate material IDs in {folder}")
    return indexed


def context(count):
    index_file = ROOT / "material_index.npz"
    with np.load(index_file) as f:
        eligible, split, chosen, iota, theta = [f[k] for k in ("ids", "split", "chosen", "iota", "theta")]
    test = np.flatnonzero(split == "test")
    if len(test) != 389 or len(set(eligible)) != len(eligible):
        raise ValueError("Unexpected material index or held-out split")
    take = test[:300]
    seed = ROOT / "grid290" / "seed0" / f"{DOMAIN}__{count}__fno"
    meta = read(seed.with_suffix(".json"))
    prep = read(ROOT / "grid290" / "preparation.json")
    required = dict(dataset_revision=REVISION, source_grid=[290,151], seed=0, architecture="fno", currents=count,
                    domain=DOMAIN, n_test=389, n_train=1402, n_val=156, epoch_metrics="same_fit",
                    training_precision="float32_tf32_off",cpu_threads=1,
                    source_folder=prep["source_folder"], input_sha256=prep["input_sha256"], source_sha256=prep["source_sha256"])
    if any(meta.get(k) != v for k,v in required.items()):
        raise ValueError(f"Wrong source or initializer in {seed}")
    raw_seed = seed.with_suffix(".npz").read_bytes()
    with np.load(io.BytesIO(raw_seed)) as f:
        np.testing.assert_array_equal(f["ids"], eligible[test])
        estimates = f["theta"].astype(np.float64)
        bounds = np.column_stack((f["lo"][:300], f["hi"][:300]))
    if estimates.shape != (389,8) or not np.isfinite(estimates).all() or np.any((estimates < 0) | (estimates > 1)):
        raise ValueError("Invalid seed-0 FNO material coefficients")
    index_digest = sha(index_file.read_bytes())
    if prep["index_sha256"] != index_digest or prep["source_folder"] != "Corrected_noisy_data/290_151_noised":
        raise ValueError("Grid source or index does not match corrected training")
    files = sources(DATA / prep["source_folder"])
    with ThreadPoolExecutor(max_workers=4) as pool:
        hashes = list(pool.map(lambda mid: hashlib.sha256(files[int(mid)].read_bytes()).digest(), eligible))
    if sha(b"".join(hashes)) != meta["source_sha256"]:
        raise ValueError("The actual corrected input files differ from those used to train the FNO")
    clean = sources(DATA / "clean")
    if clean and any(int(mid) not in clean for mid in eligible[take]):
        raise ValueError("Incomplete native clean evidence for the paired noise measurement")
    protocol = dict(version="corrected_df_290_v1", dataset_revision=REVISION, source_grid=[290,151], domain=DOMAIN,
                    source_folder=prep["source_folder"], source_sha256=meta["source_sha256"], input_sha256=meta["input_sha256"],
                    index_sha256=index_digest, initializer=str(seed.with_suffix(".npz").relative_to(HERE).as_posix()),
                    initializer_sha256=sha(raw_seed), initializer_seed=0, initializer_grid=[290,151],
                    initializer_clip=[1e-6,1-1e-6], n_test=300, currents=count, starts=list(STARTS), solver_cells=580,
                    budget_evals=LIMITS, target_concentration_rmse=TARGET, max_wall_s=WALL,
                    timing="Per-material elapsed wall time includes the initial objective and optimizer overhead; excludes loading, JIT and FNO inference.",
                    evaluation_policy="Initial evaluation consumes one call and is eligible as the best fit. BO and CMA randomization use the benchmark's material-ID seed.",
                    observed_range="Minimum and maximum across all valid supplied concentration bins and times for the selected currents.",
                    paired_clean_noise=bool(clean))
    return dict(ids=eligible[take], chosen=chosen[take,:count], iota=iota[take,:count], truth=theta[take],
                files=[files[int(mid)] for mid in eligible[take]], clean=[clean[int(mid)] for mid in eligible[take]] if clean else [],
                hashes=[hashes[i].hex() for i in take], fno=np.clip(estimates[:300],1e-6,1-1e-6), bounds=bounds,
                signature=sha(json.dumps(protocol,sort_keys=True).encode()), protocol=protocol)


def observations(state, row):
    mid = int(state["ids"][row])
    js = state["chosen"][row]
    raw = state["files"][row].read_bytes()
    if sha(raw) != state["hashes"][row]:
        raise ValueError(f"Corrected file changed during run for material {mid}")
    with np.load(io.BytesIO(raw)) as f:
        np.testing.assert_array_equal(f["noise_seed"], [20261124,290,mid])
        np.testing.assert_array_equal(f["material_sample_id"], mid)
        np.testing.assert_array_equal(np.r_[f["D_coefficients_01"],f["t_plus_coefficients_01"]], state["truth"][row])
        np.testing.assert_array_equal(f["iota"][js], state["iota"][row])
        full = f["concentration_nd"]
        if full.shape != (8,151,290):
            raise ValueError(f"Unexpected corrected grid for {mid}: {full.shape}")
        c = full[js].astype(np.float64)
        mask = f["valid_time_mask"][js]
        times = np.tile(f["tau"], (len(js),1))
        density = f["current_density_a_m2"][js]
    if mask.shape != (len(js),151) or not np.isfinite(c[mask]).all() or not np.isnan(c[~mask]).all():
        raise ValueError(f"Inconsistent finite support for {mid}")
    lo, hi = float(c[mask].min()),float(c[mask].max())
    np.testing.assert_array_equal([lo,hi],state["bounds"][row])
    p = (np.ascontiguousarray(state["iota"][row]),times,mask,c)
    return p, dict(lo=lo,hi=hi,current_indices=js.tolist(),current_density_a_m2=density.tolist(),
                   iota=p[0].tolist(),valid_frames=mask.sum(-1).tolist())


def init_worker(state):
    global STATE
    STATE = state
    warnings.simplefilter("ignore", UserWarning)
    import cma, skopt  # Load optimizer dependencies outside fitted wall-time measurements.
    p,_ = observations(state, 0)
    objective(np.full(8,.5),*p)


def checkpoint(state,row):
    return STORE / str(state["protocol"]["currents"]) / f"{state['ids'][row]:06d}.json"


def expected(state,row):
    return dict(id=int(state["ids"][row]), signature=state["signature"], source_file_sha256=state["hashes"][row],
                initializer_sha256=state["protocol"]["initializer_sha256"],
                fno_start_sha256=sha(np.asarray(state["fno"][row],dtype="<f8").tobytes()))


def matches(case, check):
    if any(case.get(k) != v for k,v in check.items()):
        raise ValueError(f"Stale corrected checkpoint for material {check['id']}")


def fit_case(row):
    state = STATE
    path = checkpoint(state,row)
    check = expected(state,row)
    p, info = observations(state,row)
    if path.exists():
        case = read(path)
        matches(case,{**check,**info})
    else:
        truth_rmse = float(np.sqrt(objective(state["truth"][row],*p)))
        case = dict(**check,**info,true_c_rmse=truth_rmse,fits={})
        if state["clean"]:
            with np.load(state["clean"][row]) as clean:
                js = state["chosen"][row]
                np.testing.assert_array_equal(clean["iota"][js],p[0])
                np.testing.assert_array_equal(clean["valid_time_mask"][js],p[2])
                np.testing.assert_array_equal(clean["tau"],p[1][0])
                native = clean["concentration_nd"][js][p[2]]
                baseline = native.reshape(-1,290,4).mean(-1,dtype=np.float64)
            noise = p[3][p[2]] - baseline
            case.update(noise_ss=float(np.sum(noise**2)),noise_points=int(noise.size))
    added = 0
    for method in LIMITS:
        for start in STARTS:
            slug = f"{method}_{start}"
            if slug not in case["fits"]:
                initial = np.full(8,.5) if start == "center" else state["fno"][row]
                case["fits"][slug] = optimize(method,check["id"],initial,p)
                atomic(path,case)
                added += 1
    return added


def describe(values):
    v=np.asarray(values)
    if not np.isfinite(v).all():
        raise ValueError("Nonfinite forward reference")
    return dict(median=float(np.median(v)),p95=float(np.quantile(v,.95)),p99=float(np.quantile(v,.99)),
                min=float(v.min()),max=float(v.max()),at_or_below_target_1e_4=int((v<=TARGET).sum()))


def aggregate(state):
    protocol = state["protocol"]
    count = protocol["currents"]
    cases = [read(checkpoint(state,row)) for row in range(len(state["ids"]))]
    for row,case in enumerate(cases):
        matches(case,expected(state,row))
        if set(case["fits"]) != set(SLUGS):
            raise ValueError(f"Incomplete fits for {case['id']}")
    common = dict(ids=state["ids"],theta_true=state["truth"])
    for k in ("lo","hi","current_indices","iota","current_density_a_m2","valid_frames","true_c_rmse"):
        common[k] = np.array([r[k] for r in cases])
    np.testing.assert_array_equal(common["current_indices"],state["chosen"])
    np.testing.assert_array_equal(np.column_stack((common["lo"],common["hi"])),state["bounds"])
    for slug in SLUGS:
        method, start = slug.split("_")
        values = [r["fits"][slug] for r in cases]
        arr = {k:np.array([v[k] for v in values]) for k in values[0]}
        init = np.full((300,8),.5) if start == "center" else state["fno"]
        np.testing.assert_array_equal(arr["theta0"],init)
        if (not np.isfinite(arr["theta"]).all() or not np.isfinite(arr["concentration_rmse"]).all()
                or np.any(arr["concentration_rmse"] > arr["initial_c_rmse"]) or np.any(arr["evals"] < 1)
                or np.any(arr["evals"] > LIMITS[method]) or np.any(arr["elapsed_s"]<=0)):
            raise ValueError(f"Inconsistent optimization record for {slug}")
        stem = OUT / f"{DOMAIN}__{count}__{slug}"
        atomic(stem.with_suffix(".npz"),{**common,**arr})
        statuses,counts=np.unique(arr["status"],return_counts=True)
        metadata = dict(protocol)
        metadata.update(method=method,start=start,n_train=0,train_s=0.,budget_evals=LIMITS[method],
                        test_s=float(arr["elapsed_s"].sum()),median_fit_s=float(np.median(arr["elapsed_s"])),
                        median_evals=float(np.median(arr["evals"])),median_c_rmse=float(np.median(arr["concentration_rmse"])),
                        statuses=dict(zip(statuses.tolist(),counts.tolist())))
        atomic(stem.with_suffix(".json"),metadata)
    noise=[]
    if protocol["paired_clean_noise"]:
        ss=np.array([r["noise_ss"] for r in cases]); n=np.array([r["noise_points"] for r in cases])
        noise=[dict(currents=count,pooled_rmse=float(np.sqrt(ss.sum()/n.sum())),**describe(np.sqrt(ss/n)))]
    reference=dict(n_materials=300,units="Dimensionless concentration",dataset_revision=REVISION,
                   description="RMSE at the generating coefficients using 580 solver cells and all valid corrected 290-bin observations.",
                   cases=[dict(domain=DOMAIN,currents=count,**describe(common["true_c_rmse"]))],paired_observation_noise=noise)
    atomic(OUT/f"forward_validation_{count}.json",reference)
    paths=[OUT/f"forward_validation_{k}.json" for k in (1,3)]
    if all(f.is_file() for f in paths):
        parts=[read(f) for f in paths]
        combined={**reference,**{key:[r for part in parts for r in part[key]] for key in ("cases","paired_observation_noise")}}
        atomic(OUT/"forward_validation.json",combined)


def run(count,workers,check_only=False):
    state=context(count)
    observations(state,0)
    if check_only:
        print(f"Verified corrected 290×151 inputs, index, seed-0 FNO, and first material for {count} current(s).",flush=True)
        return
    atomic(OUT/f"protocol_{count}.json",state["protocol"])
    pending=[]; done=0
    for row in range(len(state["ids"])):
        path=checkpoint(state,row)
        completed={}
        if path.exists():
            case=read(path)
            matches(case,expected(state,row))
            completed=case["fits"]
            if not set(completed)<=set(SLUGS):
                raise ValueError(f"Unknown fit in {path}")
        done+=len(completed)
        if len(completed)<len(SLUGS):
            pending.append(row)
    print(f"{count} current(s): {done}/1800 fits complete; {len(pending)} cases pending; {workers} CPU workers.",flush=True)
    if pending:
        init_worker(state)
        with ProcessPoolExecutor(max_workers=workers,initializer=init_worker,initargs=(state,)) as pool:
            futures=[pool.submit(fit_case,row) for row in pending]
            for i,future in enumerate(as_completed(futures),1):
                done+=future.result()
                if i<=3 or i%5==0 or i==len(futures):
                    print(f"{count} current(s): {done}/1800 fits; {i}/{len(futures)} cases returned.",flush=True)
    aggregate(state)
    print(f"Published all six 300-material conditions for {count} current(s).",flush=True)


if __name__ == "__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--currents",type=int,choices=(1,3),required=True)
    cli.add_argument("--workers",type=int,default=14)
    cli.add_argument("--check-only",action="store_true",help="Verify source and initializer without fitting or writing results.")
    args=cli.parse_args()
    if args.workers<1:
        cli.error("--workers must be positive")
    run(args.currents,args.workers,args.check_only)
