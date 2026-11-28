"""Run the existing DiffIon numerical experiments on the one-current dataset.

From diffion-ms run: python -m one_current.solver_runs.run --workers 12
Completed records are saved atomically after each material or compact start.
Keep the process running while the one-current learning jobs publish their starts.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from functools import lru_cache
import json
import multiprocessing
import os
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXPERIMENT = HERE.parent / "random"
OUT = EXPERIMENT / "solver_runs" / "results"
DATA = EXPERIMENT / "data"
LEARNED = EXPERIMENT / "learned"
NAMES = ("center", "fno", "fno_ttt", "mlp", "mlp_ttt")
COMPACT_TARGETS = ("target_0.01pct", "target_0.1pct", "target_1pct")

# A worker solves one independent optimization at a time. Threaded linear
# algebra in BO and JAX must not oversubscribe the other CPU workers.
for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")


def setup_worker(core_ids):
    # Make the XLA worker pools small on Windows. The inference and GPU
    # training processes remain free to use the other logical processors.
    if os.name == "nt":
        import ctypes
        kernel = ctypes.windll.kernel32
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        core = core_ids[(multiprocessing.current_process()._identity[-1] - 1) % len(core_ids)]
        kernel.SetProcessAffinityMask(ctypes.c_void_p(kernel.GetCurrentProcess()), ctypes.c_size_t(1 << core))


def outfile(task):
    method, start, k = task
    if method == "compact":
        target, optimizer = start.split("/")
        return OUT / "compact" / target / f"{optimizer}_start{k}.json"
    if method == "lbfgs":
        return OUT / "lbfgs" / start / f"m{k:03d}.json"
    return OUT / "gradfree" / start / method / f"m{k:03d}.json"


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, allow_nan=False))
    temp.replace(path)


@lru_cache(maxsize=1)
def test():
    import numpy as np
    with np.load(DATA / "train.npz") as d:
        ids = np.flatnonzero(np.load(DATA / "train_split.npy") == "id_test")[:300]
        out = {name: d[name][ids] for name in ("theta", "window", "y", "iotas", "index")}
    assert out["y"].shape == (300, 1, 13, 20) and out["iotas"].shape == (300, 1)
    return out


def prediction_path(start, pilot=False):
    model = start.split("_")[0]
    stage = "ttt" if start.endswith("_ttt") else "supervised"
    return LEARNED / f"{model}_s0" / (f"{stage}_first8.npz" if pilot else f"{stage}.npz")


@lru_cache
def predictions(path):
    import numpy as np
    with np.load(path) as d:
        ids, th = d["index"], d["theta"]
    assert len(ids) == len(np.unique(ids)) and th.shape == (len(ids), 8)
    assert np.isfinite(th).all() and (th >= 0).all() and (th <= 1).all()
    return dict(zip(ids.tolist(), th))


def start_theta(start, k, index):
    import numpy as np
    if start == "center":
        return np.full(8, .5)
    path = prediction_path(start)
    if not path.exists() and k < 8:
        path = prediction_path(start, True)
    return predictions(path)[int(index)]


def property_scores(z, truth, window):
    # Same 301-point window and Bernstein bounds as analysis.metrics.score.
    # Use numpy here to avoid loading PyTorch alongside JAX in every worker.
    import numpy as np
    u = np.linspace(*window, 301) / 2
    u = np.clip(u, 0., 1.)
    basis = np.stack(((1-u)**3, 3*u*(1-u)**2, 3*u*u*(1-u), u**3), axis=1)
    def curves(coeff):
        d = 0.1*np.exp(np.log(20.)*(basis @ coeff[:4]))
        t = -.25+1.25*(basis @ coeff[4:])
        return d, t
    d, t = curves(z)
    dt, tt = curves(truth)
    return dict(D_relL2_pct=float(100*np.linalg.norm(d-dt)/np.linalg.norm(dt)),
                tplus_rmse=float(np.sqrt(np.mean((t-tt)**2))),
                group_rmse=float(np.sqrt(np.mean((np.log(np.maximum(1-t,1e-3)/d)
                                                -np.log(np.maximum(1-tt,1e-3)/dt))**2))))


@lru_cache(maxsize=1)
def compact_model():
    import jax.numpy as jnp
    import numpy as np
    from solver import transport_1d as model
    protocol = json.loads((DATA / "protocol.json").read_text())
    if protocol["protocol"] != "random":
        raise ValueError("Expected one randomized current per material")
    # The compact benchmark contains a single material, designated ID 0.
    # Match exactly the per-material seeded selection used for the full run.
    selected = int(np.random.default_rng([236, 0]).integers(3))
    current = (.3, .6, .9)[selected]
    # The original jitted compact simulate() captures this constant on first
    # trace. This process uses one experiment and never traces the triplet.
    model.CURRENTS = jnp.array([current])
    observed = model.synthetic_data()
    assert observed.shape == (1,10,20)
    def loss(z):
        return float(model.loss(z, observed))
    loss(np.full(8, .5))
    return model, current, observed, loss


def run_compact(task):
    import numpy as np
    from inverse import baselines
    _, spec, k = task
    target, method = spec.split("/")
    base = json.loads((ROOT / "results" / "numerical baselines" / target / f"{method}_start{k}.json").read_text())
    model, current, observed, loss = compact_model()
    rel = base["target_rel"]
    limit = base["max_wall_s"]
    summary, rows = baselines.run(method, loss, baselines.sobol_starts(5)[k], k, rel*float(np.mean(observed)),
                                  base["budget_evals"], limit, model.theta_from_z)
    # Omit nonfinite wall-time sentinels from standard JSON files.
    summary["max_wall_s"] = float(limit) if np.isfinite(limit) else None
    summary.update(start=k, target_rel=rel, iotas=[float(current)])
    path = outfile(task)
    path.parent.mkdir(parents=True, exist_ok=True)
    csv = path.with_suffix(".csv")
    temp = csv.with_suffix(".csv.tmp")
    np.savetxt(temp, rows, delimiter=",", header=",".join(baselines.CSV_COLUMNS), comments="")
    temp.replace(csv)
    atomic_json(path, summary)


@lru_cache
def warm(method):
    import jax.numpy as jnp
    import numpy as np
    from inverse import lbfgs
    d = test()
    y, io = d["y"][0], d["iotas"][0]
    args = tuple(map(jnp.asarray, (np.full(8,.5), y, np.isfinite(y[0,:,0]), io)))
    fn = lbfgs.loss_and_grad if method == "lbfgs" else lbfgs.loss_only
    fn(*args)[0].block_until_ready()


def run(task):
    method, start, k = task
    if method == "compact":
        run_compact(task)
        return
    import numpy as np
    from inverse import gradfree, lbfgs
    d = test()
    y, io, material = d["y"][k], d["iotas"][k], int(d["index"][k])
    theta0 = start_theta(start, k, material)
    warm("lbfgs" if method == "lbfgs" else "gradfree")
    mean_c = float(np.nanmean(y))
    if method == "lbfgs":
        r = lbfgs.fit(theta0, y, io, max_iter=500, max_wall_s=600)
        r["to_target"] = {str(t): lbfgs.to_target(r["log"], t*mean_c) for t in lbfgs.TARGETS_REL}
    else:
        budget = 150 if method == "BO" else 2000
        r = gradfree.fit_one(method, theta0, y, io, k, budget, 300.)
    r.update(k=k, index=material, start=start, method=method, theta0=theta0.tolist(), iotas=io.tolist(),
             mean_c=mean_c, K=int(np.isfinite(y[0,:,0]).sum()),
             scores=property_scores(r["theta"], d["theta"][k], d["window"][k]),
             start_scores=property_scores(theta0, d["theta"][k], d["window"][k]))
    atomic_json(outfile(task), r)


def task_list():
    tasks = []
    # Publish paired pilot records for every method first as the starts arrive.
    for k in range(300):
        for method in ("lbfgs", "NM", "CMAES", "BO"):
            if method == "BO" and k >= 30:
                continue
            for start in (NAMES if method == "lbfgs" else ("center", "fno_ttt")):
                tasks.append((method, start, k))
    compact = [("compact", f"{target}/{method}", k) for k in range(5)
               for target in COMPACT_TARGETS for method in ("NM","CMAES","BO")]
    # Interleave compact tasks with full fits so the 30-minute BO wall caps do
    # not monopolize initial worker slots.
    for i, task in enumerate(compact):
        tasks.insert(min(len(tasks), (i+1)*9), task)
    return tasks


def prerequisites(task):
    method, start, k = task
    if method == "compact":
        return (DATA / "protocol.json").exists()
    if not (DATA / "train.npz").exists():
        return False
    return start == "center" or prediction_path(start).exists() or (k < 8 and prediction_path(start,True).exists())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--cores", type=str, help="comma-separated distinct logical CPUs reserved for this pool")
    parser.add_argument("--handoff", type=Path, help="JSON runtime control with active_workers and min_material; compact fits are deferred while min_material > 0")
    args = parser.parse_args()
    cores = list(map(int,args.cores.split(","))) if args.cores else [(2*i)%os.cpu_count() for i in range(args.workers)]
    if len(cores)!=args.workers or len(cores)!=len(set(cores)) or any(c<0 or c>=os.cpu_count() for c in cores):
        raise ValueError("Supply one distinct available logical CPU per worker.")
    tasks = [t for t in task_list() if not outfile(t).exists()]
    inflight = {}
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=setup_worker,initargs=(cores,)) as pool:
        while tasks or inflight:
            control = json.loads(args.handoff.read_text()) if args.handoff else {}
            slots = min(args.workers,int(control.get("active_workers",args.workers)))-len(inflight)
            lower = int(control.get("min_material",0))
            # Check each shared model artifact once per dispatch, rather than
            # repeatedly statting the same OneDrive file for hundreds of fits.
            ready = {}
            available = []
            for t in tasks:
                if len(available) >= slots:
                    break
                if lower and (t[0]=="compact" or t[2]<lower):
                    continue
                group = (t[0] == "compact", t[1] if t[0] != "compact" else "", t[2] < 8)
                if group not in ready:
                    ready[group] = prerequisites(t)
                if ready[group]:
                    available.append(t)
            for t in available:
                tasks.remove(t)
                if outfile(t).exists():
                    continue
                inflight[pool.submit(run,t)] = t
            if not inflight:
                time.sleep(3)
                continue
            finished,_ = wait(inflight, timeout=2, return_when=FIRST_COMPLETED)
            for future in finished:
                task = inflight.pop(future)
                future.result()
                done += 1
                print(done, *task, "completed", flush=True)
    print("All solver runs complete.", flush=True)


if __name__ == "__main__":
    main()
