"""Run the 290 x 151 forward-only inverse benchmark. Resume with the same command."""
import argparse
import json
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from itertools import product

for key in ("MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np
from inverse import baselines
from .fine_df_data import (COHORT, COUNTS, DOMAINS, RESULTS, STARTS, STORE, cohort, get_starts,
                           prepare, sample_path, write_json, write_npz)
from .fine_fv import objective

LIMITS = {"nm": 500, "cmaes": 500, "bo": 35}
METHODS = tuple(LIMITS)
TARGET = 1.e-4
WALL = 300.


def checkpoint(row, domain, count):
    return STORE / "fits" / f"{domain}__{count}" / f"{int(cohort()['ids'][row]):06d}.json"


@lru_cache(maxsize=1)
def initial_values():
    return get_starts()[0]


@lru_cache(maxsize=1)
def load_material(row):
    with np.load(sample_path(int(cohort()["ids"][row]))) as z:
        return {key: z[key] for key in ("chosen", "iota", "mask", "tau", *DOMAINS)}


def observations(row, domain, count):
    z = load_material(row)
    return z["iota"][:count], z["tau"][:count], z["mask"][:count], z[domain][:count]


def init_worker():
    warnings.simplefilter("ignore", UserWarning)
    import cma, skopt
    initial_values()
    p = observations(0, "clean", 1)
    objective(np.full(8, .5), *p)


def optimize(method, mid, start, p):
    failed = 0
    def rmse(z):
        nonlocal failed
        try:
            mse = objective(z, *p)
            if np.isfinite(mse) and mse >= 0:
                return np.sqrt(mse)
        except (OverflowError, ZeroDivisionError):
            pass
        failed += 1
        return 1.e6

    limit = LIMITS[method]
    clock = time.perf_counter()
    rec = baselines.Recorder(rmse, TARGET, limit, WALL, lambda z: z)
    try:
        if method == "cmaes":
            rec(start)
            baselines.cma_es(rec, start, mid, limit-1)
        else:
            baselines.OPTIMIZERS[method.upper()](rec, start, mid, limit)
    except baselines._EndRun:
        pass
    elapsed = time.perf_counter() - clock
    rows = np.asarray(rec.rows)
    np.testing.assert_array_equal(start, rows[0, 5:])
    best = np.argmin(rows[:, 3])
    if rows[0, 3] >= 1.e6 or not np.isfinite(rows[:, 3]).all():
        raise ValueError(f"Invalid initial forward solve for {mid} / {method}")
    return dict(theta=rows[best, 5:].tolist(), theta0=start.tolist(), concentration_rmse=float(rows[best, 3]),
                initial_c_rmse=float(rows[0, 3]), evals=len(rows), elapsed_s=elapsed, solver_s=float(rec.solver_s),
                status=rec.status(), invalid_evals=failed)


def fit_case(task):
    row, domain, count = task
    mid = int(cohort()["ids"][row])
    path = checkpoint(*task)
    p = observations(*task)
    bounds = p[-1][p[2]]
    new = dict(id=mid, current_indices=cohort()["chosen"][row, :count].tolist(),
               lo=float(np.min(bounds)), hi=float(np.max(bounds)), fits={})
    case = json.loads(path.read_text()) if path.exists() else new
    for key in ("id", "current_indices", "lo", "hi"):
        if new[key] != case[key]:
            raise ValueError(f"Checkpoint changed for {path} / {key}")
    added = 0
    for method, start in product(METHODS, STARTS):
        name = f"{method}_{start}"
        if name in case["fits"]:
            continue
        z0 = initial_values()[domain, count, start][row]
        case["fits"][name] = optimize(method, mid, z0, p)
        write_json(path, case)
        added += 1
    return added


def aggregate():
    for domain, count in product(DOMAINS, COUNTS):
        cases = [json.loads(checkpoint(row, domain, count).read_text()) for row in range(COHORT)]
        common = dict(ids=np.array([r["id"] for r in cases]), lo=np.array([r["lo"] for r in cases]),
                      hi=np.array([r["hi"] for r in cases]), current_indices=np.array([r["current_indices"] for r in cases]))
        np.testing.assert_array_equal(common["ids"], cohort()["ids"])
        np.testing.assert_array_equal(common["current_indices"], cohort()["chosen"][:, :count])
        for method, start in product(METHODS, STARTS):
            slug = f"{method}_{start}"
            values = [r["fits"][slug] for r in cases]
            arrays = {key: np.array([v[key] for v in values]) for key in values[0]}
            np.testing.assert_array_equal(arrays["theta0"], initial_values()[domain, count, start])
            if np.any(arrays["concentration_rmse"] > arrays["initial_c_rmse"]) or np.any(arrays["evals"] > LIMITS[method]):
                raise ValueError(f"Invalid solver result in {domain} / {count} / {slug}")
            stem = RESULTS / f"{domain}__{count}__{slug}"
            write_npz(stem.with_suffix(".npz"), **common, **arrays)
            labels, counts = np.unique(arrays["status"], return_counts=True)
            write_json(stem.with_suffix(".json"), dict(n_test=COHORT, n_train=0, train_s=0.,
                       test_s=float(arrays["elapsed_s"].sum()), median_fit_s=float(np.median(arrays["elapsed_s"])),
                       median_evals=float(np.median(arrays["evals"])), budget_evals=LIMITS[method],
                       median_c_rmse=float(np.median(arrays["concentration_rmse"])),
                       statuses={str(k): int(v) for k, v in zip(labels, counts)}))


def run(workers):
    if not (RESULTS / "protocol.json").is_file() or not all(
        sample_path(int(mid)).is_file() for mid in cohort()["ids"]
    ):
        prepare()
    tasks = []
    existing = 0
    for row, domain, count in product(range(COHORT), DOMAINS, COUNTS):
        path = checkpoint(row, domain, count)
        path.parent.mkdir(exist_ok=True, parents=True)
        n = len(json.loads(path.read_text())["fits"]) if path.exists() else 0
        existing += n
        if n < len(METHODS) * len(STARTS):
            tasks.append((row, domain, count))
    print(f"Resuming {existing}/18000 fits; {len(tasks)} pending material/domain/current cases, {workers} workers.", flush=True)
    if tasks:
        init_worker()
        clock = time.perf_counter()
        retry = []
        with ProcessPoolExecutor(max_workers=workers, initializer=init_worker) as pool:
            futures = {pool.submit(fit_case, task): task for task in tasks}
            for i, future in enumerate(as_completed(futures), 1):
                try:
                    existing += future.result()
                except OSError as exc:
                    task = futures[future]
                    retry.append(task)
                    print(f"Will retry interrupted checkpoint for {task}: {exc}", flush=True)
                if i <= 8 or i % 10 == 0 or i == len(tasks):
                    print(f"Completed {existing}/18000 fits, {i}/{len(tasks)} cases; {time.perf_counter()-clock:.0f}s.", flush=True)
        for task in retry:
            fit_case(task)
    aggregate()
    print("All 60 conditions contain 300 verified fits.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=18)
    args = parser.parse_args()
    run(args.workers)
