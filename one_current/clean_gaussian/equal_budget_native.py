"""Run the native CNN-initialized PINN on the common eight-material budget cohort."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json

import numpy as np

from cnn_pinn import DOMAINS, HERE, SOURCE, fit_material, pinn


SCHEDULE = dict(field_fit_steps=143, physics_warmup_steps=286, joint_steps=571, lbfgs_steps=0)
STAGES = dict(field_prefit=143, physics_warmup=286, joint=571)
CACHE = HERE / "cache" / "equal_native_adam1000"
OUT = HERE / "equal_budget_results"


def fit_one(domain, count, material_id, device):
    destination = CACHE / f"{domain}_{count}" / f"{material_id:06d}"
    done = destination / "completed.json"
    if done.exists():
        row = json.loads(done.read_text())
    else:
        row = fit_material(domain, count, device, material_id, destination, SCHEDULE)
    assert row["id"] == material_id and row["steps"] == 1000, row
    assert row["phases"] == dict(field_fit=143, physics_warmup=286, joint=571), row
    assert row["status"] in ("accepted","rejected")
    if not done.exists():
        temp = done.with_suffix(".tmp")
        temp.write_text(json.dumps(row, indent=2, allow_nan=False))
        temp.replace(done)
    print(domain, count, material_id, row["status"], round(row["fit_s"], 1), "s", flush=True)
    return row


def native_cohort(domain, count, selected_currents, current_epsilon):
    """The native steady composite requires at least one eligible nonzero-current profile."""
    pinn.N_EXPERIMENTS = count
    eligible = []
    ineligible = []
    for mid, currents in selected_currents.items():
        selected = currents[:count]
        source = next((SOURCE / "data" / domain).glob(f"*/samples/material_{mid:06d}.npz"))
        obs = pinn.load_three_current_observations(source, selected, concentration_field="concentration_nd")
        if np.any(obs.steady_eligible & (np.abs(obs.iota) > current_epsilon)):
            eligible.append(mid)
        else:
            ineligible.append(dict(id=mid, current_indices=selected.tolist(), statuses=list(obs.status),
                                   reason="No steady nonzero-current profile in the selected observations."))
    return eligible, ineligible


def save_group(domain, count, ids, ineligible):
    paths = [CACHE / f"{domain}_{count}" / f"{mid:06d}" / "completed.json" for mid in ids]
    if not all(p.exists() for p in paths):
        return
    rows = [json.loads(p.read_text()) for p in paths]
    assert [r["id"] for r in rows] == ids and all(r["steps"] == 1000 for r in rows)
    assert all(r["phases"] == dict(field_fit=143, physics_warmup=286, joint=571) for r in rows)
    OUT.mkdir(exist_ok=True)
    stem = OUT / f"{domain}__{count}__cnn_pinn__control"
    arrays = {k:np.asarray([r[k] for r in rows]) for k in ("theta","lo","hi","steps","init_s")}
    arrays.update(ids=np.array(ids), gate_status=np.array([r["status"] for r in rows]),
                  elapsed_s=np.array([r["fit_s"] for r in rows]), total_updates=arrays["steps"],
                  pinn_updates=arrays["steps"], ttt_updates=np.zeros(len(rows), dtype=int))
    temp_npz = stem.with_suffix(".tmp.npz")
    np.savez(temp_npz, **arrays)
    temp_npz.replace(stem.with_suffix(".npz"))
    record = dict(budget=1000, stages=STAGES, train_s=0., test_s=sum(r["fit_s"] for r in rows),
                  cnn_init_s=sum(r["init_s"] for r in rows), material_ids=ids, n_test=len(ids), n_train=0,
                  device=", ".join(sorted({r["device"] for r in rows})), warm_start="cnn",
                  current_indices=[r["current_indices"] for r in rows],
                  ineligible_material_ids=[v["id"] for v in ineligible], ineligible=ineligible,
                  gate_status=[r["status"] for r in rows], failed_gates=[r["failed_gates"] for r in rows],
                  checkpoints=[r["checkpoint"] for r in rows], lbfgs_iters=0,
                  description="Native CNN-initialized PINN. 143 field, 286 physics warmup and 571 joint AdamW updates. "
                              "Fixed checkpoint gates; rejected fits use the terminal checkpoint for diagnosis. "
                              "One CPU thread per fit when run on CPU. Fit timings are individual worker wall times.")
    temp_json = stem.with_suffix(".tmp.json")
    temp_json.write_text(json.dumps(record, indent=2, allow_nan=False))
    temp_json.replace(stem.with_suffix(".json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--device", choices=("cpu","cuda"), default="cpu")
    parser.add_argument("--currents", type=int, nargs="+", choices=(1,3), default=[1,3])
    parser.add_argument("--domains", nargs="+", choices=DOMAINS, default=list(DOMAINS))
    parser.add_argument("--materials", type=int, nargs="+")
    args = parser.parse_args()
    with np.load(HERE / "cache" / "index.npz") as idx:
        test = np.flatnonzero(idx["split"] == "test")[:8]
        ids = idx["ids"][test].astype(int).tolist()
        currents = dict(zip(ids, idx["chosen"][test]))
    selected = args.materials or ids
    if any(mid not in ids for mid in selected):
        parser.error("Select materials from the first eight held-out materials")
    config = json.loads((SOURCE / "reference_runs/pinn_clean_accepted/run_config.json").read_text())
    epsilon = config["steady_composite"]["current_epsilon"]
    cohorts = {(d,n):native_cohort(d, n, currents, epsilon) for n in args.currents for d in args.domains}
    jobs = [(domain,count,mid,args.device) for mid in selected
            for (domain,count),(eligible,_) in cohorts.items() if mid in eligible]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fit_one,*job): job[:2] for job in jobs}
        failures = []
        for future in as_completed(futures):
            domain, count = futures[future]
            try:
                future.result()
                save_group(domain, count, *cohorts[domain,count])
            except Exception as exc:
                failures.append(exc)
                print(domain,count,repr(exc),flush=True)
        if failures:
            raise RuntimeError(f"{len(failures)} native fits failed") from failures[0]


if __name__ == "__main__":
    main()
