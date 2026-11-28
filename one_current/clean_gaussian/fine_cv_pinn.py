"""Run the full-observation CV PINN study on paired held-out materials."""
import argparse
import hashlib
import json
import os
import shutil
import time
from functools import lru_cache
from itertools import product

import numpy as np
import torch

from .fine_df_data import (ARCHIVE, COUNTS, DOMAINS, HERE, cohort, replace_when_readers_close,
                           sample_path, write_json, write_npz)
from .pinn_benchmark import DTYPE, engine, make_corrections, statistics
from .prepare import DATA

RESULTS = HERE / "fine_cv_results"
FIELDS = RESULTS / "checkpoints"
BASE_WIDTHS = (32, 32, 32)
DOUBLE_WIDTHS = (44, 47, 46)
RESULTS_BY_WIDTH = {BASE_WIDTHS: RESULTS, DOUBLE_WIDTHS: HERE / "fine_cv_double_results"}
# Each tuple specifies field-only prefit, joint base fit, and one correction branch.
STAGES_BY_UPDATES = {400: (189, 111, 100), 500: (236, 139, 125)}
STARTS = ("center", "fno_ttt")
METHODS = ("standard", "correction_w1", "correction_w100")
IDS = (2, 3, 11, 21, 22, 32, 37, 45)


def tree_map(tree, transform):
    return engine._rebuild(tree, iter(transform(t) for t in engine._leaves(tree)))


def digest(params):
    h = hashlib.sha256()
    for t in engine._leaves(params):
        a = t.detach().contiguous().cpu().numpy()
        h.update(str((a.dtype, a.shape)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


@lru_cache(maxsize=2)
def material_ids(materials):
    ids = tuple(map(int, cohort()["ids"][:materials]))
    if len(ids) != materials or ids[:8] != IDS:
        raise ValueError("Cohort differs from the paired held-out material ordering")
    return ids


@lru_cache(maxsize=1)
def material_rows():
    return {int(mid): row for row, mid in enumerate(cohort()["ids"])}


@lru_cache(maxsize=8)
def starts(domain, count, materials=8):
    archive = ARCHIVE / f"{domain}__{count}__fno_ttt.npz"
    with np.load(archive) as z:
        by_id = dict(zip(z["ids"], z["theta"]))
    return np.clip(np.array([by_id[mid] for mid in material_ids(materials)], dtype=np.float64), 1.e-5, 1.-1.e-5)


@lru_cache(maxsize=1)
def observation(mid, domain, count):
    row = material_rows()[mid]
    chosen = cohort()["chosen"][row, :count]
    if sample_path(mid).is_file():
        with np.load(sample_path(mid)) as z:
            assert z["id"].item() == mid
            np.testing.assert_array_equal(z["chosen"][:count], chosen)
            mask, y = z["mask"][:count], z[domain][:count]
            ions, times = z["iota"][:count], z["tau"][:count]
    else:
        # A new checkout can read the selected materials directly from source.
        file, = (DATA / domain).glob(f"*/samples/material_{mid:06d}.npz")
        with np.load(file) as z:
            assert z["material_sample_id"].item() == mid
            if domain == "gaussian_noisy":
                assert "gaussian" in str(z["noise_model"]).lower() and z["noise_sigma_ref_mM"].item() == 2.2
            mask = z["valid_time_mask"][chosen]
            raw = z["concentration_nd"][chosen]
            y = np.full((count,151,290), np.nan)
            y[mask] = raw[mask].reshape(-1,290,4).mean(axis=-1, dtype=np.float64)
            ions, times = z["iota"][chosen], np.tile(z["tau"], (count,1))
    np.testing.assert_array_equal(ions, cohort()["iota"][row, :count])
    np.testing.assert_array_equal(np.isfinite(y[..., 0]), mask)
    # Empty trailing frames contribute zero to both observation losses.
    nt = int(mask.sum(axis=1).max())
    y, times = y[:, :nt], times[:, :nt]
    stats = statistics(y, ions, times)
    valid = stats["mask"]
    if not np.array_equal(stats["vmask"], np.broadcast_to(valid[..., None], y.shape)):
        raise ValueError(f"Nonfinite reconstructed velocity for material {mid}")
    return y, ions, chosen, stats


def result_directory(hidden_widths, updates, materials=8):
    folder = RESULTS_BY_WIDTH[hidden_widths]
    suffix = (f"_{updates}" if updates != 400 else "") + (f"_{materials}" if materials != 8 else "")
    return folder.with_name(folder.name.replace("_results", f"{suffix}_results"))


def field_file(mid, domain, count, start, hidden_widths=BASE_WIDTHS, updates=400, materials=8):
    return result_directory(hidden_widths, updates, materials) / "checkpoints" / f"{domain}__{count}__{start}__{mid:06d}.pt"


def layer_counts(hidden_widths):
    sizes = (3, *hidden_widths, 1)
    return [(a + 1) * b for a, b in zip(sizes, sizes[1:])]


def verify_networks(params, hidden_widths):
    sizes = (3, *hidden_widths, 1)
    fields = ("c", "v", "c1", "v1") if "c1" in params else ("c", "v")
    for field in fields:
        layers = params[field]
        assert len(layers) == len(sizes) - 1
        for (weight, bias), a, b in zip(layers, sizes, sizes[1:]):
            assert tuple(weight.shape) == (1, a, b) and tuple(bias.shape) == (1, b)


def save_pack(path, data):
    tmp = path.with_suffix(".tmp.pt")
    torch.save(data, tmp)
    replace_when_readers_close(tmp, path)


def fit(mid, domain, count, start, hidden_widths=BASE_WIDTHS, updates=400, materials=8, graph_cache=None):
    prefit_steps, joint_steps, correction_steps = STAGES_BY_UPDATES[updates]
    split = prefit_steps + joint_steps
    path = field_file(mid, domain, count, start, hidden_widths, updates, materials)
    pack = torch.load(path, map_location="cpu", weights_only=True) if path.exists() else {}
    if pack and tuple(pack.get("hidden_widths", BASE_WIDTHS)) != hidden_widths:
        raise ValueError(f"Incompatible network architecture in {path}")
    if pack and pack.get("total_updates", 400) != updates:
        raise ValueError(f"Incompatible update budget in {path}")
    if all(method in pack for method in METHODS):
        return False
    y, ions, chosen, stats = observation(mid, domain, count)
    z0 = np.full(8, .5) if start == "center" else starts(domain, count, materials)[material_rows()[mid]]
    grid = engine.ObsGrid(np.arange(y.shape[1]), 290, 2)
    width = 32 if hidden_widths == BASE_WIDTHS else hidden_widths
    options = dict(width=width, depth=3, dtype=DTYPE, device="cuda", obs=grid, n_res=64,
                   n_times=6, n_cells=12, log_every=5000, fit_v=True, spatial_derivative_weight=0.,
                   observation_stats=[stats], lbfgs_iters=0, fit_lbfgs_iters=0)
    pack.update(id=mid, hidden_widths=hidden_widths, total_updates=updates, current_indices=chosen.tolist(), theta0=z0.tolist(),
                lo=float(np.nanmin(y)), hi=float(np.nanmax(y)), n_valid=stats["mask"].sum(1).tolist(),
                vscale=float(stats["vscale"]), mean_c=float(stats["mean_c"]))

    def train(theta, step_ids, kwargs):
        states = {}
        graph_stats = graph_cache.stats() if graph_cache is not None else {}
        started = time.perf_counter()
        def capture(step, params, info):
            verify_networks(params, hidden_widths)
            states[step] = dict(params=params, theta=info["theta"][0].tolist(), c=float(info["c_rmse"][0]),
                                v=float(info["v_rmse"][0]), elapsed=time.perf_counter()-started)
        rows = engine.train_batch(y[None], ions[None], np.asarray(theta)[None], "cv", [mid],
                                  snapshot_steps=step_ids, snapshot_callback=capture, graph_cache=graph_cache,
                                  **(options | kwargs))
        torch.cuda.synchronize()
        wall = time.perf_counter()-started
        if rows[0]["stopped_by_wall"] or rows[0]["steps"] != max(step_ids) or rows[0]["evals"] != max(step_ids):
            raise RuntimeError(f"Incomplete schedule for material {mid}: {rows[0]['steps']}")
        for s in states.values():
            if not np.isfinite([s["c"], s["v"], *s["theta"]]).all():
                raise ValueError(f"Nonfinite fitted checkpoint for material {mid}")
        execution = {"execution": "separate_graph"}
        if graph_cache is not None:
            execution = dict(execution="reused_graph", graph_stats={key:value-graph_stats[key]
                             for key,value in graph_cache.stats().items()})
        return states, wall, execution

    if "standard" not in pack:
        shots, wall, mode = train(z0, (0, split, updates), dict(steps=updates, lr=1.e-3, phi_lr_mult=.1,
                            w=dict(data=1., vdata=1., pde=100., vol=100., bc=100.)))
        pack["standard"] = dict(shots=shots, stage1_s=shots[split]["elapsed"], stage2_s=wall-shots[split]["elapsed"], **mode)
        save_pack(path, pack)
        print(f"{domain} {count} {start} {mid} standard: {wall:.2f}s", flush=True)
    if "base" not in pack:
        shots, wall, mode = train(z0, (0, prefit_steps, split), dict(fit_steps=prefit_steps, fit_lr=1.e-3, steps=joint_steps, lr=1.e-4,
                            phi_lr_mult=10., w=dict(data=.01, vdata=1., pde=1., vol=1., bc=1.)))
        np.testing.assert_array_equal(shots[0]["theta"], shots[prefit_steps]["theta"])
        base = tree_map(shots[split]["params"], lambda t: t.to("cuda"))
        begin = time.perf_counter()
        correction = make_corrections(base, [stats], grid, "cuda", hidden_widths=hidden_widths)[0]
        torch.cuda.synchronize()
        setup = time.perf_counter()-begin
        pack["base"] = dict(shots=shots, stage1_s=wall, correction=correction, setup_s=setup, initial_hash=digest(correction), **mode)
        save_pack(path, pack)
        del base
        print(f"{domain} {count} {start} {mid} shared {split}: {wall:.2f}s", flush=True)
    base = pack["base"]
    for method, wd in (("correction_w1", 1.), ("correction_w100", 100.)):
        if method in pack:
            continue
        # train_batch stacks and copies these tensors. Both arms restart the sampler identically.
        init = tree_map(base["correction"], lambda t: t.clone())
        assert digest(init) == base["initial_hash"]
        shots, wall, mode = train(base["shots"][split]["theta"], (0, correction_steps), dict(steps=correction_steps, lr=1.e-3, phi_lr_mult=1.,
                            rng_seed=0, init_params=[init], freeze=("c","v","ms"),
                            w=dict(data=wd, vdata=1., pde=1., vol=1., bc=1.)))
        for key in ("c", "v", "ms"):
            assert digest(shots[0]["params"][key]) == digest(shots[correction_steps]["params"][key])
        assert abs(shots[0]["c"]-base["shots"][split]["c"]) < 1.e-6
        assert abs(shots[0]["v"]-base["shots"][split]["v"]) < 1.e-5 * (1.+base["shots"][split]["v"])
        pack[method] = dict(shots=shots, final_s=wall, stage2_s=wall + base["setup_s"],
                            initial_hash=digest(shots[0]["params"]), **mode)
        if method == "correction_w100":
            assert pack[method]["initial_hash"] == pack["correction_w1"]["initial_hash"]
        save_pack(path, pack)
        print(f"{domain} {count} {start} {mid} {method}: {wall:.2f}s", flush=True)
    return True


def aggregate(domain, count, start, hidden_widths=BASE_WIDTHS, updates=400, materials=8):
    prefit_steps, joint_steps, correction_steps = STAGES_BY_UPDATES[updates]
    split = prefit_steps + joint_steps
    packs = []
    ids = material_ids(materials)
    destination = result_directory(hidden_widths, updates, materials)
    manifest = destination / "reuse_manifest.json"
    reused = json.loads(manifest.read_text())["copied_sha256"] if materials > 8 and manifest.is_file() else {}
    for row, mid in enumerate(ids):
        path = field_file(mid, domain, count, start, hidden_widths, updates, materials)
        if not path.exists():
            return
        p = torch.load(path, weights_only=True, map_location="cpu")
        if any(m not in p for m in METHODS):
            return
        for method in METHODS:
            endpoint = updates if method == "standard" else correction_steps
            verify_networks(p[method]["shots"][endpoint]["params"], hidden_widths)
        expected_start = np.full(8, .5) if start == "center" else starts(domain, count, materials)[row]
        if p["id"] != mid:
            raise ValueError(f"Incorrect material in {path}")
        np.testing.assert_array_equal(p["current_indices"], cohort()["chosen"][row, :count])
        np.testing.assert_array_equal(p["theta0"], expected_start)
        packs.append(p)
    for method in METHODS:
        rows = []
        for p in packs:
            st = p["standard"] if method=="standard" else p["base"]
            end = p[method]["shots"][updates if method=="standard" else correction_steps]
            zero, before = st["shots"][0], st["shots"][split]
            s1, s2 = st["stage1_s"], p[method]["stage2_s"]
            rows.append(dict(ids=p["id"], current_indices=p["current_indices"], valid_frames=p["n_valid"],
                        theta0=p["theta0"], theta=end["theta"], lo=p["lo"], hi=p["hi"],
                        c_rmse0=zero["c"], v_rmse0=zero["v"],
                        c_rmse=end["c"], v_rmse=end["v"], c_rel_rmse=end["c"]/p["mean_c"], v_rel_rmse=end["v"]/p["vscale"],
                        stage1_s=s1,stage2_s=s2,elapsed_s=s1+s2, total_updates=updates,
                        correction_setup_s=0. if method=="standard" else p["base"]["setup_s"],
                        correction_initial_hash="" if method=="standard" else p[method]["initial_hash"],
                        **{f"theta{split}": before["theta"], f"c_rmse{split}": before["c"],
                           f"v_rmse{split}": before["v"], f"updates{split}": split,
                           f"base{split}_hash": digest(before["params"])}))
        arr={k:np.asarray([row[k] for row in rows]) for k in rows[0]}
        np.testing.assert_array_equal(arr["ids"], ids)
        if materials > 8:
            arr["reused_from_eight"] = np.array([field_file(mid, domain, count, start, hidden_widths, updates, materials).name
                                                in reused for mid in ids])
            arr["base_execution"] = np.array([p["standard" if method == "standard" else "base"].get("execution","separate_graph")
                                             for p in packs])
            arr["final_execution"] = np.array([p[method].get("execution","separate_graph") for p in packs])
        stem = destination / f"{domain}__{count}__{method}_{start}"
        write_npz(stem.with_suffix(".npz"), **arr)
        stages = dict(joint=updates) if method=="standard" else dict(field_prefit=prefit_steps, joint=joint_steps, correction=correction_steps)
        write_json(stem.with_suffix(".json"), dict(n_test=materials, gpu=torch.cuda.get_device_name(), dtype="float32",
                   batch_size=1, method=method,start=start, stages=stages,total_updates=updates, archived_ttt_updates=150 if start=="fno_ttt" else 0,
                   stage1_s=float(arr["stage1_s"].sum()),stage2_s=float(arr["stage2_s"].sum()),test_s=float(arr["elapsed_s"].sum()),
                   correction_setup_s=float(arr["correction_setup_s"].sum()), checkpoints="checkpoints", data_domain=domain, currents=count,
                   field_widths=[3, *hidden_widths, 1], per_field_parameters=sum(layer_counts(hidden_widths)),
                   material_logits=8, trainable_parameters=2*sum(layer_counts(hidden_widths))+8,
                   total_nn_parameters=(2 if method=="standard" else 4)*sum(layer_counts(hidden_widths))))


def reuse_eight(hidden_widths, updates, materials):
    """Copy completed pilot packs only after checking their protocol, inputs and saved outcomes."""
    source = result_directory(hidden_widths, updates)
    target = result_directory(hidden_widths, updates, materials)
    copied = {}
    protocol_file = source / "protocol.json"
    if protocol_file.exists():
        protocol = json.loads(protocol_file.read_text())
        prefit, joint, correction = STAGES_BY_UPDATES[updates]
        split = prefit + joint
        assert protocol["field_widths"] == [3, *hidden_widths, 1]
        assert protocol["material_ids"] == list(IDS) and protocol["checkpoint_steps"] == [0, split, updates]
        assert protocol["materials_per_batch"] == 1 and protocol["spatial_bins"] == 290 and protocol["gauss_points_per_bin"] == 2
        for domain, count, start in product(DOMAINS, COUNTS, STARTS):
            archives = {}
            for method in METHODS:
                output = source / f"{domain}__{count}__{method}_{start}.npz"
                if output.is_file():
                    with np.load(output) as z:
                        np.testing.assert_array_equal(z["ids"], IDS)
                        archives[method] = dict(z)
            if len(archives) < len(METHODS):
                continue
            for row, mid in enumerate(IDS):
                path = field_file(mid, domain, count, start, hidden_widths, updates)
                if not path.is_file():
                    continue
                pack = torch.load(path, weights_only=True, map_location="cpu")
                assert pack["id"] == mid and tuple(pack.get("hidden_widths", BASE_WIDTHS)) == hidden_widths
                assert pack.get("total_updates", 400) == updates and pack["correction_w1"]["initial_hash"] == pack["correction_w100"]["initial_hash"]
                y, _, chosen, stats = observation(mid, domain, count)
                np.testing.assert_array_equal(pack["current_indices"], chosen)
                np.testing.assert_array_equal(pack["n_valid"], stats["mask"].sum(1))
                np.testing.assert_array_equal([pack["lo"], pack["hi"], pack["vscale"], pack["mean_c"]],
                                              [np.nanmin(y), np.nanmax(y), stats["vscale"], stats["mean_c"]])
                expected = np.full(8, .5) if start=="center" else starts(domain,count,materials)[row]
                np.testing.assert_array_equal(pack["theta0"], expected)
                for method, summary in archives.items():
                    base = pack["standard"] if method == "standard" else pack["base"]
                    endpoint = updates if method == "standard" else correction
                    final = pack[method]["shots"][endpoint]
                    assert base["shots"][split]["params"] and summary[f"base{split}_hash"][row] == digest(base["shots"][split]["params"])
                    verify_networks(final["params"], hidden_widths)
                    np.testing.assert_array_equal(summary["theta"][row], final["theta"])
                    np.testing.assert_array_equal([summary["c_rmse"][row], summary["v_rmse"][row]], [final["c"],final["v"]])
                dest = target / "checkpoints" / path.name
                checksum = hashlib.sha256(path.read_bytes()).hexdigest()
                if not dest.exists():
                    temp = dest.with_suffix(".tmp.pt")
                    shutil.copyfile(path, temp)
                    replace_when_readers_close(temp, dest)
                if hashlib.sha256(dest.read_bytes()).hexdigest() == checksum:
                    copied[dest.name] = checksum
    write_json(target / "reuse_manifest.json", dict(source=source.name, copied_sha256=copied))
    print(f"Verified {len(copied)} reused eight-material checkpoints.", flush=True)
    return copied


def run(domains, counts, starts_requested, hidden_widths=BASE_WIDTHS, updates=400, materials=8, reuse_graphs=False):
    os.environ.setdefault("DIFFION_PINN_GRAPHS", "1")
    torch.manual_seed(0)
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError("The benchmark requires a CUDA GPU")
    ids = material_ids(materials)
    prefit_steps, joint_steps, correction_steps = STAGES_BY_UPDATES[updates]
    split = prefit_steps + joint_steps
    assert split + correction_steps == updates
    destination = result_directory(hidden_widths, updates, materials)
    (destination / "checkpoints").mkdir(exist_ok=True, parents=True)
    counts_per_layer = layer_counts(hidden_widths)
    if hidden_widths == DOUBLE_WIDTHS and sum(counts_per_layer) != 2 * sum(layer_counts(BASE_WIDTHS)):
        raise ValueError("The doubled study requires exactly twice the neural parameters per field")
    proto = dict(material_ids=ids, spatial_bins=290, max_time_frames=151, gauss_points_per_bin=2,
            residual_points_per_current=64, materials_per_batch=1, dtype="float32",gpu=torch.cuda.get_device_name(),
            torch=torch.__version__, cuda=torch.version.cuda, starts=list(STARTS), methods=list(METHODS),
            start_source=f"Archived matching 20 x 13 seed-0 FNO + 150 TTT, or all material coefficients 0.5. The {updates} new updates exclude archived TTT.",
            concentration="Average of four raw spatial cells, all valid native time points retained; end padding omitted per material.",
            velocity="Reconstructed independently from the observed concentration and time for each current and noise domain using the benchmark conservation equations.",
            metrics="Absolute dimensionless c and v RMSE over all finite observations across currents, bins and times. c uses two quadrature nodes per bin; v is at bin centers.",
            standard=f"{updates} continuous joint Adam updates, including c and v data. Data weight 1, salt, volume and wall weight 100 each.",
            shared_base=f"{prefit_steps} data-only field-prefit updates (theta fixed), then {joint_steps} joint updates with data weight 0.01 and physics weights 1.",
            correction=f"{correction_steps} updates to additive c/v networks and theta, frozen base fields and scales. Combined c/v data weight 1 or 100, physics weights 1.",
            timing="Measured GPU-synchronized elapsed wall time per material, batch size one. Each correction pipeline includes the shared stage-1 time and correction initialization; count these once per material when summing actual study cost.",
            checkpoint_steps=[0,split,updates], total_updates=updates, spatial_slope_weight=0., field_widths=[3, *hidden_widths, 1],
            per_layer_parameters=counts_per_layer, per_field_parameters=sum(counts_per_layer), material_logits=8,
            architecture="Both c/v base networks and both additive c/v networks use these layer widths, with tanh base and sine correction activations.")
    if materials > 8:
        proto["execution"] = "Single-material GPU fits. The packed base_execution and final_execution fields identify CUDA graph reuse."
    write_json(destination/"protocol.json", proto)
    graph_cache = None
    if reuse_graphs:
        from one_current.pinn_runs.single_fit_graph import SingleFitGraphCache
        graph_cache = SingleFitGraphCache()
    cases = [(domain, count, start) for count, domain, start in product(counts, domains, starts_requested)]
    if materials > 8:
        copied = reuse_eight(hidden_widths, updates, materials)
        completed = new_contexts = 0
        clock = time.perf_counter()
        for row, mid in enumerate(ids):
            for domain, count, start in cases:
                new_contexts += int(fit(mid,domain,count,start,hidden_widths,updates,materials,graph_cache))
                completed += 1
                elapsed = time.perf_counter() - clock
                remaining = len(ids) * len(cases) - completed
                write_json(destination / "progress.json", dict(
                    last_material=mid, last_domain=domain, last_currents=count, last_start=start,
                    complete_materials=completed // len(cases), completed_contexts=completed,
                    total_contexts=len(ids) * len(cases), outcomes=completed * len(METHODS),
                    copied_pilot_contexts=len(copied), trained_contexts_this_invocation=new_contexts,
                    wall_seconds_this_invocation=elapsed,
                    estimated_seconds_remaining=remaining * elapsed / new_contexts if new_contexts else None))
    for domain, count, start in product(domains, counts, starts_requested):
        if materials == 8:
            for mid in ids:
                fit(mid,domain,count,start,hidden_widths,updates,materials,graph_cache)
        aggregate(domain,count,start,hidden_widths,updates,materials)
        print(f"Aggregated {domain}, {count} current(s), {start}", flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domains",nargs="+",choices=DOMAINS,default=DOMAINS)
    parser.add_argument("--currents",nargs="+",type=int,choices=COUNTS,default=COUNTS)
    parser.add_argument("--starts",nargs="+",choices=STARTS,default=STARTS)
    parser.add_argument("--double-parameters", action="store_true",
                        help="Double each field and correction network's parameter count.")
    parser.add_argument("--updates", type=int, choices=tuple(STAGES_BY_UPDATES), default=400)
    parser.add_argument("--materials", type=int, choices=(8, 300), default=8)
    parser.add_argument("--reuse-graphs", action="store_true",
                        help="Reuse matching single-material CUDA graphs across cases of the same shape.")
    args=parser.parse_args()
    run(args.domains,args.currents,args.starts, DOUBLE_WIDTHS if args.double_parameters else BASE_WIDTHS,
        args.updates, args.materials, args.reuse_graphs)
