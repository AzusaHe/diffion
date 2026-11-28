"""Collect the original supervised checkpoints and their training/material provenance.

Run with the original benchmark caches and checkpoints present. Results use paths
relative to the benchmark root unless explicitly named relative to trained_models.
"""
import csv
import hashlib
import json
import math
import shutil
import sys
from itertools import product
from pathlib import Path

import numpy as np
import torch

PACKAGE = Path(__file__).resolve().parent
BENCHMARK = PACKAGE.parent
sys.path.insert(0, str(BENCHMARK))
from grid40_learn import COARSENING_PASSES, FRAMES, ConvMLP
from learn import FNO, MLP, n_params
from prepare import CACHE, DOMAINS, paths


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_write(path, rows):
    with path.open("w", encoding="utf-8", newline="") as dest:
        writer = csv.DictWriter(dest, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def export_materials():
    with np.load(CACHE / "index.npz", allow_pickle=False) as z:
        ids, split, chosen, iota = (z[k] for k in ("ids", "split", "chosen", "iota"))
    clean, noisy = (paths(d) for d in DOMAINS)
    if sorted(clean) != list(range(2000)) or set(clean) != set(noisy):
        raise ValueError("Expected 2,000 paired source materials in original table order")
    if len(set(ids)) != 1947 or chosen.shape != (1947, 3):
        raise ValueError("Unexpected eligible material index")
    positions = dict.fromkeys(("train", "val", "test"), 0)
    rows = []
    for row, (mid, group, slots) in enumerate(zip(ids, split, chosen)):
        if len(set(slots)) != 3:
            raise ValueError(f"Repeated current for material {mid}")
        with np.load(clean[mid]) as a, np.load(noisy[mid]) as b:
            for name in ("source_line_number", "material_sample_id", "iota", "current_density_a_m2"):
                np.testing.assert_array_equal(a[name], b[name])
            line = int(a["source_line_number"])
            np.testing.assert_equal(mid, a["material_sample_id"].item())
            np.testing.assert_equal(mid, line - 2)  # one-based line number includes the header
            np.testing.assert_array_equal(iota[row], a["iota"][slots])
            densities = a["current_density_a_m2"][slots]
        entry = dict(benchmark_row_zero_based=row, material_id=int(mid),
                     source_dataset_row_zero_based=line - 2, source_line_number_one_based=line,
                     split=group, split_row_zero_based=positions[group])
        for j, slot in enumerate(slots, start=1):
            entry[f"current_{j}_source_index_zero_based"] = int(slot)
            entry[f"current_{j}_iota"] = float(iota[row, j-1])
            entry[f"current_{j}_a_m2"] = float(densities[j-1])
        positions[group] += 1
        rows.append(entry)
    if positions != {"train": 1402, "val": 156, "test": 389}:
        raise ValueError(f"Unexpected material split {positions}")
    csv_write(PACKAGE / "materials.csv", rows)
    split_index = {}
    for label, original in (("train", "train"), ("validation", "val"), ("test", "test")):
        selected_rows = np.flatnonzero(split == original)
        split_index[label] = dict(material_ids=ids[selected_rows].tolist(),
                                  benchmark_rows_zero_based=selected_rows.tolist())
    (PACKAGE / "split_indices.json").write_text(json.dumps(split_index, indent=2) + "\n", encoding="utf-8")
    return ids[split == "test"]


def source_paths(nx, seed, stem):
    if nx == 20:
        ckpt = (BENCHMARK / "learned_checkpoints" if seed == 0 else
                BENCHMARK / "learned_seeds" / f"seed_{seed}" / "checkpoints")
        result = BENCHMARK / "results" if seed == 0 else BENCHMARK / "results_seeds" / str(seed)
    else:
        ckpt = CACHE / f"grid{nx}_checkpoints" / f"seed{seed}"
        result = BENCHMARK / f"grid{nx}_results" / f"seed{seed}"
    return ckpt / stem, result / stem


def architecture(arch, k, nx):
    if arch == "fno":
        return dict(module_file="learn.py", class_name="FNO", constructor_kwargs={"currents": k},
                    lift_channels=[5*k+1, 11], padding_time_space=[3, 4], fourier_blocks=4,
                    spectral_channels=11, retained_modes_time_space=[5, 6],
                    spectral_weight_banks_per_block=2, pointwise_kernel=[1, 1], projection_channels=[11, 22],
                    pooling="concatenated masked spatial-temporal mean and maximum",
                    dense_layer_sizes=[44, 128, 8],
                    activation="GELU after Fourier blocks 1 through 3, after projection, and after dense 128.")
    config = dict(module_file="learn.py", class_name="MLP", constructor_kwargs={"currents": k},
                  dense_layer_sizes=[547*k, 40, 40, 8], activation="GELU after each 40-unit hidden layer.")
    if arch == "conv_mlp":
        config.update(module_file="grid40_learn.py", class_name="ConvMLP",
                      constructor_kwargs={"currents": k, "n_space": nx},
                      conv_in_channels=4, conv_out_channels=3, conv_kernel_time_space=[3, 3],
                      conv_padding_time_space=[1, 1], conv_strides_time_space=[[2, 2]]*COARSENING_PASSES[nx]+[[2, 3]],
                      conv_input_channels=["concentration_deviation", "velocity", "time", "validity_mask"],
                      conv_weights_shared_across_selected_currents=True,
                      conv_weights_shared_across_passes=(nx > 40),
                      adaptive_pool_time_space=([26, 40] if nx == 290 else None),
                      conv_features_per_current=[3, 13, 14],
                      activation="GELU after each convolution pass and after each 40-unit hidden layer.")
    return config


def relative(path):
    return path.relative_to(BENCHMARK).as_posix()


def collect_model(nx, nt, domain, k, seed, arch, test_ids):
    stem = f"{domain}__{k}__{arch}"
    source, result = source_paths(nx, seed, stem)
    history = json.loads((source / "training.json").read_text())
    measurements = json.loads(result.with_suffix(".json").read_text())
    losses = np.asarray(history["epoch_val_mse"], dtype=float)
    if losses.shape != (100,) or not np.isfinite(losses).all():
        raise ValueError(f"Invalid original validation trace: {source}")
    best_epoch = int(losses.argmin()) + 1
    np.testing.assert_equal(losses.min(), history["best_val_coef_mse"])
    if "best_epoch" in measurements:
        np.testing.assert_equal(best_epoch, measurements["best_epoch"])
        np.testing.assert_equal(history["best_val_coef_mse"], measurements["best_val_coef_mse"])
    for key, value in (("seed", seed), ("n_train", 1402), ("n_val", 156), ("epochs", 100)):
        np.testing.assert_equal(history[key], value)
        np.testing.assert_equal(measurements[key], value)
    np.testing.assert_equal(measurements["n_test"], len(test_ids))
    np.testing.assert_equal(history["batch_size"], 64)
    np.testing.assert_equal(history["train_s"], measurements["train_s"])
    with np.load(result.with_suffix(".npz"), allow_pickle=False) as z:
        np.testing.assert_array_equal(z["ids"], test_ids)
        if z["theta"].shape != (len(test_ids), 8) or not np.isfinite(z["theta"]).all():
            raise ValueError(f"Invalid archived predictions for {result}")
    rms = np.asarray(history["input_rms"])
    if rms.shape != (2, k) or not np.all(np.isfinite(rms) & (rms > 0)):
        raise ValueError(f"Invalid training input RMS: {source}")
    config = architecture(arch, k, nx)
    cls = {"fno": FNO, "mlp": MLP, "conv_mlp": ConvMLP}[arch]
    model = cls(**config["constructor_kwargs"])
    state = torch.load(source / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    if not all(torch.isfinite(v).all() for v in state.values()):
        raise ValueError(f"Nonfinite checkpoint: {source}")
    params = n_params(model)
    np.testing.assert_equal(params, history["params"])
    np.testing.assert_equal(params, measurements["parameters"])
    leaf = PACKAGE / f"grid_{nx}x{nt}" / domain / f"{k}_current" / f"seed_{seed}" / arch
    leaf.mkdir(exist_ok=True, parents=True)
    weights = leaf / "best.pt"
    original_hash = sha256(source / "best.pt")
    if not weights.exists() or sha256(weights) != original_hash:
        temp = weights.with_suffix(".partial")
        shutil.copyfile(source / "best.pt", temp)
        temp.replace(weights)
    if sha256(weights) != original_hash:
        raise ValueError(f"Published bytes differ from the original best checkpoint: {leaf}")
    total_steps = math.ceil(history["n_train"] / history["batch_size"]) * history["epochs"]
    manifest = dict(
        grid_space_time=[nx, nt], domain=domain, currents=k, selected_current_slots_one_based=list(range(1, k+1)),
        seed=seed, architecture=arch, model=config, trainable_real_parameters=params,
        checkpoint=dict(file="best.pt", format="PyTorch state_dict", sha256=original_hash,
                        bytes=weights.stat().st_size, original_file=relative(source / "best.pt"),
                        selection="first minimum of the original validation coefficient MSE",
                        epoch_one_based=best_epoch, validation_coefficient_mse=float(losses.min())),
        inputs=dict(tensor_order=["batch", "channels", "time", "space"], shape_without_batch=[5*k+1, nt, nx],
                    channel_order="all k concentration deviations, k velocities, k validity masks, k times, k iotas, then shared x",
                    concentration_offset=1.0, concentration_deviation_rms=rms[0].tolist(),
                    velocity_rms=rms[1].tolist(), rms_fit_split="train", time_divisor=36000/33640,
                    dimensionless_iota_divisor=5.0, x_coordinate="linspace(0,1,n_space), endpoints included",
                    invalid_values="zero concentration deviation, velocity, time and frame mask; retain current and x"),
        outputs=dict(raw="eight logits; apply sigmoid to obtain normalized coefficients",
                     order=["D_0", "D_1", "D_2", "D_3", "t_plus_0", "t_plus_1", "t_plus_2", "t_plus_3"]),
        training=dict(n_train=1402, n_validation=156, n_test=389, epochs=100, batch_size=64,
                      shuffle_seed=seed+1, loss=history["loss"], original_epoch_validation_mse=losses.tolist(),
                      optimizer="AdamW", initial_learning_rate=0.001,
                      weight_decay=0.01, betas=[0.9, 0.999], eps=1e-8, scheduler="CosineAnnealingLR",
                      scheduler_steps=total_steps, minimum_learning_rate=0, scheduler_frequency="after every optimizer step"),
        timing=dict(training_seconds=measurements["train_s"], test_seconds_for_389=measurements["test_s"],
                    test_batch=measurements["test_batch"], test_repeats=measurements["test_repeats"],
                    device=measurements["device"]),
        files=dict(materials_from_package_root="materials.csv", split_indices_from_package_root="split_indices.json",
                   epoch_history_from_model_folder="epochs.csv",
                   epoch_history_provenance_from_model_folder="loss_provenance.json",
                   original_history_from_benchmark_root=relative(source / "training.json"),
                   original_history_sha256=sha256(source / "training.json"),
                   predictions_from_benchmark_root=relative(result.with_suffix(".npz")),
                   timing_from_benchmark_root=relative(result.with_suffix(".json"))),
    )
    (leaf / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return dict(grid_space=nx, grid_time=nt, domain=domain, currents=k, seed=seed, architecture=arch,
                model_dir=leaf.relative_to(PACKAGE).as_posix(), parameter_count=params,
                best_epoch_one_based=best_epoch, best_validation_mse=float(losses.min()),
                weight_bytes=weights.stat().st_size, weight_sha256=original_hash)


def main():
    torch.set_num_threads(1)
    ids = export_materials()
    models = []
    for nx, nt in {20: 13, **FRAMES}.items():
        for domain, k, seed, arch in product(DOMAINS, (1, 3), range(3), ("fno", "mlp" if nx == 20 else "conv_mlp")):
            models.append(collect_model(nx, nt, domain, k, seed, arch, ids))
    if len(models) != 120 or len({r["model_dir"] for r in models}) != 120:
        raise ValueError("Expected 120 unique model folders")
    csv_write(PACKAGE / "inventory.csv", models)
    print(f"Verified {len(models)} checkpoints, {sum(r['weight_bytes'] for r in models):,} bytes.")


if __name__ == "__main__":
    main()
