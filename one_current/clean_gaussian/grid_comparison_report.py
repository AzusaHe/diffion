"""Pair supervised resolutions by material, current selection and training seed."""
import csv
import io
import json
from base64 import b64encode
from copy import deepcopy
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

if __package__:
    from .equal_budget_report import summary as accuracy_summary
else:
    from equal_budget_report import summary as accuracy_summary


GRIDS = ((20, 13), (40, 26), (80, 52), (160, 104), (290, 151))
DOMAINS = ("clean", "gaussian_noisy")


def slug_for(width, family):
    if width == 20:
        return family
    return f"{'conv_mlp' if family == 'mlp' else 'fno'}_grid{width}"


def model_name(width, frames, family):
    name = "FNO" if family == "fno" else "MLP" if width == 20 else "Conv-MLP"
    return f"{name} · {width} × {frames}"


def result_folder(width, seed):
    if width > 20:
        return Path(f"grid{width}_results") / f"seed{seed}"
    return Path("results") if seed == 0 else Path("results_seeds") / str(seed)


def supervised_architecture(original, width, frames):
    family = original["slug"]
    model = deepcopy(original)
    model.update(slug=slug_for(width, family), name=model_name(width, frames, family))
    model["sources"] = [
        "one_current/clean_gaussian/learn.py", "one_current/clean_gaussian/grid40_learn.py",
        f"one_current/clean_gaussian/grid{width}_results/seed{{0,1,2}}/",
    ]
    for count, key in ((1, "one"), (3, "three")):
        config = model[key]
        if family == "fno":
            config["input"] = f"{5 * count + 1} × {frames} × {width} tensor (channels × frames × spatial bins). Each selected current provides five channels. All currents share one spatial coordinate."
            config["layers"] = [line.replace("16 × 24", f"{frames+3} × {width+4}").replace("13 × 20", f"{frames} × {width}") for line in config["layers"]]
        else:
            config["params"] = f"{int(config['params'].replace(',', '')) + 111:,}"
            config["input"] = f"{count} current{'s' if count == 3 else ''}, each with four {frames} × {width} maps. The maps contain normalized concentration deviation, reconstructed velocity, observation time and the validity mask."
            layers = []
            stage_t, stage_x = frames, width
            for pass_index in range({40: 0, 80: 1, 160: 2, 290: 2}[width]):
                stage_t, stage_x = (stage_t + 1) // 2, (stage_x + 1) // 2
                action = "Per current, apply the shared" if pass_index == 0 else "Reuse the same"
                layers.append(f"{action} 3 × 3 convolution, 4 → 3 channels, with unit padding and stride (2 time, 2 space). Mask its GELU outputs and append the pooled validity mask to form 4 × {stage_t} × {stage_x}.")
            if width == 290:
                layers.append("Reduce the 38 × 73 feature maps to 26 × 40 using mask-aware adaptive area averaging. Each output averages valid entries and retains a deterministic fourth validity channel. This step has zero trainable parameters.")
            layers += [f"{'Reapply the same' if width > 40 else 'Per current, apply a shared'} 3 × 3 convolution, 4 → 3 channels, with unit padding and stride (2 time, 3 space). Apply GELU and mask receptive fields with no valid observations. All convolution passes together learn 108 weights and three biases.",
                       f"Flatten the 3 × 13 × 14 outputs and append the current magnitude. Concatenate {count} × 547 features for the dense head."]
            config["layers"] = layers + config["layers"]
    if family == "mlp":
        model["activation"] = "GELU follows each convolution pass and each 40-unit hidden layer. The eight output logits pass through sigmoid."
        model["notes"] = "The convolution shares weights across all selected currents and repeated passes. One and three currents have different first dense layers. Each domain, current count, grid and seed has its own fitted checkpoint."
    return model


def input_bounds(root, ids, preview):
    """Verify all grids use the original index and obtain bounds from valid observations."""
    if __package__:
        from .grid40_prepare import mmap_grid_field
    else:
        from grid40_prepare import mmap_grid_field
    cache = root / "cache"
    keys = ("ids", "split", "chosen", "iota", "theta")
    with np.load(cache / "index.npz") as z:
        index = {k: z[k] for k in keys}
    test = np.flatnonzero(index["split"] == "test")
    np.testing.assert_array_equal(index["ids"][test], ids)
    ranges = {}
    for width, frames in GRIDS:
        path = cache / ("binned.npz" if width == 20 else f"binned_{width}x{frames}.npz")
        if preview and not path.exists():
            return {}
        with np.load(path) as z:
            for k in keys:
                np.testing.assert_array_equal(z[k], index[k])
        for domain in DOMAINS:
            field = mmap_grid_field(path, domain)
            assert field.shape == (len(index["ids"]), 3, frames, width)
            bounds = {n: np.empty((2, len(test)), dtype=field.dtype) for n in (1, 3)}
            for offset in range(0, len(test), 16):
                block = slice(offset, offset + 16)
                obs = field[test[block]]
                for n, (lo, hi) in bounds.items():
                    lo[block] = np.nanmin(obs[:, :n], axis=(1, 2, 3))
                    hi[block] = np.nanmax(obs[:, :n], axis=(1, 2, 3))
            for n, (lo, hi) in bounds.items():
                assert np.isfinite(lo).all() and np.isfinite(hi).all() and np.all(lo < hi)
                ranges[width, domain, n] = lo, hi
            del field
    return ranges


def grid_figures(rows):
    figures = {}
    for domain in DOMAINS:
        for seed in range(3):
            selected = [r for r in rows if r["domain"] == domain and r["seed"] == seed]
            if len(selected) != 4 * len(GRIDS):
                continue
            for metric, label in (("D_median_pct", "Median D error (%)"), ("t_median_rmse", "Median t₊ RMSE")):
                fig = Figure(figsize=(6, 3.6), dpi=120, layout="constrained")
                FigureCanvasAgg(fig)
                ax = fig.subplots()
                for family, style in (("fno", "o-"), ("mlp", "s--")):
                    for count, color in ((1, "#006b63"), (3, "#a74328")):
                        matches = sorted((r for r in selected if r["family"] == family and r["currents"] == count),
                                         key=lambda r: r["space_bins"])
                        method = "FNO" if family == "fno" else "MLP / Conv-MLP"
                        ax.plot(range(len(GRIDS)), [r[f"common_{metric}"] for r in matches], style, color=color,
                                linewidth=1.7, markersize=5, label=f"{method}, {count} current{'s' if count == 3 else ''}")
                ax.set_xticks(range(len(GRIDS)), [f"{w} × {f}" for w, f in GRIDS])
                ax.set(xlabel="Spatial bins × maximum observed frames", ylabel=label, xlim=(-0.15, len(GRIDS)-0.85))
                ax.grid(axis="y", color="#d4dfda", linewidth=0.7)
                ax.spines[["top", "right"]].set_visible(False)
                ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=2, frameon=False, fontsize=8)
                fig.suptitle(f"{'Clean' if domain == 'clean' else 'Gaussian-noisy'} · seed {seed} · shared interval", fontsize=10)
                stream = io.BytesIO()
                fig.savefig(stream, format="png")
                figures[f"{domain}__{seed}__{metric}"] = b64encode(stream.getvalue()).decode("ascii")
    return figures


def build_grids(root, materials, bank, seeds, errors, equal, preview):
    ids = np.array([m["id"] for m in materials])
    truth = np.array([m["truth"] for m in materials])
    ranges = input_bounds(root, ids, preview)
    summaries, primary = [], []
    for domain in DOMAINS:
        for count in (1, 3):
            if not ranges:
                continue
            bounds = [ranges[w, domain, count] for w, _ in GRIDS]
            common_lo = np.maximum(0, np.maximum.reduce([lo for lo, _ in bounds]))
            common_hi = np.minimum(2, np.minimum.reduce([hi for _, hi in bounds]))
            assert np.all(common_lo < common_hi)
            original_lo, original_hi = bounds[0]
            matches_original = ((common_lo == np.maximum(0, original_lo)) &
                                (common_hi == np.minimum(2, original_hi)))
            for family in ("fno", "mlp"):
                for seed in range(3):
                    scores = {}
                    for (width, frames), (lo, hi) in zip(GRIDS, bounds):
                        model = "conv_mlp" if width > 20 and family == "mlp" else family
                        stem = result_folder(width, seed) / f"{domain}__{count}__{model}"
                        meta_path = (root / stem).with_suffix(".json")
                        if preview and not meta_path.exists():
                            continue
                        meta = json.loads(meta_path.read_text(encoding="utf-8"))
                        with np.load((root / stem).with_suffix(".npz")) as z:
                            np.testing.assert_array_equal(z["ids"], ids)
                            np.testing.assert_allclose(z["lo"], lo, atol=1e-6, rtol=0)
                            np.testing.assert_allclose(z["hi"], hi, atol=1e-6, rtol=0)
                            theta = z["theta"]
                        assert (meta["seed"], meta["n_train"], meta["n_val"], meta["n_test"], meta["epochs"]) == (seed, 1402, 156, len(ids), 100)
                        expected_params = (65741 if count == 1 else 65851) if family == "fno" else (23888 if count == 1 else 67648) + 111 * (width > 20)
                        assert meta["parameters"] == expected_params, (stem, expected_params)
                        assert (meta["test_batch"], meta["test_repeats"]) == (128, 10)
                        assert meta["train_s"] > 0 and meta["test_s"] > 0
                        if width > 20:
                            assert (meta["domain"], meta["currents"], meta["architecture"], meta["n_space"], meta["n_time"]) == (domain, count, model, width, frames)
                            assert meta["optimizer"] == "AdamW" and meta["learning_rate"] == 0.001 and meta["batch_size"] == 64
                            assert 1 <= meta["best_epoch"] <= meta["epochs"] and np.isfinite(meta["best_val_coef_mse"])
                        assert theta.shape == truth.shape and np.isfinite(theta).all()
                        assert np.all((theta >= 0) & (theta <= 1))
                        d, t = errors(theta, truth, lo, hi)
                        dc, tc = errors(theta, truth, common_lo, common_hi)
                        scores[width] = dc, tc
                        per_s = meta["test_s"] / len(ids)
                        row = dict(domain=domain, currents=count, family=family, seed=seed, space_bins=width, max_frames=frames,
                                   model=model, n_train=meta["n_train"], n_val=meta["n_val"], n_test=len(ids),
                                   parameters=meta["parameters"], own_D_median_pct=float(np.median(d)),
                                   own_t_median_rmse=float(np.median(t)),
                                   own_median_c_span=float(np.median(np.minimum(2, hi) - np.maximum(0, lo))),
                                   common_grids_space_x_time="|".join(f"{w}x{f}" for w, f in GRIDS),
                                   common_D_median_pct=float(np.median(dc)), common_t_median_rmse=float(np.median(tc)),
                                   common_median_c_span=float(np.median(common_hi - common_lo)),
                                   n_common_bounds_equal_20=int(np.count_nonzero(matches_original)),
                                   common_paired_D_change_vs_20_pctpt=float(np.median(dc - scores[20][0])),
                                   common_paired_t_change_vs_20=float(np.median(tc - scores[20][1])),
                                   train_s=meta["train_s"], test_s=meta["test_s"], mean_inference_s_per_material=per_s,
                                   inference_batch=meta["test_batch"], inference_repeats=meta["test_repeats"], device=meta["device"],
                                   result_path=stem.as_posix())
                        summaries.append(row)
                        if width == 20:
                            continue
                        slug = slug_for(width, family)
                        key = f"{domain}__{count}__{slug}"
                        seeds.setdefault(key, []).append(dict(seed=seed, d=row["own_D_median_pct"],
                                                              t=row["own_t_median_rmse"], train=meta["train_s"], test=per_s))
                        if seed == 0:
                            meta.setdefault("description", f"Fresh supervised {model_name(width, frames, family)} fit on {domain} data. "
                                            f"The validation checkpoint from epoch {meta['best_epoch']} of 100 produced these predictions.")
                            obs = np.column_stack((ids, d, t, lo, hi, theta)).tolist()
                            for observation, mid in zip(obs, ids):
                                observation[0] = int(mid)
                            bank[key] = dict(name=model_name(width, frames, family), slug=slug, domain=domain, count=count,
                                             n=len(ids), meta=meta, family="Supervised and TTT", training=meta["train_s"],
                                             tuning=0, stage_s=per_s, init_s=0, per_s=per_s, individual_times=None,
                                             observations=obs)
                            primary.append(accuracy_summary(bank[key], np.asarray(obs), {}, [], equal["shared_ids"]))
    if not preview:
        assert len(summaries) == len(GRIDS) * 24 and len(primary) == (len(GRIDS) - 1) * 8
    summary_csv = io.StringIO(newline="")
    if summaries:
        writer = csv.DictWriter(summary_csv, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)
    if not preview:
        (root / "grid_accuracy_cost.csv").write_bytes(summary_csv.getvalue().encode("utf-8"))
    added_csv = io.StringIO(newline="")
    if primary:
        writer = csv.DictWriter(added_csv, fieldnames=next(csv.reader(io.StringIO(equal["csv"]))))
        writer.writerows(primary)
    prep = {}
    for w, f in GRIDS[1:]:
        path = root / "cache" / f"preparation_{w}x{f}.json"
        if path.exists():
            prep[str(w)] = json.loads(path.read_text(encoding="utf-8"))
    return dict(grids=GRIDS, rows=summaries, csv=summary_csv.getvalue(), accuracy_csv_append=added_csv.getvalue(),
                preparation=prep, figures=grid_figures(summaries))
