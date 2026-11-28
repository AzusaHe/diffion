"""Verify the supplied corrected grids and publish their observation and material inventory."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from corrected_grid_inputs import GRIDS, OUTPUT, SOURCE


EXPECTED_IDS = set(range(2000))
COMMON = ("material_sample_id", "D_coefficients_01", "t_plus_coefficients_01",
          "rho_coefficients_01", "iota", "current_density_a_m2", "status", "run_fingerprint")


def source_files(folder):
    files = list(folder.glob("*/samples/material_*.npz"))
    indexed = {int(p.stem[9:]): p for p in files}
    if len(files) != len(indexed) or set(indexed) != EXPECTED_IDS:
        raise ValueError(f"{folder}: expected one file for each ID 0 through 1999; "
                         f"{len(files) - len(indexed)} duplicates, {len(EXPECTED_IDS - indexed.keys())} missing")
    return indexed


def geometry(nx):
    edges = np.linspace(0., 1160., nx + 1)
    native = np.arange(1160)
    overlap = np.maximum(0., np.minimum(edges[1:, None], native + 1) - np.maximum(edges[:-1, None], native))
    centers = (overlap @ ((native + .5) / 1160.)) / np.diff(edges)
    return edges, centers


def clean_bin_means(c, edges):
    # Integrate cell averages at bin faces, including partial native cells.
    sums = np.empty((*c.shape[:-1], c.shape[-1] + 1), np.float64)
    sums[..., 0] = 0
    np.cumsum(c, axis=-1, dtype=np.float64, out=sums[..., 1:])
    cell = edges.astype(np.int64)
    integral = sums[..., cell] + (edges - cell) * c[..., cell.clip(max=c.shape[-1] - 1)]
    return np.diff(integral) / np.diff(edges)


def digest(data):
    return hashlib.sha256(data).digest()


def audit(source, index_path, clean_source=None, legacy_grid=None, workers=4):
    with np.load(index_path) as z:
        ids, splits, chosen, iota, theta = (z[key] for key in ("ids", "split", "chosen", "iota", "theta"))
    split_counts = dict(Counter(splits.tolist()))
    if split_counts != {"train": 1402, "val": 156, "test": 389} or len(set(ids)) != len(ids):
        raise ValueError("Material index does not match the benchmark split")
    positions = {int(mid): i for i, mid in enumerate(ids)}
    folders = {nx: source / f"{nx}_{nt}_noised" for nx, nt in GRIDS.items()}
    files = {nx: source_files(folder) for nx, folder in folders.items()}
    clean = None if clean_source is None else source_files(clean_source)
    legacy = None
    if legacy_grid is not None:
        from grid40_prepare import mmap_grid_field
        with np.load(legacy_grid) as z:
            for key, value in (("ids", ids), ("chosen", chosen), ("split", splits)):
                np.testing.assert_array_equal(z[key], value)
        legacy = mmap_grid_field(legacy_grid, "gaussian_noisy")
        if clean is None:
            raise ValueError("A paired clean source is required to measure legacy observation noise")
    axes = {nx: geometry(nx) for nx in GRIDS}
    frames = {nx: np.rint(np.linspace(0, 150, nt)).astype(int) for nx, nt in GRIDS.items()}

    def check_material(mid):
        row = positions.get(mid)
        test = row is not None and splits[row] == "test"
        native_clean = None
        with np.load(files[290][mid]) as ref:
            common = {k: ref[k] for k in COMMON}
            raw_mask, seconds, tau = (ref[k] for k in ("valid_time_mask", "time_s", "tau"))
        np.testing.assert_array_equal(seconds, np.arange(151) * 240.)
        if int(common["material_sample_id"]) != mid:
            raise ValueError(f"Stored material ID differs from filename for material {mid}")
        if clean is not None:
            with np.load(clean[mid]) as z:
                for k, value in {**common, "valid_time_mask": raw_mask, "time_s": seconds, "tau": tau}.items():
                    np.testing.assert_array_equal(z[k], value)
                if test:
                    native_clean = z["concentration_nd"][chosen[row]]
        if row is not None:
            picked = chosen[row]
            if len(set(picked)) != 3:
                raise ValueError(f"Repeated selected current for material {mid}")
            np.testing.assert_array_equal(common["iota"][picked], iota[row])
            np.testing.assert_array_equal(np.r_[common["D_coefficients_01"], common["t_plus_coefficients_01"]], theta[row])
        results = {}
        for nx, nt in GRIDS.items():
            raw = files[nx][mid].read_bytes()
            edges, centers = axes[nx]
            ix = frames[nx]
            with np.load(io.BytesIO(raw)) as z:
                for k, value in common.items():
                    np.testing.assert_array_equal(z[k], value)
                np.testing.assert_array_equal(z["noise_seed"], [20261124, nx, mid])
                if z["noise_sigma_ref_mM"].item() != 2.2 or "gaussian" not in str(z["noise_model"]).lower():
                    raise ValueError(f"Unexpected noise model for {mid} at {nx} cells")
                np.testing.assert_allclose(z["x_cells_nd"], centers, rtol=0, atol=1.e-14)
                np.testing.assert_allclose(z["x_faces_nd"], edges / 1160, rtol=0, atol=1.e-14)
                np.testing.assert_array_equal(z["time_s"], seconds[ix])
                np.testing.assert_array_equal(z["tau"], tau[ix])
                valid, c = z["valid_time_mask"], z["concentration_nd"]
                np.testing.assert_array_equal(valid, raw_mask[:, ix])
                if (c.shape != (8, nt, nx) or c.dtype != np.float32 or valid.dtype != np.bool_
                        or not np.isfinite(c[valid]).all() or not np.isnan(c[~valid]).all()):
                    raise ValueError(f"Invalid field grid or padding for material {mid} at {nx} cells")
            entry = {"digest": digest(raw)}
            if test:
                mask = valid[picked]
                n = mask.sum(axis=1)
                same = []
                for selected, observed in zip(picked, mask):
                    native = np.flatnonzero(raw_mask[selected])
                    old = native[np.rint(np.linspace(0, len(native)-1, min(nt, len(native)))).astype(int)]
                    same.append(np.array_equal(ix[observed], old))
                entry.update(counts=n, same=np.asarray(same))
                if native_clean is not None:
                    baseline = clean_bin_means(native_clean[:, ix], edges)
                    delta = c[picked] - baseline
                    sq = np.where(mask[..., None], np.square(delta), 0.).sum(axis=(1, 2))
                    entry.update(ss=sq)
                    if nx == 290 and legacy is not None:
                        diff = np.asarray(legacy[row], np.float64) - baseline
                        entry["legacy_ss"] = np.where(mask[..., None], np.square(diff), 0.).sum(axis=(1, 2))
            results[nx] = entry
        return results

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(check_material, sorted(EXPECTED_IDS)))
    out = dict(dataset_revision="corrected_grid_native_v1", source_materials=2000, total_source_files=10000,
               material_id_range=[0, 1999], split=split_counts, indexed_materials=len(ids),
               material_index_sha256=hashlib.sha256(index_path.read_bytes()).hexdigest(),
               source_digest_definition="SHA-256 of file SHA-256 digests concatenated in ascending material ID order.",
               source_checks=["Filename and stored IDs, concentration dtype and shape, batch 10 ID coverage, noise seed and model",
                              "Independent current/label/fingerprint agreement across five resolutions and selected benchmark IDs",
                              "Original material split, distinct current choices, matching selected iota and supervised labels",
                              "Native time vector and masks sliced at rounded equally spaced indices of 0 through 150",
                              "Finite observed concentration, NaN padding, exact grid faces and area-weighted native cell centers"],
               test_time_comparison="Legacy per-current sampling takes up to the grid frame count across each valid native trajectory, including its endpoints.",
               paired_native_clean_checked=clean is not None, grids={})
    if clean is not None:
        out["native_clean_check"] = ("All 2,000 native files agree on IDs, coefficients, currents, statuses, times and masks. "
                                     "Noise RMS uses the exact valid test frames and conservative spatial means of native clean concentrations.")
    for nx, nt in GRIDS.items():
        indexed_digest = hashlib.sha256(b"".join(rows[mid][nx]["digest"] for mid in ids)).hexdigest()
        source_digest = hashlib.sha256(b"".join(r[nx]["digest"] for r in rows)).hexdigest()
        tail = [mid for mid, path in files[nx].items() if "batch_10_1800_1999" in path.parent.parent.name]
        if sorted(tail) != list(range(1800, 2000)):
            raise ValueError(f"Unexpected batch 10 coverage at {nx} cells")
        summary = dict(source_folder=folders[nx].name, concentration_shape=[8, nt, nx], dtype="float32",
                       files=len(rows), unique_ids=len(EXPECTED_IDS), missing_ids=0, duplicate_ids=0, stored_id_mismatches=0,
                       concentration_shape_mismatches=0, noise_seed_mismatches=0, batch_10_valid_files=len(tail),
                       batch_10_id_range=[min(tail),max(tail)], source_sha256=source_digest,
                       indexed_source_sha256=indexed_digest, test={})
        obs = [rows[mid][nx] for mid in ids[splits == "test"]]
        counts = np.stack([r["counts"] for r in obs])
        same = np.stack([r["same"] for r in obs])
        for k in (1, 3):
            one = counts[:, :k] == 1
            points = int(counts[:, :k].sum()) * nx
            stats = dict(materials=len(obs), trajectories=len(obs)*k, observed_field_values=points,
                         single_frame_trajectories=int(one.sum()), all_currents_single_frame_materials=int(one.all(axis=1).sum()),
                         materials_matching_legacy_observed_times=int(same[:, :k].all(axis=1).sum()))
            if clean is not None:
                stats["corrected_noise_rms_vs_native_clean"] = float(np.sqrt(sum(r["ss"][:k].sum() for r in obs) / points))
            if nx == 290 and legacy is not None:
                stats["legacy_noise_rms_vs_native_clean"] = float(np.sqrt(sum(r["legacy_ss"][:k].sum() for r in obs) / points))
            summary["test"][str(k)] = stats
        out["grids"][str(nx)] = summary
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--index", type=Path, default=OUTPUT / "material_index.npz")
    parser.add_argument("--output", type=Path, default=OUTPUT / "source_inventory.json")
    parser.add_argument("--clean-source", type=Path, help="Optional native clean data root for paired noise RMS")
    parser.add_argument("--legacy-290-cache", type=Path, help="Optional previous 290×151 archive for paired legacy noise RMS")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    inventory = audit(args.source, args.index, args.clean_source, args.legacy_290_cache, args.workers)
    pending = args.output.with_suffix(".tmp")
    pending.write_text(json.dumps(inventory,indent=2)+"\n",encoding="utf-8")
    pending.replace(args.output)
    print(f"Verified {inventory['total_source_files']} files: {args.output}")
