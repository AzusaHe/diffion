"""Conservatively coarsen the paired source fields using the original benchmark index."""
import argparse
import hashlib
import json
import struct
import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

if __package__:
    from .prepare import CACHE, DOMAINS, USABLE, paths, save
else:
    from prepare import CACHE, DOMAINS, USABLE, paths, save


NATIVE_X, NATIVE_T = 1160, 151
GRID_TIMES = {40: 26, 80: 52, 160: 104, 290: 151}


def mmap_grid_field(cache_path, domain):
    """Open one concentration array inside a stored NumPy ZIP without loading it."""
    if domain not in DOMAINS:
        raise ValueError(f"Unknown domain {domain}")
    with zipfile.ZipFile(cache_path) as archive:
        entry = archive.getinfo(f"{domain}.npy")
        if entry.compress_type != zipfile.ZIP_STORED:
            raise ValueError("Memory mapping requires an uncompressed np.savez archive")
    with open(cache_path, "rb") as stream:
        stream.seek(entry.header_offset)
        header = struct.unpack("<4s5H3L2H", stream.read(30))
        if header[0] != b"PK\x03\x04":
            raise ValueError("Invalid ZIP local file header")
        stream.seek(header[-2] + header[-1], 1)  # local filename and extra data
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError(f"Unsupported npy header version {version}")
        offset = stream.tell()
    if len(shape) != 4 or shape[1] != 3 or dtype != np.float32:
        raise ValueError(f"Unexpected grid field {shape}, {dtype}")
    return np.memmap(cache_path, mode="r", dtype=dtype, shape=shape, offset=offset,
                     order="F" if fortran else "C")


def spatial_average(native, nx):
    """Conserve concentration integrated over each uniform output control volume."""
    if nx == 290:
        return native.reshape(-1, 290, 4).mean(axis=-1)
    paired = native.reshape(-1, 40, 29)
    if nx == 40:
        return paired.mean(axis=-1)
    if nx == 160:
        # Each group of 29 native cells is quartered at 7.25, 14.5, 21.75.
        a = paired[..., :7].sum(-1) + paired[..., 7] * 0.25
        b = paired[..., 7] * 0.75 + paired[..., 8:14].sum(-1) + paired[..., 14] * 0.5
        c = paired[..., 14] * 0.5 + paired[..., 15:21].sum(-1) + paired[..., 21] * 0.75
        d = paired[..., 21] * 0.25 + paired[..., 22:].sum(-1)
        return (np.stack((a, b, c, d), axis=-1) / 7.25).reshape(-1, 160)
    if nx != 80:
        raise ValueError(f"Unsupported grid {nx}")
    # A pair of equal-width bins shares 29 native cells. The center native
    # cell contributes half its content to each of the two output bins.
    middle = paired[..., 14] * 0.5
    left = (paired[..., :14].sum(axis=-1) + middle) / 14.5
    right = (middle + paired[..., 15:].sum(axis=-1)) / 14.5
    return np.stack((left, right), axis=-1).reshape(-1, 80)


def audit_volume_averages(native, binned, nx):
    """Independent integral of the piecewise-constant native FV observations."""
    samples = native[[0, -1]]
    edges = np.linspace(0, NATIVE_X, nx + 1)
    integrals = np.concatenate((np.zeros((2, 1)), samples.cumsum(axis=-1)), axis=-1)
    edge_integrals = np.stack([np.interp(edges, np.arange(NATIVE_X + 1), row) for row in integrals])
    np.testing.assert_allclose(binned[[0, -1]], np.diff(edge_integrals, axis=-1) / np.diff(edges),
                               atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(binned.mean(axis=-1), native.mean(axis=-1), atol=1e-14, rtol=1e-14)


class ObservationGrid:
    def __init__(self, nx, count, temporary):
        self.nx, self.nt = nx, GRID_TIMES[nx]
        self.name = f"{nx}x{self.nt}"
        shape = (count, 3, self.nt)
        self.frame = np.zeros(shape, np.int64)
        self.mask = np.zeros(shape, bool)
        self.tau = np.zeros(shape, np.float64)
        self.observed = {
            d: np.lib.format.open_memmap(temporary / f"{self.name}_{d}.npy", mode="w+",
                                         dtype=np.float32, shape=(*shape, nx))
            for d in DOMAINS
        }
        self.ranges = {d: np.zeros((count, 2), dtype=np.float32) for d in DOMAINS}
        self.noise_ss = np.zeros(count)

    def select(self, row, valid, times):
        for j, okay in enumerate(valid):
            available = np.flatnonzero(okay)
            n = min(len(available), self.nt)
            pick = available[np.rint(np.linspace(0, len(available) - 1, n)).astype(int)]
            if n < 2 or np.any(np.diff(pick) <= 0) or not np.array_equal(pick[[0, -1]], available[[0, -1]]):
                raise ValueError(f"Invalid frame selection for row {row}, current {j}")
            self.frame[row, j, :n] = pick
            self.mask[row, j, :n] = True
        self.tau[row] = times[self.frame[row]]

    def store(self, row, domain, native, audit):
        binned = spatial_average(native, self.nx)
        if audit:
            audit_volume_averages(native, binned, self.nx)
        dest = self.observed[domain][row]
        dest.fill(np.nan)
        dest[self.mask[row]] = binned
        if not np.array_equal(np.isfinite(dest), np.broadcast_to(self.mask[row, :, :, None], dest.shape)):
            raise ValueError(f"Invalid finite mask for {domain} {self.name} row {row}")
        self.ranges[domain][row] = (binned.min(), binned.max())

    def publish(self, common, summary, start):
        for arr in self.observed.values():
            arr.flush()
        save(f"binned_{self.name}.npz", **common, **self.observed, mask=self.mask, tau=self.tau, frame=self.frame)
        counts = self.mask.sum(-1)
        summary = dict(summary, space_bins=self.nx, cells_per_bin=NATIVE_X / self.nx,
                       max_selected_frames=self.nt, selected_frames_per_current=[int(counts.min()), int(counts.max())],
                       **{f"trajectories_with_fewer_than_{self.nt}": int((counts < self.nt).sum())},
                       valid_frames_total=int(counts.sum()),
                       concentration_ranges={d: [float(a[:, 0].min()), float(a[:, 1].max())]
                                             for d, a in self.ranges.items()},
                       selected_native_noise_rms=float(np.sqrt(self.noise_ss.sum() / (NATIVE_X * counts.sum()))),
                       prepare_wall_s=time.perf_counter() - start)
        temp = CACHE / f"preparation_{self.name}.tmp.json"
        temp.write_text(json.dumps(summary, indent=2) + "\n")
        temp.replace(CACHE / f"preparation_{self.name}.json")
        print(json.dumps(summary, indent=2), flush=True)

    def close(self):
        self.observed.clear()


def build(spaces, workers=4):
    spaces = tuple(dict.fromkeys(spaces))
    start = time.perf_counter()
    index_path = CACHE / "index.npz"
    with np.load(index_path) as index:
        common = {key: index[key] for key in ("ids", "chosen", "iota", "theta", "split")}
    ids, chosen = common["ids"], common["chosen"]
    split_counts = {s: int((common["split"] == s).sum()) for s in ("train", "val", "test")}
    if split_counts != {"train": 1402, "val": 156, "test": 389} or len(set(ids)) != 1947:
        raise ValueError(f"Unexpected original material splits: {split_counts}")
    files = {domain: paths(domain) for domain in DOMAINS}
    if files["clean"].keys() != files["gaussian_noisy"].keys() or set(ids) - files["clean"].keys():
        raise ValueError("The two source folders must contain paired files for every indexed material")
    native_counts = np.zeros((len(ids), 3), np.int64)
    with np.load(files["clean"][ids[0]]) as first:
        reference_x = first["x_cells_nd"]
    np.testing.assert_allclose(reference_x, (np.arange(NATIVE_X) + 0.5) / NATIVE_X)

    def bin_material(row):
        mid, currents = ids[row], chosen[row]
        if len(set(currents)) != 3:
            raise ValueError(f"Repeated current in benchmark index for {mid}")
        with np.load(files["clean"][mid]) as clean, np.load(files["gaussian_noisy"][mid]) as noisy:
            for key in ("material_sample_id", "status", "valid_time_mask", "D_coefficients_01",
                        "t_plus_coefficients_01", "iota", "tau", "x_cells_nd", "run_fingerprint"):
                np.testing.assert_array_equal(clean[key], noisy[key])
            np.testing.assert_array_equal(clean["x_cells_nd"], reference_x)
            np.testing.assert_array_equal(clean["material_sample_id"].item(), mid)
            np.testing.assert_array_equal(clean["iota"][currents], common["iota"][row])
            coeffs = np.r_[clean["D_coefficients_01"], clean["t_plus_coefficients_01"]]
            np.testing.assert_array_equal(coeffs, common["theta"][row])
            if "gaussian" not in str(noisy["noise_model"]).lower() or noisy["noise_sigma_ref_mM"].item() != 2.2:
                raise ValueError(f"Unexpected Gaussian noise metadata for material {mid}")
            valid = clean["valid_time_mask"][currents]
            times = clean["tau"]
            if (valid.shape != (3, NATIVE_T) or times.shape != (NATIVE_T,)
                    or not np.isin(clean["status"][currents], USABLE).all() or np.any(valid.sum(-1) < 2)
                    or np.any(valid[:, 1:] & ~valid[:, :-1]) or not np.all(np.diff(times) > 0)):
                raise ValueError(f"Unexpected source observation mask, status or time axis for material {mid}")
            native_counts[row] = valid.sum(-1)
            for grid in grids:
                grid.select(row, valid, times)
            clean_fields = {}
            for domain, source in zip(DOMAINS, (clean, noisy)):
                raw = source["concentration_nd"]  # decompress only once per source, shared by all target grids
                if raw.shape[1:] != (NATIVE_T, NATIVE_X):
                    raise ValueError(f"Unexpected native grid {raw.shape} in {domain}/{mid}")
                for grid in grids:
                    j, k = np.nonzero(grid.mask[row])
                    native = raw[currents[j], grid.frame[row, j, k]].astype(np.float64)
                    if not np.isfinite(native).all():
                        raise ValueError(f"Nonfinite observed concentration in {domain}/{mid}")
                    grid.store(row, domain, native, row in (0, len(ids) // 2, len(ids) - 1))
                    if domain == "clean":
                        clean_fields[grid.nx] = native
                    else:
                        grid.noise_ss[row] = np.square(native - clean_fields.pop(grid.nx)).sum()

    with tempfile.TemporaryDirectory(dir=CACHE, prefix="build_grid_") as temp:
        grids = [ObservationGrid(nx, len(ids), Path(temp)) for nx in spaces]
        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for i, _ in enumerate(pool.map(bin_material, range(len(ids))), 1):
                    if i % 200 == 0:
                        print(f"Prepared {i}/{len(ids)}", flush=True)
            summary = dict(
                index_file="cache/index.npz", index_sha256=hashlib.sha256(index_path.read_bytes()).hexdigest(),
                source_domains=list(DOMAINS), source_materials_per_domain=len(files["clean"]),
                eligible=len(ids), split=split_counts, currents="original index column 0 for one, columns 0-2 for three",
                native_space=NATIVE_X, native_time=NATIVE_T,
                spatial_average="uniform output FV bins; native cells weighted by exact overlap length",
                selection="rounded uniform indices among valid native frames including endpoints",
                native_valid_frames_per_current=[int(native_counts.min()), int(native_counts.max())],
                grids_sharing_raw_read=[grid.name for grid in grids],
                timing_definition="elapsed wall seconds from shared preparation startup to publication of this grid",
            )
            for grid in grids:
                grid.publish(common, summary, start)
        finally:
            for grid in grids:
                grid.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--space", nargs="+", type=int, choices=GRID_TIMES, default=[40])
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    build(args.space, args.workers)
