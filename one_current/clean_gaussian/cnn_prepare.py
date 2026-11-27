"""Prepare fixed, paired 48-keyframe CNN observations from the two source domains."""
import argparse
import heapq
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[2] / "latest_synthetic_ml"
sys.path.insert(0, str(SOURCE / "src"))
import three_current_cnn as cnn

DOMAINS = ("clean", "gaussian_noisy")
FRAMES = 48


def keyframes(field, t, valid):
    """Greedy maximum-chord selection with each unchanged segment scored just once."""
    indices = np.flatnonzero(valid)
    c = field[indices].astype(np.float64)
    times = t[indices]
    if len(indices) <= FRAMES:
        return indices
    tolerance = 32 * np.finfo(np.float64).eps * max(1.0, np.abs(c).max())
    segments = []

    def score(left, right):
        if right - left <= 1:
            return
        inside = np.arange(left + 1, right)
        f = (times[inside] - times[left]) / (times[right] - times[left])
        chord = (1 - f[:, None]) * c[left] + f[:, None] * c[right]
        errors = np.abs(c[inside] - chord).max(axis=1)
        local = int(errors.argmax())
        heapq.heappush(segments, (-errors[local], int(inside[local]), left, right))

    selected = [0, len(times) - 1]
    score(*selected)
    while len(selected) < FRAMES:
        neg_error, best, left, right = heapq.heappop(segments)
        if -neg_error <= tolerance:
            # At the linear threshold, follow the original deterministic midpoint rule.
            return indices[cnn.greedy_chord_keyframe_indices(c, times, n_keyframes=FRAMES)]
        selected.append(best)
        score(left, best)
        score(best, right)
    return indices[np.sort(selected)]


def prepare(domain, workers):
    started = time.perf_counter()
    folder = HERE / "cache"
    with np.load(folder / "index.npz") as index:
        ids, chosen = index["ids"], index["chosen"]
    paths = {int(p.stem[9:]): p for p in (SOURCE / "data" / domain).glob("*/samples/material_*.npz")}
    sample = paths[int(ids[0])]
    with np.load(sample) as a:
        x, t = a["x_cells_nd"], a["tau"]
    shape = (len(ids), 3, FRAMES, len(x))
    field_path = folder / f"cnn_{domain}_fields.npy"
    tmp_path = field_path.with_suffix(".tmp.npy")
    fields = np.lib.format.open_memmap(tmp_path, mode="w+", dtype=np.float32, shape=shape)
    times = np.zeros(shape[:-1], dtype=np.float32)
    masks = np.zeros(shape[:-1], dtype=bool)
    lo = np.empty(shape[:2], dtype=np.float32)
    hi = np.empty_like(lo)

    def process(row):
        with np.load(paths[int(ids[row])], allow_pickle=False) as a:
            data = a["concentration_nd"]
            valid = a["valid_time_mask"]
            for column, j in enumerate(chosen[row]):
                frames = keyframes(data[j], t, valid[j])
                n = len(frames)
                prof = data[j, frames]
                if not np.isfinite(prof).all():
                    raise ValueError(f"Invalid observation for material {ids[row]} current {j}")
                fields[row, column, :n] = prof
                fields[row, column, n:] = 1.
                times[row, column, :n] = t[frames]
                masks[row, column, :n] = True
                lo[row, column], hi[row, column] = prof.min(), prof.max()
                if row == 0:
                    ref, tref, mref = cnn.change_aware_temporal_compression(data[j], t, valid[j], n_keyframes=FRAMES)
                    np.testing.assert_array_equal(fields[row, column], ref)
                    np.testing.assert_array_equal(times[row, column], tref)
                    np.testing.assert_array_equal(masks[row, column], mref)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for count, _ in enumerate(pool.map(process, range(len(ids))), 1):
            if count % 250 == 0:
                print(domain, count, flush=True)
    fields.flush()
    del fields
    os.replace(tmp_path, field_path)
    meta = folder / f"cnn_{domain}.npz"
    np.savez(meta.with_suffix(".tmp.npz"), ids=ids, chosen=chosen, time=times, mask=masks,
             lo=lo, hi=hi, x=x, horizon=t[-1])
    os.replace(meta.with_suffix(".tmp.npz"), meta)
    (folder / f"cnn_{domain}_prep.json").write_text(json.dumps(
        {"seconds": time.perf_counter() - started, "materials": len(ids), "workers": workers}, indent=2))
    print(domain, "prepared", round(time.perf_counter()-started, 1), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("domain", choices=DOMAINS)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    prepare(args.domain, args.workers)
