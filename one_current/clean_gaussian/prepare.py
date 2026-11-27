"""Build paired material splits and FV-averaged inputs from the two source folders."""
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[2] / "latest_synthetic_ml" / "data"
CACHE = HERE / "cache"
DOMAINS = ("clean", "gaussian_noisy")
USABLE = ("steady", "max_time", "concentration_limited")


def paths(domain):
    files = sorted((DATA / domain).glob("*/samples/material_*.npz"))
    result = {int(p.stem[9:]): p for p in files}
    if not files or len(result) != len(files):
        raise ValueError(f"Missing or duplicate sample IDs in {domain}")
    return result


def save(name, **arrays):
    destination = CACHE / name
    temp = destination.with_suffix(".tmp.npz")
    np.savez(temp, **arrays)
    temp.replace(destination)


def build():
    start = time.perf_counter()
    CACHE.mkdir(exist_ok=True, parents=True)
    files = {domain: paths(domain) for domain in DOMAINS}
    assert files["clean"].keys() == files["gaussian_noisy"].keys()
    def metadata(mid):
        with np.load(files["clean"][mid]) as a, np.load(files["gaussian_noisy"][mid]) as b:
            for key in ("material_sample_id", "status", "valid_time_mask", "D_coefficients_01",
                        "t_plus_coefficients_01", "iota", "tau", "x_cells_nd", "run_fingerprint"):
                np.testing.assert_array_equal(a[key], b[key])
            assert mid == a["material_sample_id"].item()
            assert "gaussian" in str(b["noise_model"]).lower() and b["noise_sigma_ref_mM"].item() == 2.2
            valid = a["valid_time_mask"]
            assert not np.any(valid[:,1:] & ~valid[:,:-1]), "Expected contiguous observations before censoring"
            counts = valid.sum(1)
            pool = np.flatnonzero(np.isin(a["status"], USABLE) & (counts >= 2))
            if len(pool) < 3:
                return None
            rng = np.random.default_rng([236, mid])
            one = rng.choice(pool)
            j = np.r_[one, np.sort(rng.choice(pool[pool != one], 2, replace=False))]
            return (mid, j, a["iota"][j], np.r_[a["D_coefficients_01"], a["t_plus_coefficients_01"]],
                    counts[j], a["tau"])
    with ThreadPoolExecutor(8) as ex:
        rows = [r for r in ex.map(metadata, sorted(files["clean"])) if r is not None]
    ids, chosen, iota, theta, counts, axes = map(np.array, zip(*rows))
    def ranked(sequence, prefix):
        return sorted(sequence, key=lambda mid: hashlib.blake2b(f"{prefix}:{mid}".encode(), digest_size=16).digest())
    order = ranked(ids, "three-current-split-80-20-v1:242")
    n_training_pool = int(np.floor(0.8*len(ids)+0.5))
    test = set(order[n_training_pool:])
    val = set(ranked(order[:n_training_pool], "clean-gaussian-validation-v1:242")[:round(n_training_pool/10)])
    split = np.array(["test" if mid in test else "val" if mid in val else "train" for mid in ids])
    common = dict(ids=ids, chosen=chosen, iota=iota, theta=theta, split=split)
    save("index.npz", **common)
    print("Index ready", dict(zip(*np.unique(split, return_counts=True))), flush=True)
    index_s = time.perf_counter()-start
    frame = np.zeros((len(ids), 3, 13), int)
    mask = np.zeros_like(frame, bool)
    for row, triple in enumerate(counts):
        for j, n in enumerate(triple):
            m = min(n, 13)
            frame[row,j,:m] = np.rint(np.linspace(0, n-1, m)).astype(int)
            mask[row,j,:m] = True
    tau = np.take_along_axis(axes[:,None,:], frame, axis=2)
    domain_arrays = {domain: np.full((*mask.shape, 20), np.nan, np.float32) for domain in DOMAINS}
    noise_ss = np.zeros(len(ids))
    noise_n = np.zeros(len(ids), int)
    def bin_one(row):
        mid = ids[row]
        selected = []
        for domain in DOMAINS:
            with np.load(files[domain][mid]) as a:
                z = a["concentration_nd"][chosen[row,:,None],frame[row]]
            selected.append(z[mask[row]].astype(np.float64))
            assert np.isfinite(selected[-1]).all()
            domain_arrays[domain][row][mask[row]] = selected[-1].reshape(-1,20,58).mean(axis=-1)
        delta = selected[0]-selected[1]
        noise_ss[row] = np.vdot(delta.ravel(), delta.ravel())
        noise_n[row] = delta.size
    with ThreadPoolExecutor(6) as ex:
        for n,_ in enumerate(ex.map(bin_one, range(len(ids))), 1):
            if n%400==0:
                print("Binned", n, flush=True)
    save("binned.npz", **common, **domain_arrays, mask=mask, tau=tau, frame=frame)
    summary = dict(source_materials=len(files["clean"]), eligible=len(ids), index_s=index_s,
                   total_prepare_s=time.perf_counter()-start, selected_native_noise_rms=float(np.sqrt(noise_ss.sum()/noise_n.sum())),
                   split={str(s):int(n) for s,n in zip(*np.unique(split, return_counts=True))})
    (CACHE/"preparation.json").write_text(json.dumps(summary,indent=2))
    print(summary, flush=True)


if __name__ == "__main__":
    build()
