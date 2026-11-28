"""Build a paired one-current dataset on the original material IDs and splits.

Run from the diffion-ms directory with python -m one_current.build_data --protocol random.
Only the selected current is simulated. Valid time prefixes and scoring windows are
computed from that trajectory. Original material selection is retained for pairing.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from solver import dataset as coarse
from solver import dataset_fine as fine


OUT = Path(__file__).resolve().parent / "data"


def atomic_save(path, **data):
    temp = path.with_suffix(".incomplete")
    with temp.open("wb") as handle:
        np.savez_compressed(handle, **data)
    os.replace(temp, path)


def simulate_batches(fn, theta, currents, batch=64):
    fields, validity = [], []
    for start in range(0, len(theta), batch):
        end = min(start + batch, len(theta))
        size = end - start
        pad = np.minimum(np.arange(start, start + batch), end - 1)
        values, ok = fn(theta[pad], currents[pad])
        fields.append(np.asarray(values)[:size])
        validity.append(np.asarray(ok)[:size])
    return np.concatenate(fields), np.concatenate(validity)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=["fixed", "low", "middle", "high", "random"], default="random")
    parser.add_argument("--seed", type=int, default=236)
    parser.add_argument("--current", type=float, default=1.0)
    args = parser.parse_args()
    out = OUT.parent / "random" / "data" if args.protocol == "random" else OUT
    out.mkdir(exist_ok=True, parents=True)
    if (out / "train.npz").exists():
        raise FileExistsError("Data already present. Use a separate output folder for another protocol.")
    base = np.load(coarse.DATA / "train.npz")
    split = np.load(coarse.DATA / "train_split.npy")
    theta, ids = base["theta"], base["index"]
    if args.protocol == "fixed":
        currents = np.full((len(ids), 1), args.current)
    else:
        if args.protocol == "random":
            rank = np.asarray([np.random.default_rng([args.seed,int(i)]).integers(3) for i in ids])
        else:
            rank = np.full(len(ids),{"low": 0, "middle": 1, "high": 2}[args.protocol])
        currents = base["iotas"][np.arange(len(ids)),rank,None]
    print("Simulating", len(ids), args.protocol, flush=True)
    y, ok = simulate_batches(coarse.simulate_batch, theta, currents)
    ok = ok[:, 0]
    K = np.where(ok.all(axis=1), ok.shape[1], np.argmin(ok, axis=1))
    if np.any(K < 3):
        raise ValueError(f"{(K < 3).sum()} paired materials have fewer than 3 valid frames")
    y = np.where(np.arange(coarse.K_MAX)[None, None, :, None] < K[:, None, None, None], y, np.nan)
    assert np.isfinite(y).sum() == coarse.N_BINS * K.sum()
    window = np.stack([np.nanmin(y, axis=(1,2,3)), np.nanmax(y, axis=(1,2,3))], axis=1)
    meta = dict(protocol=args.protocol, current_range=[float(currents.min()),float(currents.max())],
                number_of_currents=1, split_counts={s:int((split==s).sum()) for s in np.unique(split)},
                frames_different_from_original=int(np.count_nonzero(K!=base["K"])))
    if args.protocol != "fixed":
        # Independent control against the original measurement wherever both are defined.
        overlap_error = float(np.nanmax(np.abs(y - base["y"][np.arange(len(ids)),rank][:,None])))
        assert overlap_error < 1e-9, overlap_error
        meta["max_original_overlap_difference"] = overlap_error
    if args.protocol == "random":
        meta["seed"] = args.seed
        meta["selection"] = "Uniform draw among the original three current ranks, seeded by [seed, original material ID]"
        meta["rank_counts"] = {s:np.bincount(rank[split==s],minlength=3).tolist() for s in np.unique(split)}
        np.save(out / "current_rank.npy",rank)
    (out / "protocol.json").write_text(json.dumps(meta, indent=2))
    np.save(out / "train_split.npy", split)
    atomic_save(out / "train.npz", theta=theta, index=ids, iotas=currents, y=y, window=window, K=K)
    print("Coarse observations ready", meta, flush=True)
    ix = np.flatnonzero(split == "id_test")[:300]
    yf, _ = simulate_batches(fine.simulate_batch, theta[ix], currents[ix])
    yf = np.where(np.arange(len(fine.T_FINE))[None,None,:,None] < fine.SUB*K[ix,None,None,None], yf, np.nan)
    yback = yf[:,:,fine.SUB-1::fine.SUB] @ coarse.W_BINS.T
    np.testing.assert_allclose(yback, y[ix], atol=1e-9, rtol=0, equal_nan=True)
    atomic_save(out / "fine_test.npz", theta=theta[ix], index=ids[ix], iotas=currents[ix], y=yf, K=K[ix], t=fine.T_FINE)
    print("Fine observations ready", yf.shape, flush=True)


if __name__ == "__main__":
    main()
