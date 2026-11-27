"""Frozen inference on all held-out in-distribution and parameter-tail materials."""
from pathlib import Path
import argparse
import json

import numpy as np
import torch

from run import FNO, MLP, feature_tensor, predict, score_all, write_json


def main(protocol, architectures):
    learned = protocol / "learned"
    data = protocol / "data"
    split = np.load(data / "train_split.npy")
    with np.load(data / "train.npz") as source:
        y, currents, theta, window, indices = (source[k] for k in ("y", "iotas", "theta", "window", "index"))
    expected_scales = json.loads((learned / "mlp_s0" / "training.json").read_text())["scales"]
    result = {}
    for name in ("id_test", "ood_params"):
        ix = np.flatnonzero(split == name)
        feats, _ = feature_tensor(y[ix], currents[ix], expected_scales, "cpu")
        split_results = {}
        for arch, build in (("mlp", MLP), ("fno", FNO)):
            if arch not in architectures:
                continue
            seed_stats = []
            for seed in range(3):
                path = learned / f"{arch}_s{seed}"
                config = json.loads((path / "training.json").read_text())
                if not np.array_equal(config["scales"], expected_scales):
                    raise ValueError("Checkpoint normalization differs from the training split")
                net = build()
                net.load_state_dict(torch.load(path / "best.pt", weights_only=True, map_location="cpu"))
                est = predict(net.eval(), feats).numpy()
                if name == "id_test":
                    with np.load(path / "supervised.npz") as reference:
                        if not np.array_equal(indices[ix][:300], reference["index"]):
                            raise ValueError("Held-out material identifiers differ")
                        err = float(np.max(np.abs(est[:300] - reference["theta"])))
                    if err > 1e-4:
                        raise ValueError(f"Saved supervised predictions differ by {err}")
                else:
                    err = None
                scores = score_all(est, theta[ix], window[ix])
                seed_stats.append({
                    "seed": seed, "median": {k:float(np.median(v)) for k,v in scores.items()},
                    "reproduction_max_abs_first300": err,
                })
            split_results[arch] = {
                "seeds": seed_stats, "mean_of_seed_medians": {
                    key: float(np.mean([row["median"][key] for row in seed_stats])) for key in seed_stats[0]["median"]
                }
            }
        result[name] = {"n": len(ix), "results": split_results}
    fname = "frozen_splits.json" if set(architectures)=={"mlp","fno"} else f"frozen_splits_{'_'.join(architectures)}.json"
    write_json(learned / fname, {
        "scoring_support": "observed window of each material at its one assigned current",
        "inference": "frozen supervised checkpoints before TTT", "splits":result,
    })


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root",type=Path,required=True)
    parser.add_argument("--arch",nargs="+",choices=["mlp","fno"],default=["mlp","fno"])
    args=parser.parse_args()
    torch.set_num_threads(1)
    main(args.run_root.resolve(), args.arch)
