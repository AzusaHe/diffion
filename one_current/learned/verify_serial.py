"""Reproduce seed-0 predictions case by case and time 150-step serial CPU TTT."""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

from run import MLP, FNO, observed_fields, pde_loss, feature_tensor, write_json, STEPS


def main(protocol):
    d = protocol / "data"
    out = protocol / "learned"
    test = np.flatnonzero(np.load(d / "train_split.npy") == "id_test")[:8]
    with np.load(d / "train.npz") as dataset:
        y = dataset["y"][test]
        currents = dataset["iotas"][test]
    fields = observed_fields(y, currents)
    results = {}
    for arch, build in (("mlp", MLP), ("fno", FNO)):
        path = out / f"{arch}_s0"
        cfg = json.loads((path / "training.json").read_text())
        fitted = json.loads((path / "summary.json").read_text())
        feat, _ = feature_tensor(y, currents, cfg["scales"], "cpu")
        net = build()
        net.load_state_dict(torch.load(path / "best.pt", map_location="cpu", weights_only=True))
        pred, costs = [], []
        # Initialize AdamW and FFT caches before timing the eight independent fits.
        with torch.no_grad():
            net(feat[:1])
        torch.optim.AdamW(net.parameters(), lr=fitted["ttt_lr"])
        for k in range(len(test)):
            start = time.perf_counter()
            individual = copy.deepcopy(net)
            opt = torch.optim.AdamW(individual.parameters(), lr=fitted["ttt_lr"])
            ix = torch.tensor([k])
            for _ in range(STEPS):
                opt.zero_grad(set_to_none=True)
                th = individual(feat[k:k+1]).sigmoid()
                loss = pde_loss(th,fields,ix,fitted["residual_scale"])
                loss.backward()
                opt.step()
            with torch.no_grad():
                pred.append(individual(feat[k:k+1]).sigmoid().numpy()[0])
            costs.append(time.perf_counter()-start)
        saved = np.load(path / "ttt.npz")["theta"][:len(test)]
        errors = np.max(np.abs(np.stack(pred) - saved), axis=1)
        results[arch] = {"n":len(test),"seconds_per_material":costs,"median_s":float(np.median(costs)),
                         "max_abs_theta_difference":float(errors.max()),"per_material_max_abs_theta_difference":errors.tolist()}
    write_json(out / "serial_ttt_check.json", {"device":"CPU, one thread","subset":"test materials 0-7",
                                                "ttt_steps":STEPS,"results":results})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-root", type=Path, required=True)
    torch.set_num_threads(1)
    main(ap.parse_args().run_root.resolve())
