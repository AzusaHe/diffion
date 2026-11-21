"""Test-time training (TTT) on the PDE loss.

    python -m pinn.ttt --arch fno --seed 0          (run seed 0 first: it chooses the learning rate)

Two-stage recipe: stage 1 is the supervised network (surrogate/train.py).  Stage 2, for each test material
separately: start from the stage-1 weights and update ALL network weights for 150 AdamW steps on that
material's PDE loss alone (no labels); the weights are reset to stage 1 before every material.

Learning rate: chosen once per architecture from {1e-5, 1e-4, 1e-3} on 100 validation materials with the
seed-0 network, by the median MSE of ln D after 150 steps (labels are used only for this choice).
Recorded for every test material at steps 0, 5, 10, 25, 50, 100, 150: theta, the PDE loss and the scores.
Output results/stage2/ttt/: select_{arch}.json, {arch}_s{seed}.json.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.flop_counter import FlopCounterMode  # noqa: E402

from analysis.metrics import score_all  # noqa: E402
from surrogate.data import load  # noqa: E402
from surrogate.models import build, input_scales  # noqa: E402
from surrogate.train import RESULTS  # noqa: E402

from .pde_loss import pde_loss, residual_scale  # noqa: E402

STEPS = 150
CHECKPOINTS = (0, 5, 10, 25, 50, 100, 150)
LEARNING_RATES = (1e-5, 1e-4, 1e-3)
N_SELECT = 100
OUT = RESULTS / "ttt"
torch.set_num_threads(1)


def ttt(net0, S, i, lr, scale):
    """Fine-tune a copy of net0 on material i's PDE loss; theta and PDE loss at the checkpoints."""
    net = copy.deepcopy(net0)
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    idx = torch.tensor([i])
    traj = {}
    for k in range(STEPS + 1):
        th = torch.sigmoid(net(S["yv"][idx], S["it"][idx]))
        L = pde_loss(th, S["pde"], idx, scale)
        if k in CHECKPOINTS:
            traj[k] = dict(theta=th[0].detach().tolist(), pde_loss=L.item())
        if k == STEPS:
            return traj
        opt.zero_grad(); L.backward(); opt.step()


def score_trajectories(trajs, S):
    """{step: {metric: [per material]}} including the PDE loss."""
    out = {}
    for k in CHECKPOINTS:
        out[k] = score_all(np.array([t[k]["theta"] for t in trajs]), S["theta"], S["window"])
        out[k]["pde_loss"] = [t[k]["pde_loss"] for t in trajs]
    return out


def subset(S, n):
    return {k: ({kk: vv[:n] for kk, vv in v.items()} if isinstance(v, dict) else v[:n]) for k, v in S.items()}


def main(arch, seed):
    S = load()
    scale = residual_scale(S["train"]["pde"])
    net = build(arch, input_scales(S["train"]["yv"], S["train"]["it"]), seed)
    net.load_state_dict(torch.load(RESULTS / f"{arch}_s{seed}" / "model.pt"))
    OUT.mkdir(parents=True, exist_ok=True)

    sel_file = OUT / f"select_{arch}.json"
    if not sel_file.exists():
        assert seed == 0, "run seed 0 first: the learning rate is chosen with the seed-0 network"
        va = subset(S["val"], N_SELECT)
        grid = {}
        for lr in LEARNING_RATES:
            sc = score_trajectories([ttt(net, va, i, lr, scale) for i in range(N_SELECT)], va)
            grid[str(lr)] = {str(k): float(np.median(sc[k]["mse_lnD"])) for k in CHECKPOINTS}
            print("select", arch, lr, grid[str(lr)], flush=True)
        lr_best = min(LEARNING_RATES, key=lambda lr: grid[str(lr)][str(STEPS)])
        sel_file.write_text(json.dumps(dict(grid=grid, lr=lr_best), indent=1))
    lr = json.loads(sel_file.read_text())["lr"]

    te = S["id_test"]
    idx = torch.tensor([0])
    with FlopCounterMode(display=False) as fc:
        pde_loss(torch.sigmoid(copy.deepcopy(net)(te["yv"][idx], te["it"][idx])), te["pde"], idx, scale).backward()
    t0 = time.perf_counter()
    trajs = [ttt(net, te, i, lr, scale) for i in range(len(te["yv"]))]
    wall = (time.perf_counter() - t0) / len(trajs)
    res = dict(arch=arch, seed=seed, lr=lr, steps=STEPS, flops_step=fc.get_total_flops(), wall_s_per_material=wall,
               checkpoints=list(CHECKPOINTS), scores=score_trajectories(trajs, te))
    (OUT / f"{arch}_s{seed}.json").write_text(json.dumps(res))
    print(f"{wall:.2f} s/material", {m: float(np.median(res["scores"][STEPS][m])) for m in ("mse_lnD", "mse_tplus", "D_relL2_pct")})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--arch", choices=("mlp", "fno"), required=True)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    main(a.arch, a.seed)
