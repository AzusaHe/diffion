"""Supervised training of the MLP or FNO on the coefficient loss (data loss).

    python -m surrogate.train --arch fno --seed 0
    python -m surrogate.train --arch fno --timing        (standalone time per step, written to timing_{arch}.json)

Data loss:   L_data = mean over the batch and the 8 coefficients of (sigmoid(z) - theta)^2.
Training:    AdamW (weight decay 0.01), lr 1e-3 with cosine decay to 0 per step, batch 64, 100 epochs
             (16,200 steps), one CPU thread; checkpoint = best validation coefficient MSE.
Logged per epoch: training and validation coefficient MSE, validation PDE loss (diagnostic only; it is
not part of the training loss), wall time of the training loop, FLOPs.
Output results/stage2/{arch}_s{seed}/: model.pt, result.json (history and per-material scores on the
300 test materials).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.flop_counter import FlopCounterMode  # noqa: E402

from analysis.metrics import score_all  # noqa: E402
from pinn.pde_loss import pde_loss, residual_scale  # noqa: E402

from .data import load  # noqa: E402
from .models import build, input_scales, n_params, predict  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results" / "stage2"
EPOCHS, LR, BATCH = 100, 1e-3, 64
torch.set_num_threads(1)


@torch.no_grad()
def evaluate(model, S, scale, theta=None):
    th = predict(model, S) if theta is None else theta
    idx = torch.arange(len(th))
    return dict(coef_mse=float(((th - S["tht"]) ** 2).mean()), pde_loss=float(pde_loss(th, S["pde"], idx, scale))), th


def main(arch, seed, timing=False):
    S = load()
    tr, va, te = S["train"], S["val"], S["id_test"]
    scale = residual_scale(tr["pde"])
    model = build(arch, input_scales(tr["yv"], tr["it"]), seed)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    n = len(tr["yv"])
    steps = EPOCHS * math.ceil(n / BATCH)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: 0.5 * (1 + math.cos(math.pi * min(k, steps) / steps)))
    gen = torch.Generator().manual_seed(seed + 1)

    def data_loss(idx):
        return ((torch.sigmoid(model(tr["yv"][idx], tr["it"][idx])) - tr["tht"][idx]) ** 2).mean()

    with FlopCounterMode(display=False) as fc:
        data_loss(torch.arange(BATCH)).backward()
    flops_step = fc.get_total_flops(); opt.zero_grad()
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model(tr["yv"][:1], tr["it"][:1])
    flops_infer = fc.get_total_flops()
    RESULTS.mkdir(parents=True, exist_ok=True)

    if timing:
        ts = []
        for _ in range(220):
            idx = torch.randint(0, n, (BATCH,), generator=gen)
            t0 = time.perf_counter(); L = data_loss(idx); opt.zero_grad(); L.backward(); opt.step()
            ts.append(time.perf_counter() - t0)
        res = dict(arch=arch, step_s_median=float(np.median(ts[20:])), step_s_mean=float(np.mean(ts[20:])),
                   flops_step=flops_step, flops_infer=flops_infer, n_params=n_params(model))
        (RESULTS / f"timing_{arch}.json").write_text(json.dumps(res, indent=1))
        print(res)
        return

    run = RESULTS / f"{arch}_s{seed}"
    run.mkdir(exist_ok=True)
    d0, _ = evaluate(model, va, scale)
    hist = [dict(epoch=0, step=0, train_wall_s=0.0, flops=0, val_coef_mse=d0["coef_mse"], val_pde_loss=d0["pde_loss"])]
    best, wall, step = np.inf, 0.0, 0
    for ep in range(1, EPOCHS + 1):
        perm = torch.randperm(n, generator=gen)
        total, t0 = 0.0, time.perf_counter()
        for s in range(0, n, BATCH):
            idx = perm[s:s + BATCH]
            L = data_loss(idx)
            opt.zero_grad(); L.backward(); opt.step(); sched.step(); step += 1
            total += float(L) * len(idx)
        wall += time.perf_counter() - t0
        d, _ = evaluate(model, va, scale)
        hist.append(dict(epoch=ep, step=step, train_wall_s=wall, flops=step * flops_step, train_coef_mse=total / n,
                         val_coef_mse=d["coef_mse"], val_pde_loss=d["pde_loss"]))
        if d["coef_mse"] < best:
            best = d["coef_mse"]
            torch.save(model.state_dict(), run / "model.pt")
        print(json.dumps(hist[-1]), flush=True)

    model.load_state_dict(torch.load(run / "model.pt"))
    d, th = evaluate(model, te, scale)
    t0 = time.perf_counter(); predict(model, te); infer = (time.perf_counter() - t0) / len(te["yv"])
    truth, _ = evaluate(model, te, scale, theta=te["tht"])
    res = dict(arch=arch, seed=seed, n_params=n_params(model), epochs=EPOCHS, steps=step, train_wall_s=wall,
               flops_step=flops_step, flops_train=step * flops_step, flops_infer=flops_infer, pde_scale=scale,
               best_val_coef_mse=best, history=hist,
               test=dict(n=len(th), infer_s_per_material=infer, coef_mse=d["coef_mse"], pde_loss=d["pde_loss"],
                         truth_pde_loss=truth["pde_loss"], theta=th.tolist(),
                         **score_all(th.double().numpy(), te["theta"], te["window"])))
    (run / "result.json").write_text(json.dumps(res))
    print({k: float(np.median(res["test"][k])) for k in ("mse_lnD", "mse_tplus", "D_relL2_pct")})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--arch", choices=("mlp", "fno"), required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timing", action="store_true")
    a = p.parse_args()
    main(a.arch, a.seed, a.timing)
