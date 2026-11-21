"""Per-material errors of predicted D(c) and t+(c), on 301 concentrations spanning the material's observed window.

    mse_lnD      mean (ln D_pred - ln D_true)^2
    mse_tplus    mean (t+_pred - t+_true)^2
    D_relL2_pct  100 ||D_pred - D_true|| / ||D_true||
    tplus_rmse   sqrt(mse_tplus)
    group_rmse   RMS of ln[(1 - t+)/D]_pred - ln[(1 - t+)/D]_true   (the combination the data constrain best)
"""
from __future__ import annotations

import numpy as np
import torch

from pinn.material import d_hat, t_plus


def curves(theta, c):
    th = torch.as_tensor(np.asarray(theta, float))
    c = torch.as_tensor(c)
    return d_hat(c, th[:4]).numpy(), t_plus(c, th[4:]).numpy()


def score(theta, truth, window):
    c = np.linspace(window[0], window[1], 301)
    d, t = curves(theta, c)
    dt, tt = curves(truth, c)
    g = np.log(np.clip(1 - t, 1e-3, None) / d)
    gt = np.log(np.clip(1 - tt, 1e-3, None) / dt)
    return dict(D_relL2_pct=float(np.linalg.norm(d - dt) / np.linalg.norm(dt) * 100),
                tplus_rmse=float(np.sqrt(np.mean((t - tt) ** 2))), group_rmse=float(np.sqrt(np.mean((g - gt) ** 2))),
                mse_lnD=float(np.mean((np.log(d) - np.log(dt)) ** 2)), mse_tplus=float(np.mean((t - tt) ** 2)))


def score_all(thetas, truths, windows):
    """{metric: [per material]}."""
    sc = [score(a, b, w) for a, b, w in zip(thetas, truths, windows)]
    return {k: [s[k] for s in sc] for k in sc[0]}
