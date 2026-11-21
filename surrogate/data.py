"""Data for the networks: splits of data/train.npz, the electrolyte velocity v(x, t) from c alone, and inputs.

v from c (nothing fitted; D and t+ are not needed):
    J_{1/2} = iota,  J_{j+1/2} = J_{j-1/2} - h dc_j/dt                   (conservation, wall flux iota)
    v_{1/2} = 0,     v_{j+1/2} - v_{j-1/2} = -V(c_j) (q_{j+1/2} - q_{j-1/2}),   q = J - c_f v   (volume constraint)
solved face by face; dc/dt at a frame is the mean of the differences to the neighbouring valid frames.
v at the cell centres is the average of its two faces.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from pinn.pde_loss import H, TAU, observed_fields
from solver.properties import RHO_COEFF, alpha_V

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
N_TEST = 300


def frame_dcdt(c, valid):
    fwd = (c[:, :, 1:] - c[:, :, :-1]) / np.diff(TAU)[None, None, :, None]
    fv = (valid[:, 1:] & valid[:, :-1])[:, None, :, None].astype(float)
    num, den = np.zeros_like(c), np.zeros_like(c)
    num[:, :, 1:] += fwd * fv; den[:, :, 1:] += fv
    num[:, :, :-1] += fwd * fv; den[:, :, :-1] += fv
    return num / np.maximum(den, 1.0)


def velocity_from_c(y, iotas):
    """v (n, 3, 13, 20) at the cell centres from the measured c alone; NaN on invalid frames."""
    valid = np.isfinite(y[:, 0, :, 0])
    c = np.where(np.isfinite(y), y, 1.0)
    V = np.asarray(alpha_V(c, RHO_COEFF)[1])
    cf = np.concatenate([c[..., :1], 0.5 * (c[..., 1:] + c[..., :-1]), c[..., -1:]], -1)            # faces 0..20
    J = np.asarray(iotas, float)[:, :, None, None] - H * np.concatenate(
        [np.zeros_like(c[..., :1]), np.cumsum(frame_dcdt(c, valid), -1)], -1)
    v, q = np.zeros_like(J), J.copy()
    for j in range(20):
        v[..., j + 1] = (v[..., j] - V[..., j] * (J[..., j + 1] - q[..., j])) / (1.0 - V[..., j] * cf[..., j + 1])
        q[..., j + 1] = J[..., j + 1] - cf[..., j + 1] * v[..., j + 1]
    v_cell = 0.5 * (v[..., 1:] + v[..., :-1])
    v_cell[~np.broadcast_to(valid[:, None, :, None], v_cell.shape)] = np.nan
    return v_cell


def load(names=("train", "val", "id_test")):
    """Each split: numpy y, iotas, theta, window; torch yv = [c, v] (B, 6, 13, 20), it, tht; pde fields.
    id_test is the first N_TEST test materials (the same ones for every model)."""
    d = np.load(DATA / "train.npz")
    split = np.load(DATA / "train_split.npy")
    sets = {}
    for name in names:
        s = np.flatnonzero(split == name)
        if name == "id_test":
            s = s[:N_TEST]
        S = dict(y=d["y"][s], iotas=d["iotas"][s], theta=d["theta"][s], window=d["window"][s])
        S["pde"] = observed_fields(S["y"], S["iotas"])
        S["yv"] = torch.tensor(np.concatenate([S["y"], velocity_from_c(S["y"], S["iotas"])], 1), dtype=torch.float32)
        S["it"] = torch.tensor(S["iotas"], dtype=torch.float32)
        S["tht"] = torch.tensor(S["theta"], dtype=torch.float32)
        sets[name] = S
    return sets
