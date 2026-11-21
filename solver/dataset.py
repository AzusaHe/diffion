"""Synthetic materials: sample, simulate, split.

    python -m solver.dataset            (16,000 materials, about 20 min on one CPU core)

Sampling (one RNG per material: numpy default_rng([3_000_000, index])):
  theta_D ~ U(0,1)^4, theta_t ~ U(0,1)^4 (unit cubic Bernstein coefficients, drawn in that order);
  8 stratified currents iota_k ~ U(5k/8, 5(k+1)/8), k = 0..7; each material uses k = 0, 2, 4 (three currents).
Simulation: solver at N = 40 cells, Crank-Nicolson, 16 steps per frame interval, uniform c0 = 1.
Observation: 13 frames at 240 s + 960 s k, 20 equal bins (conservative average of the 40 cells).
A frame is kept when all three currents are valid (Newton converged, c in [1e-4, 2]) at every step up to it;
the kept frames form a prefix of length K, and frames beyond K are NaN.  Materials with K < 3 are dropped.
Split: the extreme 10 % of materials in mean theta_D or mean theta_t (below the 5th or above the 95th
percentile of either) form 'ood_params' and are held out of training; the rest split by index into
'train' (85 %), 'val' (5 %) and 'id_test' (10 %).

Output data/train.npz: theta (n, 8), iotas (n, 3), index (n,), K (n,), y (n, 3, 13, 20), window (n, 2) =
(min, max) of the observed c; data/train_split.npy: split name per material.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .integrate import Protocol, SolverOptions, simulate
from .fv import uniform_grid
from .properties import RHO_COEFF, BernsteinMaterial

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
T_REF = 33640.0
TAU = (240.0 + 960.0 * np.arange(13)) / T_REF
N_CELLS, N_BINS, K_MAX = 40, 20, 13
SEED_BASE = 3_000_000
CURRENT_STRATA = [0, 2, 4]
N_MATERIALS = 16000


def rebin_matrix(old_faces, new_faces):
    """Conservative averaging from one 1D grid onto another."""
    overlap = np.maximum(0, np.minimum(new_faces[1:, None], old_faces[None, 1:])
                         - np.maximum(new_faces[:-1, None], old_faces[None, :-1]))
    return overlap / np.diff(new_faces)[:, None]


W_BINS = rebin_matrix(np.linspace(0, 1, N_CELLS + 1), np.linspace(0, 1, N_BINS + 1))


def sample(index: int):
    rng = np.random.default_rng([SEED_BASE, index])
    theta = np.r_[rng.uniform(0, 1, 4), rng.uniform(0, 1, 4)]
    iotas = 5.0 * (np.arange(8) + rng.uniform(0, 1, 8)) / 8.0
    return theta, iotas[CURRENT_STRATA]


@jax.jit
def simulate_batch(theta, iotas):
    """theta (B, 8), iotas (B, 3) -> binned c (B, 3, 13, 20) and frame validity (B, 3, 13)."""
    grid = uniform_grid(N_CELLS)
    model = BernsteinMaterial(jnp.asarray(RHO_COEFF))
    W = jnp.asarray(W_BINS)

    def one(th, io):
        c, info = simulate(th, Protocol(model=model, c0=jnp.ones(N_CELLS), t_obs=jnp.asarray(TAU), iota=io),
                           grid, SolverOptions())
        return c @ W.T, info["valid_obs"]
    return jax.vmap(lambda th, ios: jax.vmap(lambda io: one(th, io))(ios))(theta, iotas)


def generate(indices, batch: int = 64):
    theta, iotas = map(np.asarray, zip(*[sample(int(i)) for i in indices]))
    n = len(theta)
    ys, oks = [], []
    t0 = time.perf_counter()
    for s in range(0, n, batch):
        m = min(batch, n - s)
        pad = lambda a: np.concatenate([a[s:s + m], np.repeat(a[s + m - 1:s + m], batch - m, 0)])  # noqa: E731
        y, ok = simulate_batch(jnp.asarray(pad(theta)), jnp.asarray(pad(iotas)))
        ys.append(np.asarray(y)[:m]); oks.append(np.asarray(ok)[:m])
        print(f"{s + m}/{n}  {time.perf_counter() - t0:.0f} s", flush=True)
    y, ok = np.concatenate(ys), np.concatenate(oks).all(axis=1)          # frame valid for all 3 currents
    K = np.where(ok.all(1), K_MAX, np.argmin(ok, axis=1))
    keep = (K >= 3) & np.array([np.isfinite(y[i, :, :K[i]]).all() for i in range(n)])
    y = np.where(np.arange(K_MAX)[None, None, :, None] < K[:, None, None, None], y, np.nan)
    window = np.stack([np.nanmin(y, axis=(1, 2, 3)), np.nanmax(y, axis=(1, 2, 3))], 1)
    return dict(theta=theta[keep], iotas=iotas[keep], index=np.asarray(indices)[keep], K=K[keep], y=y[keep],
                window=window[keep])


def split(theta, index):
    s_D, s_t = theta[:, :4].mean(1), theta[:, 4:].mean(1)
    (d_lo, d_hi), (t_lo, t_hi) = np.quantile(s_D, [0.05, 0.95]), np.quantile(s_t, [0.05, 0.95])
    ood = (s_D < d_lo) | (s_D > d_hi) | (s_t < t_lo) | (s_t > t_hi)
    r = (index * 0.6180339887498949) % 1.0
    return np.where(ood, "ood_params", np.where(r < 0.85, "train", np.where(r < 0.90, "val", "id_test")))


def main():
    DATA.mkdir(exist_ok=True)
    d = generate(np.arange(N_MATERIALS))
    sp = split(d["theta"], d["index"])
    np.savez_compressed(DATA / "train.npz", **d)
    np.save(DATA / "train_split.npy", sp)
    print(json.dumps({s: int((sp == s).sum()) for s in np.unique(sp)}))


if __name__ == "__main__":
    main()
