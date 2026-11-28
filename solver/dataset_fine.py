"""The id_test materials of data/train.npz on the generator's native grid, for methods that can use every point.

    python -m solver.dataset_fine                 (300 materials, about a minute)

Same materials (theta, currents), same solver, cells (40) and time steps (16 Crank-Nicolson steps per frame
interval, same Rannacher start) as solver/dataset.py, but c is saved in all 40 cells at every 2nd solver step:
8 times per frame interval, 104 times in all, instead of 20 bin averages at 13 frames.  The 13 original frames
are times 8k + 7; binned with W_BINS they reproduce data/train.npz to round-off (checked on build).
Times past the material's last valid frame (K) are NaN, as in train.npz.

Output data/fine_test.npz: index (n,), theta (n, 8), iotas (n, 3), K (n,), t (104,), y (n, 3, 104, 40).
"""
from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np

from .dataset import DATA, K_MAX, N_CELLS, TAU, W_BINS
from .fv import uniform_grid
from .integrate import Protocol, SolverOptions, simulate
from .properties import RHO_COEFF, BernsteinMaterial

SUB = 8                                                   # outputs per frame interval (every 2nd of the 16 steps)
T_FINE = np.concatenate([lo + (hi - lo) * np.arange(1, SUB + 1) / SUB for lo, hi in zip(np.r_[0.0, TAU[:-1]], TAU)])
OPTS = SolverOptions(steps_per_interval=16 // SUB)       # same step sizes as the generator
N_TEST = 300


@jax.jit
def simulate_batch(theta, iotas):
    grid = uniform_grid(N_CELLS)
    model = BernsteinMaterial(jnp.asarray(RHO_COEFF))

    def one(th, io):
        c, info = simulate(th, Protocol(model=model, c0=jnp.ones(N_CELLS), t_obs=jnp.asarray(T_FINE), iota=io), grid, OPTS)
        return c, info["valid_obs"]
    return jax.vmap(lambda th, ios: jax.vmap(lambda io: one(th, io))(ios))(theta, iotas)


def main(n=N_TEST, batch=50):
    d = np.load(DATA / "train.npz")
    split = np.load(DATA / "train_split.npy")
    s = np.flatnonzero(split == "id_test")[:n]
    theta, iotas, K, y13 = d["theta"][s], d["iotas"][s], d["K"][s], d["y"][s]
    ys = []
    for b in range(0, len(s), batch):
        c, _ = simulate_batch(jnp.asarray(theta[b:b + batch]), jnp.asarray(iotas[b:b + batch]))
        ys.append(np.asarray(c))
    y = np.concatenate(ys)                                          # (n, 3, 104, 40)
    keep_t = np.arange(len(T_FINE))[None, :] < (SUB * K)[:, None]  # up to and including frame K - 1
    y = np.where(keep_t[:, None, :, None], y, np.nan)
    frames = y[:, :, SUB - 1::SUB, :] @ W_BINS.T                    # back to 13 frames x 20 bins
    err = np.nanmax(np.abs(frames - y13))
    assert np.array_equal(np.isnan(frames), np.isnan(y13)), "valid frames differ from train.npz"
    assert err < 1e-9, f"native-grid data do not reproduce train.npz (max diff {err:.2e})"
    np.savez_compressed(DATA / "fine_test.npz", index=d["index"][s], theta=theta, iotas=iotas, K=K, t=T_FINE, y=y)
    print(json.dumps(dict(n=len(s), times=len(T_FINE), cells=N_CELLS, max_diff_vs_train_npz=float(err))))


if __name__ == "__main__":
    main()
