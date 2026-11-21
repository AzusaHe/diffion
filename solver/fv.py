"""Conservative cell-centred finite-volume fluxes.

Interior faces: harmonic-mean conductance of D alpha, distance-weighted t+, centred face concentration.
Left wall:  q_0 = J_0 = iota, v_0 = 0.   Interior v: v_{i+1} = v_i - V_i (q_{i+1} - q_i).
Right wall: J_N = iota; q_N, v_N from {v_N = v_{N-1} - V_N (q_N - q_{N-1}), J_N = q_N + c_N v_N}.
Cell update: dc_i/dt = -(J_{i+1/2} - J_{i-1/2}) / h_i.
"""
from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np


class Grid(eqx.Module):
    faces: jnp.ndarray          # (N+1,) from 0 to 1

    @property
    def n(self) -> int:
        return self.faces.shape[0] - 1

    @property
    def widths(self):
        return jnp.diff(self.faces)

    @property
    def centers(self):
        return 0.5 * (self.faces[:-1] + self.faces[1:])


def uniform_grid(n_cells: int) -> Grid:
    return Grid(jnp.asarray(np.linspace(0.0, 1.0, int(n_cells) + 1)))


def face_weights(grid: Grid):
    """Distance weights of the left / right cell centre at the interior faces, and the two distances."""
    xf, xc = grid.faces, grid.centers
    dl = xf[1:-1] - xc[:-1]
    dr = xc[1:] - xf[1:-1]
    return dr / (dl + dr), dl / (dl + dr), dl, dr


def interior_q(c, iota, props, grid: Grid):
    """q at the interior faces 1..N-1 and the centred face concentration there."""
    dal, tp, _ = props
    wl, wr, dl, dr = face_weights(grid)
    cond = 1.0 / (dl / dal[:-1] + dr / dal[1:])
    q = -cond * (c[1:] - c[:-1]) + iota * (wl * tp[:-1] + wr * tp[1:])
    return q, wl * c[:-1] + wr * c[1:]


def face_fields(c, iota, props, grid: Grid):
    """(J, v, q) at the N+1 faces for cell state c; props = (D alpha, t+, V) at the cell centres."""
    V = props[2]
    q_int, c_face = interior_q(c, iota, props, grid)
    q_lo = jnp.concatenate([jnp.atleast_1d(iota), q_int])                     # faces 0..N-1
    v_int = jnp.cumsum(-V[:-1] * (q_lo[1:] - q_lo[:-1]))                     # faces 1..N-1
    cR, VR = c[-1], V[-1]
    qN = (iota - cR * v_int[-1] - cR * VR * q_int[-1]) / (1.0 - cR * VR)
    vN = v_int[-1] - VR * (qN - q_int[-1])
    J = jnp.concatenate([jnp.atleast_1d(iota), q_int + c_face * v_int, jnp.atleast_1d(iota)])
    v = jnp.concatenate([jnp.zeros(1, c.dtype), v_int, jnp.atleast_1d(vN)])
    q = jnp.concatenate([q_lo, jnp.atleast_1d(qN)])
    return J, v, q


def rhs(c, iota, props, grid: Grid):
    """dc/dt = -div J."""
    J, _, _ = face_fields(c, iota, props, grid)
    return -(J[1:] - J[:-1]) / grid.widths
