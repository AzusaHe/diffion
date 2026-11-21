"""PDE loss: squared residual of salt conservation, discretised on the measurement grid.

    dc/dt + d(q + c v)/dx = 0                                                       (1)

On the 20-bin x 13-frame data grid (h = 1/20), for every current m, cell j and frame interval k -> k+1:

    q_{j+1/2} = -D(c_f) alpha(c_f) (c_{j+1} - c_j)/h + t+(c_f) iota,   c_f = (c_j + c_{j+1})/2   (interior faces)
    v_{1/2} = 0,   v_{j+1/2} = v_{j-1/2} - V(c_j) (q_{j+1/2} - q_{j-1/2}),   q_{1/2} = iota      (volume constraint)
    J_{j+1/2} = q_{j+1/2} + c_f v_{j+1/2},   J = iota at both walls                              (total flux)
    r_j^{k+1/2} = (c_j^{k+1} - c_j^k)/dt_k + 1/2 [ (J_{j+1/2} - J_{j-1/2})/h |_k + (same) |_{k+1} ]
    L_PDE = mean over valid (m, k, j) of (r / s)^2,   s = RMS of dc/dt over the training set.

c is the measured concentration; alpha(c) and V(c) come from the known density.  D and t+ (from the
predicted theta) are the only unknowns, so minimising L_PDE finds the D(c), t+(c) that make the measured
c(x, t) satisfy Eq. (1).  At the true coefficients L_PDE is 0.0052 (discretisation error of the 20-bin grid).
"""
from __future__ import annotations

import numpy as np
import torch

from solver.properties import RHO_COEFF, alpha_V

from .material import d_hat, t_plus

H = 1.0 / 20
TAU = (240.0 + 960.0 * np.arange(13)) / 33640.0


def observed_fields(y, iotas):
    """Quantities of Eq. (1) that come from the data.  y (n, 3, 13, 20), NaN beyond the valid frames.
    Returns float32 tensors: c_f, dc/dx, alpha at the interior faces (n, 3, 13, 19); V at the cells
    (n, 3, 13, 20); dc/dt at the frame midpoints (n, 3, 12, 20); midpoint mask (n, 12); iota (n, 3)."""
    valid = np.isfinite(y[:, 0, :, 0])
    c = np.where(np.isfinite(y), y, 1.0)
    cf = 0.5 * (c[..., 1:] + c[..., :-1])
    fields = dict(cf=cf, cx=(c[..., 1:] - c[..., :-1]) / H, af=np.asarray(alpha_V(cf, RHO_COEFF)[0]),
                  V=np.asarray(alpha_V(c, RHO_COEFF)[1]),
                  ct=(c[:, :, 1:] - c[:, :, :-1]) / np.diff(TAU)[None, None, :, None],
                  mmid=(valid[:, 1:] & valid[:, :-1]).astype(float), iota=np.asarray(iotas, float))
    return {k: torch.tensor(a, dtype=torch.float32) for k, a in fields.items()}


def residual_scale(P):
    """s = RMS of dc/dt over the valid midpoints (computed once, on the training set)."""
    m = P["mmid"][:, None, :, None].expand_as(P["ct"])
    return float(torch.sqrt((P["ct"] ** 2 * m).sum() / m.sum()))


def total_flux(theta, P, idx):
    """J at all 21 faces and 13 frames (B, 3, 13, 21) for predicted theta (B, 8)."""
    cf = P["cf"][idx]
    th = theta[:, None, None, None, :]
    io = P["iota"][idx][:, :, None, None]
    q = -d_hat(cf, th[..., :4]) * P["af"][idx] * P["cx"][idx] + t_plus(cf, th[..., 4:]) * io
    wall = io.expand(*q.shape[:-1], 1)
    q_lo = torch.cat([wall, q], -1)
    v = torch.cumsum(-P["V"][idx][..., :-1] * (q_lo[..., 1:] - q_lo[..., :-1]), -1)
    return torch.cat([wall, q + cf * v, wall], -1)


def pde_loss(theta, P, idx, scale):
    J = total_flux(theta, P, idx)
    div = (J[..., 1:] - J[..., :-1]) / H
    r = (P["ct"][idx] + 0.5 * (div[:, :, 1:] + div[:, :, :-1])) / scale
    m = P["mmid"][idx][:, None, :, None].expand_as(r)
    return (r ** 2 * m).sum() / m.sum()
