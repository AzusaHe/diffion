"""D(c) and t+(c) from the 8 unit Bernstein coefficients, in torch (same definitions as solver/properties.py).

    u = clip(c / 2, 0, 1),   ln D = ln D_min + B(u; theta_D) (ln D_max - ln D_min),   t+ = t_min + B(u; theta_t) (t_max - t_min)
"""
from __future__ import annotations

import math

import torch

LOG_D_MIN, LOG_D_MAX = math.log(1e-12), math.log(2e-11)
D_REF = 1e-11
T_MIN, T_MAX = -0.25, 1.0


def bernstein_basis(u):
    """u (...) -> cubic Bernstein basis (..., 4)."""
    return torch.stack([(1 - u) ** 3, 3 * u * (1 - u) ** 2, 3 * u ** 2 * (1 - u), u ** 3], -1)


def d_hat(c, theta_d):
    """D(c) / D_ref."""
    u = (c / 2).clamp(0, 1)
    return torch.exp(LOG_D_MIN + (bernstein_basis(u) * theta_d).sum(-1) * (LOG_D_MAX - LOG_D_MIN)) / D_REF


def t_plus(c, theta_t):
    u = (c / 2).clamp(0, 1)
    return T_MIN + (bernstein_basis(u) * theta_t).sum(-1) * (T_MAX - T_MIN)
