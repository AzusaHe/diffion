"""The material model of solver/properties.py in torch: D(c), t+(c), the density and alpha(c), V(c).

    u = clip(c / 2, 0, 1),   ln D = ln D_min + B(u; theta_D) (ln D_max - ln D_min),   t+ = t_min + B(u; theta_t) (t_max - t_min)
    rho = rho_min + B(u; rho_coeff) (rho_max - rho_min),   rho' = d rho / d c
    alpha = (rho - c rho') / (rho - M C_ref c),   V = (M C_ref - rho') / (rho - c rho')
Every function works on tensors of any shape, device and dtype, and under torch.func transforms.
tests/test_torch_pinn.py checks them against the JAX originals.
"""
from __future__ import annotations

import math

import torch

LOG_D_MIN, LOG_D_MAX = math.log(1e-12), math.log(2e-11)
D_REF = 1e-11
T_MIN, T_MAX = -0.25, 1.0
RHO_MIN, RHO_MAX = 1000.0, 2500.0
RHO_COEFF = (0.084741, 0.218525, 0.431779, 0.698133)      # = solver.properties.RHO_COEFF
M_CREF = 0.28709 * 3000.0                                  # salt molar mass x C_ref, kg/m^3
DU_DC = 3000.0 / 6000.0                                    # u = c C_ref / C_max


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


def quadratic_basis(u):
    return torch.stack([(1 - u) ** 2, 2 * u * (1 - u), u ** 2], -1)


def density(c, rho_coeff=RHO_COEFF):
    """(rho, d rho / d c) in kg/m^3 for dimensionless c (rho' is the curve's slope, also outside the clip)."""
    u = (c / 2).clamp(0, 1)
    k = torch.as_tensor(rho_coeff, dtype=c.dtype, device=c.device)
    span = RHO_MAX - RHO_MIN
    return (RHO_MIN + span * (bernstein_basis(u) * k).sum(-1),
            span * (quadratic_basis(u) * (3 * (k[1:] - k[:-1]))).sum(-1) * DU_DC)


def alpha_V(c, rho_coeff=RHO_COEFF):
    """alpha(c) and V(c) from the known density."""
    rho, rho_c = density(c, rho_coeff)
    a = rho - c * rho_c
    return a / (rho - M_CREF * c), (M_CREF - rho_c) / a


def material(c, theta, rho_coeff=RHO_COEFF):
    """(D alpha, t+, V) at c for theta (..., 8), as solver.properties.BernsteinMaterial."""
    a, V = alpha_V(c, rho_coeff)
    return d_hat(c, theta[..., :4]) * a, t_plus(c, theta[..., 4:8]), V
