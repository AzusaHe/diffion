"""Material model in dimensionless form.

Scales: x / L (L = 580 um), t / t_ref (t_ref = L^2 / D_ref = 33,640 s, D_ref = 1e-11 m^2/s),
c = C / C_ref (C_ref = 3000 mol/m^3), iota = i / i_ref (i_ref = F D_ref C_ref / L).

With u = C / 6000 mol/m^3 = c / 2 (clipped to [0, 1]) and cubic Bernstein curves B(u; theta):
    ln D  = ln D_min + B(u; theta_D) (ln D_max - ln D_min),   D in [1e-12, 2e-11] m^2/s
    t+    = t_min + B(u; theta_t) (t_max - t_min),            t+ in [-0.25, 1]
    rho   = rho_min + B(u; rho_coeff) (rho_max - rho_min),    rho in [1000, 2500] kg/m^3
    alpha = (rho - c rho') / (rho - M C_ref c),   V = (M C_ref - rho') / (rho - c rho'),   rho' = d rho / d c.
theta = (theta_D, theta_t) in [0, 1]^8 is what every inference method estimates.  The density (and hence
alpha, V) is known and the same for every material: RHO_COEFF.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import comb

import equinox as eqx
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class Scales:
    length_m: float = 580e-6
    diffusivity_m2_s: float = 1e-11
    concentration_mol_m3: float = 3000.0
    salt_molar_mass_kg_mol: float = 0.28709      # NaTFSI

    @property
    def time_s(self) -> float:
        return self.length_m ** 2 / self.diffusivity_m2_s

    @property
    def m_cref(self) -> float:
        """M C_ref in kg/m^3."""
        return self.salt_molar_mass_kg_mol * self.concentration_mol_m3


@dataclass(frozen=True)
class Bounds:
    d_min_m2_s: float = 1e-12
    d_max_m2_s: float = 2e-11
    t_min: float = -0.25
    t_max: float = 1.0
    c_max_mol_m3: float = 6000.0
    rho_min_kg_m3: float = 1000.0
    rho_max_kg_m3: float = 2500.0


SCALES = Scales()
BOUNDS = Bounds()
RHO_COEFF = np.array([0.084741, 0.218525, 0.431779, 0.698133])   # density curve shared by all materials


def bernstein(coeff, u):
    """Bernstein curve sum_k coeff_k C(n,k) u^k (1-u)^(n-k)."""
    coeff = jnp.asarray(coeff)
    n = coeff.shape[-1] - 1
    basis = jnp.stack([comb(n, k) * u ** k * (1.0 - u) ** (n - k) for k in range(n + 1)], axis=-1)
    return basis @ coeff


def bernstein_derivative(coeff, u):
    """d/du of the Bernstein curve."""
    coeff = jnp.asarray(coeff)
    return bernstein((coeff.shape[-1] - 1) * jnp.diff(coeff), u)


def unit_concentration(c):
    return jnp.clip(c * SCALES.concentration_mol_m3 / BOUNDS.c_max_mol_m3, 0.0, 1.0)


def d_hat(c, theta_d):
    """D(c) / D_ref."""
    lo, hi = np.log(BOUNDS.d_min_m2_s), np.log(BOUNDS.d_max_m2_s)
    return jnp.exp(lo + bernstein(theta_d, unit_concentration(c)) * (hi - lo)) / SCALES.diffusivity_m2_s


def t_plus(c, theta_t):
    return BOUNDS.t_min + bernstein(theta_t, unit_concentration(c)) * (BOUNDS.t_max - BOUNDS.t_min)


def density(c, rho_coeff):
    """(rho, d rho / d c) in kg/m^3, c dimensionless."""
    u = unit_concentration(c)
    span = BOUNDS.rho_max_kg_m3 - BOUNDS.rho_min_kg_m3
    du_dc = SCALES.concentration_mol_m3 / BOUNDS.c_max_mol_m3
    return (BOUNDS.rho_min_kg_m3 + span * bernstein(rho_coeff, u),
            span * bernstein_derivative(rho_coeff, u) * du_dc)


def alpha_V(c, rho_coeff=RHO_COEFF):
    rho, rho_c = density(c, rho_coeff)
    a = rho - c * rho_c
    return a / (rho - SCALES.m_cref * c), (SCALES.m_cref - rho_c) / a


class BernsteinMaterial(eqx.Module):
    """theta (8,) -> cell properties (D alpha, t+, V) at concentrations c."""
    rho_coeff: jnp.ndarray

    def __call__(self, c, theta):
        theta = jnp.asarray(theta)
        a, V = alpha_V(c, self.rho_coeff)
        return d_hat(c, theta[:4]) * a, t_plus(c, theta[4:8]), V
