"""Forward physics: transport models and their numerical solvers (JAX, float64).

transport_1d  compact model for the optimizer baselines (inverse/): thermodynamic factor 1, Vbar = 0.1,
              backward Euler, one fixed true material.

The modules below form the model behind the networks and test-time training (surrogate/, pinn/):

    dc/dt + d(q + c v)/dx = 0,   q = -D(c) alpha(c) dc/dx + t+(c) iota,   dv/dx = -V(c) dq/dx,
    J = q + c v = iota at both walls,  v = 0 at x = 0,  c(x, 0) = 1.

properties  material model: cubic-Bernstein D(c), t+(c); fixed cubic-Bernstein density -> alpha(c), V(c)
fv          conservative finite-volume fluxes on a uniform cell-centred grid
structured  exact O(N) solve of the Newton system of one time step
integrate   Crank-Nicolson time stepping, Newton per step, implicit-function-theorem adjoint
dataset     synthetic training / validation / test materials
"""
import jax

jax.config.update("jax_enable_x64", True)

from .fv import Grid, face_fields, uniform_grid  # noqa: E402
from .integrate import Protocol, SolverOptions, simulate  # noqa: E402
from .properties import RHO_COEFF, BernsteinMaterial, alpha_V, d_hat, t_plus  # noqa: E402

__all__ = ["Grid", "uniform_grid", "face_fields", "Protocol", "SolverOptions", "simulate",
           "BernsteinMaterial", "RHO_COEFF", "alpha_V", "d_hat", "t_plus"]
