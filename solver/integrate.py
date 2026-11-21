"""Crank-Nicolson time stepping with a Newton solve per step.

    R(c^{n+1}) = c^{n+1} - c^n - dt [ w F(c^{n+1}) + (1 - w) F(c^n) ] = 0,    F = -div J.

Schedule: steps_per_interval uniform steps between consecutive output times.  The uniform start c = 1 is
incompatible with the wall flux, so the first steps are `rannacher` backward-Euler half steps (w = 1),
which keeps second order globally.

Newton: adaptive iterations with the exact O(N) structured linear solve (structured.py).  Gradients with
respect to theta use the implicit-function theorem: dy/dargs = -(dR/dy)^{-1} dR/dargs, applied through
lax.custom_linear_solve, so reverse mode costs one transposed structured solve per step.

simulate() is a pure JAX function: jit it and vmap it over theta, currents or initial profiles.
"""
from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp

from . import structured
from .fv import Grid, rhs


@dataclass(frozen=True)
class SolverOptions:
    steps_per_interval: int = 16
    rannacher: int = 4             # backward-Euler half steps at the start (even)
    newton_rtol: float = 1e-11
    newton_atol: float = 1e-11
    max_newton: int = 30
    domain_c_min: float = 1e-4     # admissible concentrations; the Bernstein domain ends at c = 2
    domain_c_max: float = 2.0


class Protocol(eqx.Module):
    """Known conditions of one experiment: material model, initial cell averages, output times, current."""
    model: eqx.Module
    c0: jnp.ndarray
    t_obs: jnp.ndarray
    iota: jnp.ndarray = eqx.field(converter=jnp.asarray, default=0.0)


def schedule(t_obs, opts: SolverOptions):
    """Step sizes, implicitness weights (1 = backward Euler, 0.5 = Crank-Nicolson) and the index of the
    step that ends at each output time."""
    K, n, r = t_obs.shape[0], opts.steps_per_interval, opts.rannacher
    dts = (t_obs - jnp.concatenate([jnp.zeros(1), t_obs[:-1]])) / n
    dt = jnp.concatenate([jnp.full(r, dts[0] / 2), jnp.repeat(dts, n)[r // 2:]])
    w = jnp.concatenate([jnp.ones(r), jnp.full(K * n - r // 2, 0.5)])
    ends = jnp.asarray([(k + 1) * n - 1 + r // 2 for k in range(K)])
    return dt, w, ends


def wall_traces(c, grid: Grid):
    """Linearly extrapolated concentrations at x = 0 and x = 1."""
    xf, xc = grid.faces, grid.centers
    return (c[0] + (c[0] - c[1]) * (xc[0] - xf[0]) / (xc[1] - xc[0]),
            c[-1] + (c[-1] - c[-2]) * (xf[-1] - xc[-1]) / (xc[-1] - xc[-2]))


def simulate(theta, protocol: Protocol, grid: Grid, opts: SolverOptions = SolverOptions()):
    """c (K, N) at the output times, and validity information.

    info['valid_obs'] (K,) is True up to the last output time at which every step so far converged and kept
    c (cells and extrapolated wall traces) inside [domain_c_min, domain_c_max].
    """
    sg = jax.lax.stop_gradient
    iota, model = protocol.iota, protocol.model
    c0 = jnp.broadcast_to(jnp.asarray(protocol.c0, dtype=jnp.float64), (grid.n,))
    dt, w_all, ends = schedule(jnp.asarray(protocol.t_obs), opts)
    model0, iota0 = sg(model), sg(iota)

    def F(c, th, mdl, cur):
        return rhs(c, cur, mdl(c, th), grid)

    def residual(y, args, th, mdl, cur):
        c_old, explicit, h, w = args
        return y - c_old - h * (w * F(y, th, mdl, cur) + (1.0 - w) * explicit)

    def tolerance(y):
        return opts.newton_atol + opts.newton_rtol * jnp.max(jnp.abs(y))

    def step(c, xs):
        h, w = xs
        explicit = jnp.where(w == 1.0, 0.0, F(c, theta, model, iota))
        args = (c, explicit, h, w)
        args0, th0 = sg(args), sg(theta)
        res0 = lambda y: residual(y, args0, th0, model0, iota0)  # noqa: E731

        def lin_solve(y, r, transpose=False):
            return structured.step_solve(y, th0, model0, iota0, grid, args0[2], args0[3], r, transpose)

        def body(state):
            y, _, k = state
            dx = lin_solve(y, res0(y))
            return y - dx, dx, k + 1

        y_star, _, _ = jax.lax.while_loop(
            lambda s: (s[2] < opts.max_newton) & (jnp.max(jnp.abs(s[1])) > tolerance(s[0])),
            body, (sg(c), jnp.full_like(c, jnp.inf), 0))
        y_star = sg(y_star)
        # y = y* - A^{-1} R(y*, args), A = dR/dy(y*) fixed: equal to y* at convergence; its derivative with
        # respect to (c_old, theta, iota) is the implicit-function adjoint.
        y = y_star - jax.lax.custom_linear_solve(
            lambda x: jax.jvp(res0, (y_star,), (x,))[1], residual(y_star, args, theta, model, iota),
            solve=lambda _, r: lin_solve(y_star, r),
            transpose_solve=lambda _, r: lin_solve(y_star, r, transpose=True))
        ok = jnp.max(jnp.abs(res0(y_star))) <= tolerance(y_star)
        return y, (y, ok)

    _, (c_all, ok) = jax.lax.scan(step, c0, (dt, w_all))
    ca = sg(c_all)
    cl, cr = jax.vmap(lambda c: wall_traces(c, grid))(ca)
    finite = jnp.all(jnp.isfinite(ca), axis=1)
    lo = jnp.where(finite, jnp.minimum(jnp.min(ca, axis=1), jnp.minimum(cl, cr)), -jnp.inf)
    hi = jnp.where(finite, jnp.maximum(jnp.max(ca, axis=1), jnp.maximum(cl, cr)), jnp.inf)
    in_domain = (jax.lax.cummin(lo)[ends] >= opts.domain_c_min) & (jax.lax.cummax(hi)[ends] <= opts.domain_c_max)
    converged = jax.lax.cummin(ok.astype(jnp.int32))[ends].astype(bool)
    return c_all[ends], dict(valid_obs=in_domain & converged, newton_converged=jnp.all(ok))
