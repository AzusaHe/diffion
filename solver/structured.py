"""Exact O(N) solve of the Newton system of one time step.

For R(y) = y - c_old - h [w F(y) + ...] with F = -Delta J / h_i:

    dR/dy = I + diag(h w / h_i) Delta (dJ/dy),     J_k = q_k + cf_k v_k  (interior faces k = 1..N-1),
    v_k = sum_{m<=k} g_m,                          g_m = -V_{m-1} (q_m - q_{m-1}).

q_k, cf_k and g_m depend on at most three neighbouring cells, so

    dJ/dy = P + diag(cf) L G,   P = dq + diag(v) dcf (bidiagonal),   G = dg (tridiagonal),
    L = lower-triangular matrix of ones (the cumulative sum that gives v).

With s = L G y (s_k - s_{k-1} = (G y)_k) the system becomes block-tridiagonal with 2x2 blocks in the
interleaved unknowns z_i = (y_i, s_{i+1}); block Thomas elimination solves it in O(N).  The banded
derivatives come from three forward-mode JVPs (3-colouring of the cells).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .fv import Grid, interior_q


def local_face_terms(c, theta, model, iota, grid: Grid):
    """(q, cf, g) at the interior faces 1..N-1."""
    props = model(c, theta)
    q, cf = interior_q(c, iota, props, grid)
    q_lo = jnp.concatenate([jnp.atleast_1d(iota), q])
    return q, cf, -props[2][:-1] * (q_lo[1:] - q_lo[:-1])


def banded_jacobian(fun, c):
    """Outputs of fun and d out_k / d c_{k-2+m}, m = 0, 1, 2 (each (N-1, 3)), via 3 JVPs."""
    n = c.shape[0]
    j = jnp.arange(n)
    seeds = jnp.stack([(j % 3 == r).astype(c.dtype) for r in range(3)])
    out, tang = jax.vmap(lambda s: jax.jvp(fun, (c,), (s,)), out_axes=(None, 0))(seeds)
    cells = jnp.arange(1, n)[:, None] + jnp.arange(-2, 1)[None, :]          # cells k-2, k-1, k
    colour = jnp.mod(cells, 3)
    jac = jax.tree.map(lambda T: jnp.where(cells >= 0, jnp.take_along_axis(T.T, colour, axis=1), 0.0), tang)
    return out, jac


def jacobian_blocks(c, theta, model, iota, grid: Grid, h, w):
    """2x2 blocks (L, D, U), each (N, 2, 2), of the augmented block-tridiagonal system."""
    n = c.shape[0]
    (q, cf, g), (dq, dcf, dg) = banded_jacobian(lambda y: local_face_terms(y, theta, model, iota, grid), c)
    v = jnp.cumsum(g)
    P = dq + v[:, None] * dcf
    a = h * w / grid.widths
    zero1 = jnp.zeros((1, 3), c.dtype)
    Pp = jnp.concatenate([zero1, P, zero1])
    Gp = jnp.concatenate([zero1, dg, zero1])
    cfp = jnp.concatenate([jnp.zeros(1, c.dtype), cf, jnp.zeros(1, c.dtype)])
    i = jnp.arange(n)
    Pi, Pn, Gn = Pp[i], Pp[i + 1], Gp[i + 1]
    cum = (i <= n - 2).astype(c.dtype)
    L = jnp.stack([jnp.stack([-a * Pi[:, 1], -a * cfp[i]], -1),
                   jnp.stack([-Gn[:, 0], -cum], -1)], -2)
    D = jnp.stack([jnp.stack([1.0 + a * (Pn[:, 1] - Pi[:, 2]), a * cfp[i + 1]], -1),
                   jnp.stack([-Gn[:, 1], jnp.ones(n, c.dtype)], -1)], -2)
    U = jnp.stack([jnp.stack([a * Pn[:, 2], jnp.zeros(n, c.dtype)], -1),
                   jnp.stack([-Gn[:, 2], jnp.zeros(n, c.dtype)], -1)], -2)
    return L, D, U


def inverse_2x2(M):
    det = M[..., 0, 0] * M[..., 1, 1] - M[..., 0, 1] * M[..., 1, 0]
    adj = jnp.stack([jnp.stack([M[..., 1, 1], -M[..., 0, 1]], -1),
                     jnp.stack([-M[..., 1, 0], M[..., 0, 0]], -1)], -2)
    return adj / det[..., None, None]


def block_tridiag_solve(L, D, U, b):
    """Solve L_i z_{i-1} + D_i z_i + U_i z_{i+1} = b_i (2x2 blocks, b (N, 2)) by block Thomas elimination."""
    def forward(carry, xs):
        Dinv_prev, bp_prev, U_prev = carry
        Li, Di, Ui, bi = xs
        W = Li @ Dinv_prev
        bp = bi - W @ bp_prev
        Dinv = inverse_2x2(Di - W @ U_prev)
        return (Dinv, bp, Ui), (Dinv, bp)

    Dinv0 = inverse_2x2(D[0])
    _, (Dinv_r, bp_r) = jax.lax.scan(forward, (Dinv0, b[0], U[0]), (L[1:], D[1:], U[1:], b[1:]))
    Dinv = jnp.concatenate([Dinv0[None], Dinv_r])
    bp = jnp.concatenate([b[0][None], bp_r])
    zN = Dinv[-1] @ bp[-1]

    def backward(z_next, xs):
        Dinv_i, bp_i, U_i = xs
        z = Dinv_i @ (bp_i - U_i @ z_next)
        return z, z

    _, zs = jax.lax.scan(backward, zN, (Dinv[:-1], bp[:-1], U[:-1]), reverse=True)
    return jnp.concatenate([zs, zN[None]])


def transpose_blocks(L, D, U):
    tr = lambda X: jnp.swapaxes(X, -1, -2)  # noqa: E731
    return (jnp.concatenate([jnp.zeros_like(U[:1]), tr(U[:-1])]), tr(D),
            jnp.concatenate([tr(L[1:]), jnp.zeros_like(L[:1])]))


def step_solve(c, theta, model, iota, grid, h, w, r, transpose=False):
    """x = (dR/dy)^{-1} r, or (dR/dy)^{-T} r with transpose=True (used by the adjoint)."""
    blocks = jacobian_blocks(c, theta, model, iota, grid, h, w)
    if transpose:
        blocks = transpose_blocks(*blocks)
    return block_tridiag_solve(*blocks, jnp.stack([r, jnp.zeros_like(r)], -1))[:, 0]
