"""One-dimensional concentrated-solution transport with concentration-dependent D(c) and t+(c).

Nondimensional model on x in [0, 1], concentrations in M:

    dc/dt = d/dx [ D(c) dc/dx - t+(c) j - c v0 ]
    v0    = Vbar [ D(c) dc/dx + (1 - t+(c)) j ]

The salt flux at both walls equals -j. The thermodynamic factor is 1 and Vbar = 0.1.

D(c) and t+(c) are cubic Bernstein polynomials on [C_LO, C_HI]. The parameter vector theta holds
their 8 coefficients: theta[:4] for D, theta[4:] for t+. Optimizers work in z = (theta - LO) / (HI - LO),
which maps the parameter box onto [0, 1]^8.

Discretization: finite volumes in space (N_CELLS cells), backward Euler in time (N_STEPS steps) with a
fixed number of Newton iterations per step. Observations are FRAME_EVERY-th time steps, averaged into
N_BINS spatial bins, for every applied current.
"""
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

# Discretization and experiment design
N_CELLS = 40
N_STEPS = 80
T_END = 0.4
FRAME_EVERY = 8                      # 10 observed frames
N_BINS = 20
CURRENTS = jnp.array([0.3, 0.6, 0.9])
C0 = 1.0                             # initial uniform concentration (M)
VBAR = 0.1
NEWTON_ITERS = 4
DX = 1.0 / N_CELLS
DT = T_END / N_STEPS

# Property parameterization
C_LO, C_HI = 0.5, 1.5
THETA_LO = np.array([0.3, 0.3, 0.3, 0.3, -0.3, -0.3, -0.3, -0.3])
THETA_HI = np.array([3.0, 3.0, 3.0, 3.0, 0.8, 0.8, 0.8, 0.8])
THETA_TRUE = np.array([0.6, 1.2, 1.0, 0.7, 0.45, 0.35, 0.20, 0.10])

# Loss barrier: keeps simulated concentrations inside (C_MIN, C_MAX)
C_MIN, C_MAX = 0.05, 1.95
BARRIER_WEIGHT = 100.0
LOSS_FOR_INVALID = 10.0


def bernstein(c):
    """Cubic Bernstein basis evaluated at c, shape (..., 4)."""
    u = jnp.clip((c - C_LO) / (C_HI - C_LO), 0.0, 1.0)
    return jnp.stack([(1 - u) ** 3, 3 * u * (1 - u) ** 2, 3 * u ** 2 * (1 - u), u ** 3], axis=-1)


def properties(c, theta):
    """Return D(c) and t+(c) for coefficient vector theta."""
    B = bernstein(c)
    return B @ theta[:4], B @ theta[4:]


def theta_from_z(z):
    return jnp.asarray(THETA_LO) + jnp.asarray(z) * (jnp.asarray(THETA_HI) - jnp.asarray(THETA_LO))


def z_from_theta(theta):
    return (np.asarray(theta) - THETA_LO) / (THETA_HI - THETA_LO)


def _rate(c, theta, j):
    """dc/dt from the finite-volume flux balance."""
    c_face = 0.5 * (c[1:] + c[:-1])
    D, tp = properties(c_face, theta)
    dcdx = (c[1:] - c[:-1]) / DX
    v0 = VBAR * (D * dcdx + (1.0 - tp) * j)
    flux_interior = D * dcdx - tp * j - c_face * v0
    flux_wall = -j * jnp.ones(1)
    flux = jnp.concatenate([flux_wall, flux_interior, flux_wall])
    return (flux[1:] - flux[:-1]) / DX


def _backward_euler_step(c_old, theta, j):
    def residual(c):
        return c - c_old - DT * _rate(c, theta, j)

    c = c_old
    for _ in range(NEWTON_ITERS):
        c = c - jnp.linalg.solve(jax.jacfwd(residual)(c), residual(c))
    return c, jnp.max(jnp.abs(residual(c)))


def _simulate_current(theta, j):
    def body(c, _):
        c_new, res = _backward_euler_step(c, theta, j)
        return c_new, (c_new, res)

    _, (profiles, residuals) = jax.lax.scan(body, C0 * jnp.ones(N_CELLS), None, length=N_STEPS)
    frames = profiles[FRAME_EVERY - 1::FRAME_EVERY]
    observed = frames.reshape(frames.shape[0], N_BINS, N_CELLS // N_BINS).mean(-1)
    return observed, profiles, jnp.max(residuals)


@jax.jit
def simulate(theta):
    """Simulate every current.

    Returns
    -------
    observed : (n_currents, n_frames, N_BINS) binned concentration frames
    profiles : (n_currents, N_STEPS, N_CELLS) concentration at every step
    newton_residual : (n_currents,) largest Newton residual per current
    """
    return jax.vmap(_simulate_current, in_axes=(None, 0))(theta, CURRENTS)


def synthetic_data():
    """Noise-free observations generated from THETA_TRUE."""
    observed, _, _ = simulate(jnp.asarray(THETA_TRUE))
    return observed


def rmse_loss(z, observed):
    """RMSE (M) between simulated and observed frames at normalized parameters z, plus a domain barrier."""
    predicted, profiles, _ = simulate(theta_from_z(z))
    rmse = jnp.sqrt(jnp.mean((predicted - observed) ** 2))
    barrier = BARRIER_WEIGHT * jnp.mean(jax.nn.relu(C_MIN - profiles) ** 2 + jax.nn.relu(profiles - C_MAX) ** 2)
    total = rmse + barrier
    return jnp.where(jnp.isfinite(total), total, LOSS_FOR_INVALID)


loss = jax.jit(rmse_loss)
loss_and_grad = jax.jit(jax.value_and_grad(rmse_loss))
