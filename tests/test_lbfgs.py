"""Checks of the L-BFGS fit on the full model (solver/integrate.py), on a synthetic material made here."""
import jax.numpy as jnp
import numpy as np

from inverse import lbfgs
from solver.dataset import sample
from solver.integrate import SolverOptions

THETA, IOTAS = sample(7)


def _data(theta=THETA):
    y, valid = lbfgs._forward(jnp.asarray(theta), jnp.asarray(IOTAS), SolverOptions())
    y = np.array(y)
    K = int(np.asarray(valid).all(0).argmin()) if not np.asarray(valid).all() else y.shape[1]
    y[:, K:] = np.nan
    return y


def _args(y):
    return jnp.asarray(y), jnp.asarray(np.isfinite(y[0, :, 0])), jnp.asarray(IOTAS)


def test_loss_is_zero_at_true_parameters():
    mse, ok = lbfgs.loss_only(jnp.asarray(THETA), *_args(_data()))
    assert bool(ok) and float(mse) < 1e-24


def test_adjoint_gradient_matches_central_differences():
    args = _args(_data())
    th = np.clip(THETA + np.random.default_rng(0).uniform(-0.1, 0.1, 8), 0.05, 0.95)
    _, g, ok = lbfgs.loss_and_grad(jnp.asarray(th), *args)
    assert bool(ok)
    h = 1e-6
    fd = np.array([(float(lbfgs.loss_only(jnp.asarray(th + h * e), *args)[0])
                    - float(lbfgs.loss_only(jnp.asarray(th - h * e), *args)[0])) / (2 * h) for e in np.eye(8)])
    assert np.max(np.abs(np.asarray(g) - fd)) / np.max(np.abs(fd)) < 1e-5


def test_fit_recovers_theta_from_a_nearby_start():
    y = _data()
    r = lbfgs.fit(np.clip(THETA + 0.05, 0, 1), y, IOTAS, max_iter=200)
    assert r["rmse"] < 1e-4 * np.nanmean(y)


def test_fit_from_the_generic_center_start():
    y = _data()
    r = lbfgs.fit(np.full(8, 0.5), y, IOTAS, max_iter=300)
    assert r["rmse"] < 1e-3 * np.nanmean(y)
