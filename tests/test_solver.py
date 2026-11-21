import jax.numpy as jnp
import numpy as np

from solver import transport_1d as model


def test_newton_converges_and_mass_is_conserved():
    _, profiles, newton_residual = model.simulate(jnp.asarray(model.THETA_TRUE))
    assert float(newton_residual.max()) < 1e-10
    assert float(jnp.max(jnp.abs(profiles.mean(-1) - model.C0))) < 1e-10


def test_loss_is_zero_at_true_parameters():
    observed = model.synthetic_data()
    assert float(model.loss(model.z_from_theta(model.THETA_TRUE), observed)) < 1e-12


def test_gradient_matches_central_differences():
    observed = model.synthetic_data()
    z = np.random.default_rng(1).uniform(0.2, 0.8, 8)
    _, grad = model.loss_and_grad(z, observed)
    h = 1e-6
    fd = np.array([(float(model.loss(z + h * e, observed)) - float(model.loss(z - h * e, observed))) / (2 * h)
                   for e in np.eye(8)])
    assert np.max(np.abs(np.asarray(grad) - fd)) / np.max(np.abs(fd)) < 1e-5
