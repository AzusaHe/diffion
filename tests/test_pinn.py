"""Checks of the JAX PINN pieces (pinn/jax_pinn.py) that do not need training."""
import jax
import jax.numpy as jnp
import numpy as np

from pinn import jax_pinn as P


def _params(build="cv", seed=0):
    from inverse.lbfgs import load_test
    T = load_test(5)
    return P.make_problem(T["y"][0], T["iotas"][0], np.full(8, 0.5), build, jax.random.PRNGKey(seed)) + (T,)


def test_bin_average_of_a_constant_is_that_constant():
    assert np.isclose(float(P.WB.reshape(20, 4).sum(1).max()), 1.0) and np.isclose(float(P.WB.reshape(20, 4).sum(1).min()), 1.0)
    x = np.asarray(P.XB).reshape(20, 4)
    assert np.all((x > np.linspace(0, 1, 21)[:-1, None]) & (x < np.linspace(0, 1, 21)[1:, None]))


def test_initial_condition_and_wall_velocity_are_exact():
    p, S, _ = _params("cv")
    assert abs(float(P.c_point(p, 0.3, 0.0, 1.0, S["tmax"])) - 1.0) < 1e-12
    assert abs(float(P.v_point(p, 0.0, 0.1, 1.0, S["tmax"], S["vscale"]))) < 1e-12


def test_autodiff_derivatives_match_finite_differences():
    p, S, _ = _params("cv")
    th = P.theta_of(p["phi"])
    x, t, io, h = 0.37, 0.12, 1.3, 1e-6
    c, cx, ct, q, qx, V = P.fields(p, th, x, t, io, S["tmax"])
    f = lambda x, t: P.c_point(p, x, t, io, S["tmax"])  # noqa: E731
    assert np.isclose(float(cx), float((f(x + h, t) - f(x - h, t)) / (2 * h)), rtol=1e-6)
    assert np.isclose(float(ct), float((f(x, t + h) - f(x, t - h)) / (2 * h)), rtol=1e-6)
    g = lambda x: P.q_point(p, th, x, t, io, S["tmax"])  # noqa: E731
    assert np.isclose(float(qx), float((g(x + h) - g(x - h)) / (2 * h)), rtol=1e-5)


def test_build2_flux_integral_matches_conservation():
    """J from the quadrature of c_t: J(x) - J(0) = -int c_t, checked against a fine trapezoid sum."""
    p, S, _ = _params("c")
    th = P.theta_of(p["phi"])
    t, io = 0.05, 1.3
    xf = np.linspace(0, 1, 20001)
    ct = np.asarray(jax.vmap(lambda x: jax.grad(P.c_point, argnums=2)(p, x, t, io, S["tmax"]))(jnp.asarray(xf)))
    ref = io - np.trapezoid(ct, xf)
    _, j1, _ = P.build2_line(p, th, t, io, S)
    assert abs(float(j1) + io - ref) < 1e-6 * max(1.0, abs(ref))


def test_loss_gradient_reaches_theta_in_both_builds():
    for build, fn in (("cv", P.loss_cv), ("c", P.loss_c)):
        p, S, _ = _params(build)
        g = jax.grad(lambda q: fn(q, S, jax.random.PRNGKey(1), dict(pde=1.0, vol=1.0, bc=1.0))[0])(p)
        assert np.all(np.isfinite(np.asarray(g["phi"]))) and float(jnp.abs(g["phi"]).max()) > 0
