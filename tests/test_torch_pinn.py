"""The torch PINN (pinn/torch_pinn.py) and torch material (pinn/material.py) against their JAX originals."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from torch.func import grad

from pinn import jax_pinn as J
from pinn import material as M
from pinn import torch_pinn as P
from solver.properties import RHO_COEFF, BernsteinMaterial, alpha_V

DEV = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def test_alpha_V_and_material_match_jax():
    c = np.linspace(0.01, 2.5, 301)
    th = np.random.default_rng(0).random(8)
    a, V = alpha_V(c)
    at, Vt = M.alpha_V(torch.tensor(c))
    np.testing.assert_allclose(at.numpy(), np.asarray(a), rtol=1e-13)
    np.testing.assert_allclose(Vt.numpy(), np.asarray(V), rtol=1e-12)
    for x, y in zip(BernsteinMaterial(jnp.asarray(RHO_COEFF))(c, th), M.material(torch.tensor(c), torch.tensor(th))):
        np.testing.assert_allclose(y.numpy(), np.asarray(x), rtol=1e-12)
    assert np.allclose(M.RHO_COEFF, RHO_COEFF)


def _setup(build, device):
    from inverse.lbfgs import load_test
    T = load_test(5)
    pj, S = J.make_problem(T["y"][0], T["iotas"][0], np.full(8, 0.4), build, jax.random.PRNGKey(0))
    conv = lambda a: torch.tensor(np.asarray(a), dtype=torch.float64, device=device)  # noqa: E731
    pt = {k: ([(conv(W), conv(b)) for W, b in v] if k != "phi" else conv(v)) for k, v in pj.items()}
    st = P.problem_stats(T["y"][0], T["iotas"][0])
    St = {k: torch.tensor(st[k], device=device, dtype=torch.bool if k == "mask" else torch.float64)
          for k in ("y", "mask", "iotas", "tmax", "vscale", "st", "sigma", "s_lo")}
    for k in ("tmax", "vscale", "st", "sigma", "s_lo"):
        assert np.isclose(st[k], S[k])
    return pj, S, pt, St


@pytest.mark.parametrize("device", DEV)
@pytest.mark.parametrize("build", ["cv", "c"])
def test_losses_and_theta_gradient_match_jax(build, device):
    pj, S, pt, St = _setup(build, device)
    key = jax.random.PRNGKey(3)
    w = dict(pde=100.0, vol=100.0, bc=100.0)
    if build == "cv":
        k1, k2 = jax.random.split(key)
        x = jax.random.uniform(k1, (3, S["n_res"]))
        t = S["tmax"] * jax.random.uniform(k2, (3, S["n_res"]), minval=S["s_lo"], maxval=1.0) ** 2
        fj = J.loss_cv
    else:
        x = jnp.zeros((3, 1))
        t = S["tmax"] * jax.random.uniform(key, (3, S["n_times"]), minval=S["s_lo"], maxval=1.0) ** 2
        fj = J.loss_c
    (Lj, partsj), gj = jax.value_and_grad(lambda q: fj(q, S, key, w), has_aux=True)(pj)
    LS = P.Losses(build, torch.float64, device, w=w)
    xt = torch.tensor(np.asarray(x), dtype=torch.float64, device=device)
    tt = torch.tensor(np.asarray(t), dtype=torch.float64, device=device)
    gt = grad(lambda q: LS.loss(q, St, xt, tt)[0])(pt)
    Lt, partst = LS.loss(pt, St, xt, tt)
    assert np.isclose(float(Lt), float(Lj), rtol=1e-9)
    for k in partsj:
        assert np.isclose(float(partst[k]), float(partsj[k]), rtol=1e-8, atol=1e-14), k
    np.testing.assert_allclose(gt["phi"].cpu().numpy(), np.asarray(gj["phi"]), rtol=1e-7, atol=1e-10)
    np.testing.assert_allclose(gt["c"][0][0].cpu().numpy(), np.asarray(gj["c"][0][0]), rtol=1e-6, atol=1e-9)


def test_short_batched_training_runs_and_stays_finite():
    from inverse.lbfgs import load_test
    T = load_test(3)
    for build in ("cv", "c"):
        R = P.train_batch(list(T["y"][:2]), list(T["iotas"][:2]), [np.full(8, 0.5)] * 2, build, [0, 1], steps=3,
                          log_every=1, device=DEV[-1], n_res=64, n_times=4)
        assert len(R) == 2 and all(np.isfinite(r["rel_rmse"]) and np.all(np.isfinite(r["theta"])) for r in R)
        assert R[0]["theta"] != [0.5] * 8
