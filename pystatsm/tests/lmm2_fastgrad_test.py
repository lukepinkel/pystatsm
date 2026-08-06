import numpy as np

from pystatsm.pylmm.sim_lmm2 import (SimSpec, RanefSpec, CovariateSpec,
                                     MixedModelSim, fit_simulation)
from pystatsm.pylmm.re_mod import LMM2
from pystatsm.utilities.linalg_operations import vech
from pystatsm.utilities.numerical_derivs import fo_fc_cd


def _single_spec(n_groups=60, n_per=8, n_v=2, G=None, membership=None):
    if G is None:
        G = np.eye(n_v) * 0.8 + 0.1
        np.fill_diagonal(G, np.diag(G) + 0.2)
    re_form = "1" if n_v == 1 else "1 + " + " + ".join(
        f"x{i}" for i in range(3, 3 + n_v - 1))
    n_x = 2 + max(0, n_v - 1)
    ranef = RanefSpec(re_form, "g", G=np.asarray(G), membership=membership,
                      n_groups=None if membership is not None else n_groups,
                      n_per=None if membership is not None else n_per)
    n_obs = n_groups * n_per if membership is None else len(membership)
    return SimSpec(n_obs=n_obs, response="y", fe_formula="1 + x1 + x2",
                   beta=np.array([0.2, 0.5, -0.3]), ranef=[ranef],
                   resid_var=0.5,
                   cov_spec=CovariateSpec([f"x{i+1}" for i in range(n_x)],
                                          np.zeros(n_x), np.eye(n_x)))


def _model(spec, seed=0):
    sim = MixedModelSim(spec, np.random.default_rng(seed))
    y, _ = sim.draw()
    return sim, sim.to_lmm(y)


def _thetas(p, rng):
    thetas = [np.r_[vech(np.eye(p)), 1.0],
              np.r_[vech(np.diag(np.linspace(1.0, 1e-3, p))), 1.0]]
    A = rng.normal(size=(p, p)) * 0.25
    G = A.dot(A.T) + np.diag(np.linspace(1.2, 0.6, p))
    thetas.append(np.r_[vech(G), rng.uniform(0.3, 1.5)])
    return thetas


def test_fast_path_active_and_matches_sparse():
    rng = np.random.default_rng(3)
    for n_v in (1, 2, 3):
        sim, model = _model(_single_spec(n_v=n_v), seed=n_v)
        mme = model.mme
        assert mme._use_blockdiag_grad
        for theta in _thetas(n_v, rng):
            for reml in (True, False):
                g_f = mme._gradient_blockdiag(theta, reml)
                g_s = mme._gradient_sparse(theta, reml)
                assert np.allclose(g_f, g_s, rtol=1e-8, atol=1e-8), \
                    (n_v, reml, np.max(np.abs(g_f - g_s)))


def test_fast_path_unbalanced_membership():
    rng = np.random.default_rng(11)
    m = np.repeat(np.arange(40), rng.integers(1, 12, size=40))
    spec = _single_spec(membership=m)
    sim, model = _model(spec, seed=5)
    assert model.mme._use_blockdiag_grad
    theta = sim.theta_true.copy()
    g_f = model.mme._gradient_blockdiag(theta, True)
    g_s = model.mme._gradient_sparse(theta, True)
    assert np.allclose(g_f, g_s, rtol=1e-8, atol=1e-8)


def test_fast_gradient_matches_finite_differences():
    sim, model = _model(_single_spec(n_groups=80), seed=42)
    theta = np.array([1.2, 0.1, 0.6, 0.7])
    for reml in (True, False):
        g_an = model.gradient(theta, reml=reml)
        g_fd = fo_fc_cd(lambda t: model.loglike(t, reml=reml), theta)
        assert np.max(np.abs(g_an - g_fd)) < 1e-5


def test_reparam_gradient_matches_finite_differences():
    sim, model = _model(_single_spec(n_groups=80), seed=42)
    eta = np.array([0.1, 0.05, -0.2, 0.0])
    g_an = model.gradient_reparam(eta, True)
    g_fd = fo_fc_cd(lambda e: model.loglike_reparam(e, True), eta)
    assert np.max(np.abs(g_an - g_fd)) < 1e-5


def test_multiterm_falls_back_to_sparse():
    rng = np.random.default_rng(0)
    n_obs = 800
    spec = SimSpec(
        n_obs=n_obs, response="y", fe_formula="1 + x1",
        beta=np.array([0.0, 0.5]),
        ranef=[RanefSpec("1", "ga", G=np.array([[0.8]]),
                         membership=rng.integers(0, 40, size=n_obs)),
               RanefSpec("1", "gb", G=np.array([[0.4]]),
                         membership=rng.integers(0, 25, size=n_obs))],
        resid_var=0.5,
        cov_spec=CovariateSpec(["x1"], np.zeros(1), np.eye(1)))
    sim, model = _model(spec)
    assert not model.mme._use_blockdiag_grad
    g = model.gradient(sim.theta_true, True)
    g_fd = fo_fc_cd(lambda t: model.loglike(t, reml=True), sim.theta_true)
    assert np.max(np.abs(g - g_fd)) < 1e-4


def test_fit_agrees_with_sparse_forced_fit():
    sim, model = _model(_single_spec(n_groups=100, n_per=10), seed=1)
    th_fast, opt_fast = fit_simulation(model, reml=True)
    model.mme._use_blockdiag_grad = False
    th_sparse, opt_sparse = fit_simulation(model, reml=True)
    assert opt_fast.success and opt_sparse.success
    assert np.allclose(th_fast, th_sparse, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    import sys
    fns = [v for k, v in list(globals().items())
           if k.startswith("test_") and callable(v)]
    n_pass = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            n_pass += 1
        except AssertionError as e:
            print(f"  FAIL  {fn.__name__}: {e}")
        except Exception as e:
            print(f"  ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{n_pass}/{len(fns)} tests passed")
    sys.exit(0 if n_pass == len(fns) else 1)
