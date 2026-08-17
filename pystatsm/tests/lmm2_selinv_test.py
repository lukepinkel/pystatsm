import numpy as np
import patsy
import scipy.sparse as sps

from pystatsm.pylmm.re_mod import LMM2, RandomEffects, MMEBlocked, KronAG
from pystatsm.pylmm.sim_lmm2 import (SimSpec, RanefSpec, CovariateSpec,
                                     MixedModelSim, fit_simulation,
                                     Grouping, Nested, build_groupings)
from pystatsm.utilities.linalg_operations import vech
from pystatsm.utilities.selected_inverse import selected_inverse
from pystatsm.utilities.numerical_derivs import fo_fc_cd


def _cov(n_x):
    return CovariateSpec([f"x{i+1}" for i in range(n_x)],
                         np.zeros(n_x), np.eye(n_x))


def _crossed_model(seed=0, n_obs=1200, q1=40, q2=25):
    rng = np.random.default_rng(seed)
    g = build_groupings(n_obs, Grouping("id1", n_levels=q1, cycle="tile"),
                        Grouping("id2", n_levels=q2, cycle="repeat"))
    spec = SimSpec(n_obs=n_obs, response="y", fe_formula="1 + x1 + x2",
                   beta=np.array([0.2, 0.5, -0.3]),
                   ranef=[RanefSpec("1 + x3", "id1",
                                    G=np.array([[1.0, 0.2], [0.2, 0.5]]),
                                    membership=g["id1"]),
                          RanefSpec("1", "id2", G=np.array([[0.6]]),
                                    membership=g["id2"])],
                   resid_var=0.5, cov_spec=_cov(3))
    sim = MixedModelSim(spec, rng)
    y, _ = sim.draw()
    return sim, sim.to_lmm(y)


def _nested_model(seed=0):
    rng = np.random.default_rng(seed)
    n_obs = 900
    g = build_groupings(n_obs, Grouping("id1", n_levels=45, cycle="tile"),
                        Grouping("id3", n_levels=15, cycle="repeat"),
                        Nested("id2", parent="id3", n_per_parent=2))
    spec = SimSpec(n_obs=n_obs, response="y", fe_formula="1 + x1 + x2",
                   beta=np.array([0.0, 0.5, -0.3]),
                   ranef=[RanefSpec("1 + x3", "id1",
                                    G=np.array([[1.0, 0.15], [0.15, 0.4]]),
                                    membership=g["id1"]),
                          RanefSpec("1 + x4", "id2",
                                    G=np.array([[0.8, 0.1], [0.1, 0.3]]),
                                    membership=g["id2"]),
                          RanefSpec("1", "id3", G=np.array([[0.5]]),
                                    membership=g["id3"])],
                   resid_var=0.5, cov_spec=_cov(4))
    sim = MixedModelSim(spec, rng)
    y, _ = sim.draw()
    return sim, sim.to_lmm(y)


def _kronag_mme(seed=0, n_obs=600, q1=25, q2=20):
    rng = np.random.default_rng(seed)
    g = build_groupings(n_obs, Grouping("ga", n_levels=q1, cycle="tile"),
                        Grouping("gb", n_levels=q2, cycle="repeat"))
    spec = SimSpec(n_obs=n_obs, response="y", fe_formula="1 + x1",
                   beta=np.array([0.2, 0.5]),
                   ranef=[RanefSpec("1 + x2", "ga",
                                    G=np.array([[1.0, 0.2], [0.2, 0.5]]),
                                    membership=g["ga"]),
                          RanefSpec("1", "gb", G=np.array([[0.6]]),
                                    membership=g["gb"])],
                   resid_var=0.5, cov_spec=_cov(2))
    sim = MixedModelSim(spec, rng)
    df = sim.df.copy()
    df["y"] = sim.draw()[0]
    B = rng.normal(size=(q1, q1)) * 0.15
    A = B.dot(B.T) + np.eye(q1)
    re_terms = [(r.re_formula, r.group_var) for r in spec.ranef]
    re_mod = RandomEffects(re_terms, data=df, a_covs=[A, None])
    assert isinstance(re_mod.gterms[0].cov_structure, KronAG)
    X = patsy.dmatrix("1 + x1", data=df, return_type="dataframe").values
    y = df["y"].values.reshape(-1, 1)
    return MMEBlocked(X, y, re_mod)


def _perturb(theta_true, rng):
    out = [np.asarray(theta_true, dtype=float).copy()]
    t = theta_true.copy()
    t[:-1] = t[:-1] * rng.uniform(0.6, 1.5, t.size - 1)
    t[-1] = t[-1] * rng.uniform(0.7, 1.4)
    out.append(t)
    return out


def test_takahashi_matches_dense_inverse():
    import sksparse.cholmod as chm
    rng = np.random.default_rng(0)
    n = 90
    A = sps.random(n, n, density=0.05, random_state=2)
    C = (A @ A.T).tocsc() + sps.eye(n, format="csc") * 3.0
    if hasattr(chm, "cho_factor"):
        f = chm.cho_factor(sps.csc_array(C))
        L = sps.csc_matrix(f.get_factor(kind="LL", lower=True))
        perm = np.asarray(f.perm)
    else:
        f = chm.cholesky(C)
        L = sps.csc_matrix(f.L())
        perm = np.asarray(f.P())
    Sx, Lp, Li = selected_inverse(L)
    Cinv = np.linalg.inv(C.toarray())[np.ix_(perm, perm)]
    cols = np.repeat(np.arange(n), np.diff(Lp))
    assert np.max(np.abs(Sx - Cinv[Li, cols])) < 1e-10


def test_selinv_matches_sparse_crossed():
    rng = np.random.default_rng(1)
    sim, model = _crossed_model()
    mme = model.mme
    assert mme._use_selinv_grad and not mme._use_blockdiag_grad
    for theta in _perturb(sim.theta_true, rng):
        for reml in (True, False):
            g_i = mme._gradient_selinv(theta, reml)
            g_s = mme._gradient_sparse(theta, reml)
            assert np.allclose(g_i, g_s, rtol=1e-8, atol=1e-8), \
                (reml, np.max(np.abs(g_i - g_s)))


def test_selinv_matches_sparse_nested_three():
    rng = np.random.default_rng(2)
    sim, model = _nested_model()
    mme = model.mme
    assert mme._use_selinv_grad
    for theta in _perturb(sim.theta_true, rng):
        for reml in (True, False):
            g_i = mme._gradient_selinv(theta, reml)
            g_s = mme._gradient_sparse(theta, reml)
            assert np.allclose(g_i, g_s, rtol=1e-8, atol=1e-8)


def test_selinv_matches_blockdiag_single_factor():
    rng = np.random.default_rng(4)
    spec = SimSpec(n_obs=600, response="y", fe_formula="1 + x1 + x2",
                   beta=np.array([0.2, 0.5, -0.3]),
                   ranef=[RanefSpec("1 + x3", "g",
                                    G=np.array([[1.0, 0.2], [0.2, 0.5]]),
                                    n_groups=60, n_per=10)],
                   resid_var=0.5, cov_spec=_cov(3))
    sim = MixedModelSim(spec, rng)
    model = sim.to_lmm(sim.draw()[0])
    mme = model.mme
    assert mme._use_blockdiag_grad and mme._use_selinv_grad
    theta = sim.theta_true.copy()
    g_b = mme._gradient_blockdiag(theta, True)
    g_i = mme._gradient_selinv(theta, True)
    assert np.allclose(g_b, g_i, rtol=1e-8, atol=1e-8)


def test_kronag_selinv_matches_sparse_and_fd():
    mme = _kronag_mme()
    assert mme._use_selinv_grad
    theta = mme.re_mod.theta.copy()
    theta[:-1] += 0.1
    for reml in (True, False):
        g_i = mme._gradient_selinv(theta, reml)
        g_s = mme._gradient_sparse(theta, reml)
        assert np.allclose(g_i, g_s, rtol=1e-7, atol=1e-7), \
            (reml, np.max(np.abs(g_i - g_s)))
    g_fd = fo_fc_cd(lambda t: mme._loglike(t, reml=True), theta)
    assert np.max(np.abs(mme._gradient_selinv(theta, True) - g_fd)) < 1e-4


def test_cinv_diagonal_matches_unit_solves():
    sim, model = _crossed_model(seed=6, n_obs=800, q1=40, q2=20)
    mme = model.mme
    theta = sim.theta_true
    model._factor_C(theta)
    d_fast = mme.cinv_diagonal()
    n_ranef = mme.Z.shape[1]
    d_slow = np.zeros(n_ranef)
    e = np.zeros((n_ranef, 1))
    for i in range(n_ranef):
        e[i, 0] = 1.0
        d_slow[i] = float(np.asarray(mme.chol_fac.solve_A(e)).reshape(-1)[i])
        e[i, 0] = 0.0
    assert np.allclose(d_fast, d_slow, rtol=1e-9, atol=1e-11)


def test_fit_crossed_selinv_vs_sparse():
    sim, model = _crossed_model(seed=9)
    th_fast, opt_fast = fit_simulation(model, reml=True)
    model.mme._use_selinv_grad = False
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
