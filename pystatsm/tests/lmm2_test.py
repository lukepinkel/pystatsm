import numpy as np
import scipy as sp

from pystatsm.pylmm.sim_lmm2 import (
    SimSpec, RanefSpec, CovariateSpec, MixedModelSim, fit_simulation,
    balanced_membership, build_frame, build_design, spec_to_theta,
    draw_ranefs, draw_residuals, variance_components,
    rescale_to_var_ratios, to_lme4_formula,
    Grouping, Nested, build_groupings,
)
from pystatsm.pylmm.re_mod import LMM2, RandomEffects
from pystatsm.utilities.linalg_operations import vech, invech
from pystatsm.utilities.numerical_derivs import fo_fc_cd


# ---------- helpers ---------------------------------------------------------

def _basic_spec(n_groups=20, n_per=10):
    return SimSpec(
        n_obs=n_groups * n_per,
        response='y',
        fe_formula='1 + x1',
        beta=np.array([0.5, 1.0]),
        ranef=[RanefSpec(re_formula='1 + x1', group_var='g',
                         G=np.array([[1.0, 0.2], [0.2, 0.5]]),
                         n_groups=n_groups, n_per=n_per)],
        resid_var=0.5,
        cov_spec=CovariateSpec(cont_vars=['x1'], mean=np.array([0.0]),
                               cov=np.array([[1.0]])),
    )


def _intercept_only_spec(n_groups=30, n_per=8):
    return SimSpec(
        n_obs=n_groups * n_per,
        response='y',
        fe_formula='1',
        beta=np.array([0.0]),
        ranef=[RanefSpec(re_formula='1', group_var='g',
                         G=np.array([[1.5]]),
                         n_groups=n_groups, n_per=n_per)],
        resid_var=1.0,
    )


def _crossed_spec(n_per_a=8, n_per_b=10, n_obs=240):
    rng = np.random.default_rng(0)
    return SimSpec(
        n_obs=n_obs,
        response='y',
        fe_formula='1 + x1',
        beta=np.array([0.0, 0.5]),
        ranef=[
            RanefSpec(re_formula='1', group_var='ga',
                      G=np.array([[0.8]]),
                      membership=rng.integers(0, n_per_a, size=n_obs)),
            RanefSpec(re_formula='1', group_var='gb',
                      G=np.array([[0.4]]),
                      membership=rng.integers(0, n_per_b, size=n_obs)),
        ],
        resid_var=0.5,
        cov_spec=CovariateSpec(cont_vars=['x1'], mean=np.array([0.0]),
                               cov=np.array([[1.0]])),
    )


# ---------- spec / dataclass ------------------------------------------------

def test_balanced_membership_layout():
    m = balanced_membership(4, 3)
    assert m.tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]


def test_to_lme4_formula_matches_expected():
    spec = _basic_spec()
    assert to_lme4_formula(spec) == 'y ~ 1 + x1 + (1 + x1 | g)'


def test_to_lme4_formula_multi_term():
    spec = _crossed_spec()
    f = to_lme4_formula(spec)
    assert f.startswith('y ~ 1 + x1')
    assert '(1 | ga)' in f and '(1 | gb)' in f


def test_spec_to_theta_layout():
    spec = _basic_spec()
    theta = spec_to_theta(spec)
    G = spec.ranef[0].G
    expected = np.concatenate([vech(G), [spec.resid_var]])
    assert np.allclose(theta, expected)
    assert theta.shape == (4,)


def test_spec_to_theta_multi_term():
    spec = _crossed_spec()
    theta = spec_to_theta(spec)
    parts = [vech(spec.ranef[0].G), vech(spec.ranef[1].G), [spec.resid_var]]
    assert np.allclose(theta, np.concatenate(parts))


# ---------- frame / design --------------------------------------------------

def test_build_frame_dimensions():
    spec = _basic_spec()
    rng = np.random.default_rng(0)
    df = build_frame(spec, rng)
    assert df.shape[0] == spec.n_obs
    assert {'g', 'x1', 'y'}.issubset(df.columns)
    assert df['g'].nunique() == 20


def test_build_frame_no_covspec():
    spec = _intercept_only_spec()
    rng = np.random.default_rng(0)
    df = build_frame(spec, rng)
    assert df.shape == (spec.n_obs, 2)
    assert set(df.columns) == {'g', 'y'}


def test_build_design_shapes():
    spec = _basic_spec()
    rng = np.random.default_rng(0)
    df = build_frame(spec, rng)
    X, re_mod = build_design(spec, df)
    assert X.shape == (spec.n_obs, 2)
    assert re_mod.Z.shape == (spec.n_obs, 20 * 2)


def test_build_design_membership_passthrough():
    """The level_indices LMM2 sees should match what we supplied."""
    n_groups, n_per = 20, 10
    spec = _basic_spec(n_groups, n_per)
    rng = np.random.default_rng(0)
    df = build_frame(spec, rng)
    _, re_mod = build_design(spec, df)
    expected = balanced_membership(n_groups, n_per)
    assert np.array_equal(re_mod.gterms[0].level_indices, expected)


# ---------- draws -----------------------------------------------------------

def test_exact_ranef_draws_have_exact_moments():
    spec = _basic_spec()
    rng = np.random.default_rng(0)
    u = draw_ranefs(spec, rng, exact=True)
    U = u.reshape(-1, 2)
    assert np.allclose(U.mean(axis=0), 0.0, atol=1e-10)
    assert np.allclose(np.cov(U.T, ddof=0), spec.ranef[0].G, atol=1e-10)


def test_exact_residuals_have_exact_moments():
    spec = _basic_spec()
    rng = np.random.default_rng(0)
    eps = draw_residuals(spec, rng, exact=True)
    assert abs(eps.mean()) < 1e-10
    assert abs(eps.var() - spec.resid_var) < 1e-10


def test_ranef_draws_match_z_layout():
    """Z @ u should equal the per-observation random-effect contribution
    computed from the per-group draws (exercises Z column ordering)."""
    spec = _basic_spec(n_groups=10, n_per=5)
    rng = np.random.default_rng(0)
    df = build_frame(spec, rng)
    X, re_mod = build_design(spec, df)
    u = draw_ranefs(spec, np.random.default_rng(1))
    U = u.reshape(-1, 2)
    re_arr = df[['x1']].assign(intercept=1.0)[['intercept', 'x1']].values
    g = re_mod.gterms[0].level_indices
    expected = (re_arr * U[g]).sum(axis=1)
    assert np.allclose(np.asarray(re_mod.Z.dot(u)).reshape(-1), expected)


# ---------- variance components & rescaling ---------------------------------

def test_variance_components_closed_form():
    spec = _basic_spec()
    rng = np.random.default_rng(0)
    df = build_frame(spec, rng)
    X, re_mod = build_design(spec, df)
    v_fe, v_re = variance_components(spec, X, re_mod)
    # x1 ~ N(0,1) (exact_rmvnorm), beta=[0.5, 1] -> Var(X b) = 1
    # E[z'Gz] = G[0,0] + G[1,1]*E[x^2] = 1 + 0.5 = 1.5
    assert abs(v_fe - 1.0) < 1e-9
    assert abs(v_re - 1.5) < 1e-9


def test_rescaling_hits_target_ratios():
    spec = _basic_spec()
    sim = MixedModelSim(spec, np.random.default_rng(0))
    sim2 = sim.rescaled(r_fe=0.2, r_re=0.3)
    v_fe, v_re = sim2.variance_components()
    v_rs = sim2.spec.resid_var
    total = v_fe + v_re + v_rs
    assert abs(v_fe / total - 0.2) < 1e-9
    assert abs(v_re / total - 0.3) < 1e-9
    assert abs(v_rs / total - 0.5) < 1e-9


def test_rescaling_rejects_invalid_ratios():
    spec = _basic_spec()
    rng = np.random.default_rng(0)
    df = build_frame(spec, rng)
    X, re_mod = build_design(spec, df)
    try:
        rescale_to_var_ratios(spec, X, re_mod, 0.6, 0.5)
    except ValueError:
        return
    raise AssertionError("expected ValueError for r_fe + r_re >= 1")


# ---------- MixedModelSim end-to-end ----------------------------------------

def test_sim_draw_produces_y_with_expected_total_var():
    """Single-replicate Var(y) is noisy because Var(Zu) is a u-conditional
    statistic; check that the average across replicates matches the closed
    form, not that any one replicate does."""
    spec = _basic_spec(n_groups=200, n_per=10)
    sim = MixedModelSim(spec, np.random.default_rng(1))
    expected = sum(sim.variance_components()) + spec.resid_var
    vars_y = np.array([sim.draw()[0].var() for _ in range(50)])
    assert abs(vars_y.mean() - expected) / expected < 0.05


def test_sim_to_lmm_returns_working_LMM2():
    spec = _basic_spec()
    sim = MixedModelSim(spec, np.random.default_rng(0))
    y, _ = sim.draw()
    model = sim.to_lmm(y)
    assert isinstance(model, LMM2)
    ll = model.loglike(sim.theta_true)
    assert np.isfinite(ll)


def test_theta_layout_matches_LMM2_theta_init():
    """spec_to_theta must produce a vector of the same length and ordering
    as RandomEffects builds from the same formula."""
    spec = _basic_spec()
    sim = MixedModelSim(spec, np.random.default_rng(0))
    model = sim.to_lmm(sim.draw()[0])
    assert sim.theta_true.shape == model.mme.re_mod.theta.shape


# ---------- objective / gradient consistency --------------------------------

def _model(seed=42, n_groups=40):
    spec = _basic_spec(n_groups=n_groups)
    sim = MixedModelSim(spec, np.random.default_rng(seed))
    y, _ = sim.draw()
    return sim, sim.to_lmm(y)


def test_gradient_reml_matches_finite_differences():
    sim, model = _model()
    theta = np.array([1.2, 0.1, 0.6, 0.7])
    g_an = model.gradient(theta, reml=True)
    g_fd = fo_fc_cd(lambda t: model.loglike(t, reml=True), theta)
    assert np.max(np.abs(g_an - g_fd)) < 1e-5


def test_gradient_ml_matches_finite_differences():
    sim, model = _model()
    theta = np.array([1.2, 0.1, 0.6, 0.7])
    g_an = model.gradient(theta, reml=False)
    g_fd = fo_fc_cd(lambda t: model.loglike(t, reml=False), theta)
    assert np.max(np.abs(g_an - g_fd)) < 1e-5


def test_gradient_reparam_matches_finite_differences():
    sim, model = _model()
    eta = np.array([0.1, 0.05, -0.2, 0.0])
    g_an = model.gradient_reparam(eta, True)
    g_fd = fo_fc_cd(lambda e: model.loglike_reparam(e, True), eta)
    assert np.max(np.abs(g_an - g_fd)) < 1e-5


def test_loglike_reparam_consistency():
    """loglike_reparam(eta) == loglike(rvs(eta))."""
    sim, model = _model()
    eta = np.array([0.1, 0.05, -0.2, 0.0])
    theta = model.mme.re_mod.reparam.rvs(eta)
    assert abs(model.loglike_reparam(eta, True) - model.loglike(theta, True)) < 1e-10


# ---------- recovery sanity check -------------------------------------------

def test_fit_converges_and_is_stationary():
    sim, model = _model(n_groups=80)
    theta_hat, opt = fit_simulation(model, reml=True)
    assert opt.success
    g_at_opt = model.gradient_reparam(opt.x, True)
    assert np.linalg.norm(g_at_opt) < 1e-2


def test_fit_recovers_truth_within_tolerance():
    """One replicate with enough groups should recover within ~3 SE per param."""
    rng = np.random.default_rng(0)
    spec = _basic_spec(n_groups=200, n_per=10)
    sim = MixedModelSim(spec, rng)
    y, _ = sim.draw()
    theta_hat, opt = fit_simulation(sim.to_lmm(y), reml=True)
    rel_err = np.abs(theta_hat - sim.theta_true) / np.maximum(np.abs(sim.theta_true), 0.1)
    assert opt.success
    assert (rel_err < 0.5).all(), f"recovery error too large: {rel_err}"


def test_intercept_only_fit():
    """Sanity check on the simplest possible model."""
    rng = np.random.default_rng(0)
    spec = _intercept_only_spec(n_groups=100, n_per=10)
    sim = MixedModelSim(spec, rng)
    y, _ = sim.draw()
    theta_hat, opt = fit_simulation(sim.to_lmm(y), reml=True)
    assert opt.success
    rel_err = np.abs(theta_hat - sim.theta_true) / sim.theta_true
    assert (rel_err < 0.2).all()


# ---------- crossed + nested grouping ----------------------------------------

def test_build_groupings_crossed():
    """Two top-level Groupings (one repeat, one tile) are crossed: every
    level of one appears under every level of the other."""
    g = build_groupings(
        1000,
        Grouping('id1', n_levels=50, cycle='tile'),
        Grouping('id3', n_levels=20, cycle='repeat'),
    )
    assert np.unique(g['id1']).size == 50
    assert np.unique(g['id3']).size == 20
    # crossed: every id3 level contains all 50 id1 levels
    for p in range(20):
        assert np.unique(g['id1'][g['id3'] == p]).size == 50


def test_build_groupings_nested():
    """Nested factor: each child level lives in exactly one parent level,
    and total children = parent_levels * n_per_parent."""
    g = build_groupings(
        1000,
        Grouping('id3', n_levels=20),
        Nested('id2', parent='id3', n_per_parent=2),
    )
    assert np.unique(g['id2']).size == 40
    for j in np.unique(g['id2']):
        parents = np.unique(g['id3'][g['id2'] == j])
        assert parents.size == 1


def test_full_crossed_nested_fit():
    """End-to-end on the user's target shape: (1+x3+x4|id1) + (1+x5|id2)
    + (1|id3) with id1 crossed and id2 nested in id3."""
    n_obs = 600
    g = build_groupings(
        n_obs,
        Grouping('id1', n_levels=30, cycle='tile'),
        Grouping('id3', n_levels=15, cycle='repeat'),
        Nested('id2', parent='id3', n_per_parent=2),
    )
    G_id1 = np.diag([1.0, 0.5, 0.4]) + 0.05
    G_id1 = (G_id1 + G_id1.T) * 0.5
    np.fill_diagonal(G_id1, np.diag(G_id1) + 0.4)
    spec = SimSpec(
        n_obs=n_obs, response='y', fe_formula='1 + x1 + x2 + x3',
        beta=np.array([0.0, 0.5, -0.3, 1.0]),
        ranef=[
            RanefSpec(re_formula='1 + x3 + x4', group_var='id1',
                      G=G_id1, membership=g['id1']),
            RanefSpec(re_formula='1 + x5', group_var='id2',
                      G=np.array([[1.0, 0.2], [0.2, 0.5]]),
                      membership=g['id2']),
            RanefSpec(re_formula='1', group_var='id3',
                      G=np.array([[0.6]]), membership=g['id3']),
        ],
        resid_var=0.5,
        cov_spec=CovariateSpec(cont_vars=['x1', 'x2', 'x3', 'x4', 'x5'],
                               mean=np.zeros(5), cov=np.eye(5)),
    )
    sim = MixedModelSim(spec, np.random.default_rng(0))
    # Z columns: id1 (30*3=90) + id2 (30*2=60) + id3 (15*1=15) = 165
    assert sim.Z.shape == (n_obs, 30 * 3 + 30 * 2 + 15)
    # theta_true: vech blocks 6 + 3 + 1 + 1 resid = 11
    assert sim.theta_true.size == 11
    y, _ = sim.draw()
    model = sim.to_lmm(y)
    g_an = model.gradient(sim.theta_true, reml=True)
    g_fd = fo_fc_cd(lambda t: model.loglike(t, reml=True), sim.theta_true)
    assert np.max(np.abs(g_an - g_fd)) < 1e-4
    theta_hat, opt = fit_simulation(model, reml=True)
    assert opt.success


if __name__ == '__main__':
    import inspect, sys
    fns = [v for k, v in globals().items() if k.startswith('test_') and callable(v)]
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
