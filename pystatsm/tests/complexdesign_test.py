import numpy as np
import pandas as pd
import pytest

from pystatsm.utilities.complexdesign import (
    SampleDesign, IndexLayout, design_sandwich,
    Population, LinearPopulation, BinomialPopulation,
    FitResult, WLSAdapter, GLMAdapter,
    ReplicationResult, replicate,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _balanced_frame(n_str=4, n_psu=3, n_obs_per_psu=5, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_str):
        for g in range(n_psu):
            for _ in range(n_obs_per_psu):
                rows.append((s, g, rng.normal(), rng.normal()))
    df = pd.DataFrame(rows, columns=["str", "psu", "x1", "x2"])
    df["w"] = 1.0
    return df


def _hand_coded_meat(df, X, n_str, n_psu):
    M = np.zeros((X.shape[1], X.shape[1]))
    for s in range(n_str):
        Ts = []
        for g in range(n_psu):
            mask = (df["str"] == s) & (df["psu"] == g)
            Ts.append(X[mask.to_numpy()].sum(axis=0))
        T = np.stack(Ts)
        Tc = T - T.mean(axis=0)
        ssf = n_psu / (n_psu - 1)
        M += np.dot(Tc.T, Tc) * ssf
    return M


def _linear_pop(intra=0.4, seed=42):
    return LinearPopulation(
        response="y", x_vars=["x1", "x2"], rhs_formula="x1 + x2",
        beta=np.array([0.5, 1.0, -0.5]),
        n_strata=5, n_psu_per_stratum=6, n_obs_per_psu=12,
        intra_cluster_var=intra, resid_var=1.0,
        rng=np.random.default_rng(seed),
    )


def _binomial_pop(intra=0.0, seed=13):
    return BinomialPopulation(
        response="y", x_vars=["x1", "x2"], rhs_formula="x1 + x2",
        beta=np.array([-0.2, 0.8, -0.4]),
        n_strata=5, n_psu_per_stratum=8, n_obs_per_psu=15,
        intra_cluster_var=intra,
        rng=np.random.default_rng(seed),
    )


def _factory(df):
    return SampleDesign(df, strata="strata", psuind="psu", weight="w")


# ---------------------------------------------------------------------------
# SampleDesign / kernel tests
# ---------------------------------------------------------------------------

def test_meat_matches_hand_coded():
    df = _balanced_frame()
    des = SampleDesign(df, "str", "psu", "w")
    X = df[["x1", "x2"]].to_numpy()
    M = des.meat(X)
    M_ref = _hand_coded_meat(df, X, n_str=4, n_psu=3)
    assert np.allclose(M, M_ref, atol=1e-12)


def test_meat_shape_error():
    df = _balanced_frame()
    des = SampleDesign(df, "str", "psu", "w")
    X = df[["x1", "x2"]].to_numpy()
    with pytest.raises(ValueError, match="U must have shape"):
        des.meat(X[:5])


def test_singleton_aggregate_two_collapse():
    df = _balanced_frame()
    df.loc[df["str"] == 0, "psu"] = 0
    df.loc[df["str"] == 1, "psu"] = 0
    des = SampleDesign(df, "str", "psu", "w", singleton="aggregate")
    assert des.layout.n_psu_per_str.tolist() == [3, 3, 2]
    assert np.all(des.layout.ssf > 0)


def test_singleton_one_survives_zeros_ssf():
    df = _balanced_frame()
    df.loc[df["str"] == 0, "psu"] = 0
    des = SampleDesign(df, "str", "psu", "w", singleton="aggregate")
    assert des.layout.n_psu_per_str.tolist() == [3, 3, 3, 1]
    assert des.layout.ssf[-1] == 0.0
    M = des.meat(des.df[["x1", "x2"]].to_numpy())
    assert np.all(np.isfinite(M))


def test_singleton_error_policy_raises():
    df = _balanced_frame()
    df.loc[df["str"] == 0, "psu"] = 0
    with pytest.raises(ValueError, match="singleton PSUs"):
        SampleDesign(df, "str", "psu", "w", singleton="error")


def test_singleton_bad_policy_raises():
    df = _balanced_frame()
    with pytest.raises(ValueError, match="unknown singleton policy"):
        SampleDesign(df, "str", "psu", "w", singleton="bogus")


def test_subset_zeros_weights_preserves_layout():
    df = _balanced_frame()
    des = SampleDesign(df, "str", "psu", "w")
    mask = np.arange(des.n) >= 10
    sub = des.subset(mask)
    assert sub.layout is des.layout
    assert sub.n == des.n
    assert (sub.w == 0).sum() == 10
    assert np.array_equal(sub.w[10:], des.w[10:])


def test_subset_chains_preserve_zeros():
    df = _balanced_frame()
    des = SampleDesign(df, "str", "psu", "w")
    mask1 = np.arange(des.n) >= 10
    mask2 = np.arange(des.n) < 50
    sub = des.subset(mask1).subset(mask2)
    assert (sub.w == 0).sum() == 10 + (des.n - 50)


def test_subset_shape_error():
    df = _balanced_frame()
    des = SampleDesign(df, "str", "psu", "w")
    with pytest.raises(ValueError, match="mask must have shape"):
        des.subset(np.zeros(des.n - 1, dtype=bool))


def test_default_strata_psu_columns():
    df = pd.DataFrame({"x": np.arange(10.0)})
    des = SampleDesign(df)
    assert des.n_str == 1
    assert des.n_grp == 10
    # Singleton stratum -> ssf should be zero (not inf/nan).
    assert np.all(np.isfinite(des.layout.ssf))


def test_design_sandwich_algebra():
    rng = np.random.default_rng(0)
    df = _balanced_frame()
    des = SampleDesign(df, "str", "psu", "w")
    X = df[["x1", "x2"]].to_numpy()
    A = rng.normal(size=(2, 2))
    A = np.dot(A, A.T)
    V = design_sandwich(A, X, des)
    M = des.meat(X)
    assert np.allclose(V, np.dot(A, np.dot(M, A)))


def test_index_layout_is_frozen():
    df = _balanced_frame()
    des = SampleDesign(df, "str", "psu", "w")
    assert isinstance(des.layout, IndexLayout)
    with pytest.raises((AttributeError, Exception)):
        des.layout.n_str = 99


# ---------------------------------------------------------------------------
# Population tests
# ---------------------------------------------------------------------------

def test_population_base_raises_unimplemented():
    p = Population()
    with pytest.raises(NotImplementedError):
        _ = p.theta_true
    with pytest.raises(NotImplementedError):
        p.draw()


def test_linear_population_shape_and_columns():
    pop = _linear_pop()
    df = pop.draw()
    assert df.shape[0] == pop.n_obs
    for col in ["x1", "x2", "strata", "psu", "y", "w"]:
        assert col in df.columns
    assert pop.theta_true.shape == (3,)


def test_linear_population_parameter_names():
    pop = _linear_pop()
    names = pop.parameter_names
    assert names == ["Intercept", "x1", "x2"]


def test_linear_population_beta_dim_check():
    with pytest.raises(ValueError, match="beta has size"):
        LinearPopulation(
            response="y", x_vars=["x1", "x2"], rhs_formula="x1 + x2",
            beta=np.zeros(2),  # wrong: needs intercept
            n_strata=2, n_psu_per_stratum=2, n_obs_per_psu=2,
        )


def test_linear_population_psu_layout_balanced():
    pop = _linear_pop()
    df = pop.draw()
    counts = df.groupby(["strata", "psu"]).size().unique()
    assert counts.tolist() == [pop.n_obs_per_psu]


def test_binomial_population_y_is_binary():
    pop = _binomial_pop(intra=0.0)
    df = pop.draw()
    assert set(np.unique(df["y"])).issubset({0.0, 1.0})


def test_population_draw_with_external_rng_reproducible():
    pop = _linear_pop()
    df_a = pop.draw(rng=np.random.default_rng(123))
    df_b = pop.draw(rng=np.random.default_rng(123))
    pd.testing.assert_frame_equal(df_a, df_b)


# ---------------------------------------------------------------------------
# Adapter / FitResult tests
# ---------------------------------------------------------------------------

def test_wls_adapter_contract_shapes():
    pop = _linear_pop()
    df = pop.draw()
    des = _factory(df)
    fit = WLSAdapter("y ~ x1 + x2").fit(des)
    assert isinstance(fit, FitResult)
    p = pop.theta_true.size
    assert fit.params.shape == (p,)
    assert fit.bread.shape == (p, p)
    assert fit.score_i.shape == (des.n, p)


def test_wls_adapter_recovers_beta_to_within_se():
    pop = _linear_pop(intra=0.0)
    df = pop.draw(rng=np.random.default_rng(0))
    des = _factory(df)
    fit = WLSAdapter("y ~ x1 + x2").fit(des)
    V = design_sandwich(fit.bread, fit.score_i, des)
    se = np.sqrt(np.diag(V))
    z = (fit.params - pop.theta_true) / se
    assert np.all(np.abs(z) < 4.0)


def test_glm_gaussian_matches_wls_sandwich():
    pop = _linear_pop(intra=0.4, seed=7)
    df = pop.draw()
    des = _factory(df)
    wls = WLSAdapter("y ~ x1 + x2").fit(des)
    glm = GLMAdapter("y ~ x1 + x2").fit(des)
    V_wls = design_sandwich(wls.bread, wls.score_i, des)
    V_glm = design_sandwich(glm.bread, glm.score_i, des)
    assert np.allclose(wls.params, glm.params, atol=1e-6)
    assert np.allclose(V_wls, V_glm, atol=1e-8)


# ---------------------------------------------------------------------------
# Replication harness tests
# ---------------------------------------------------------------------------

def test_replicate_returns_correct_shapes():
    pop = _linear_pop(intra=0.0)
    adapter = WLSAdapter("y ~ x1 + x2")
    n_rep = 5
    res = replicate(pop, _factory, adapter, n_rep,
                    rng=np.random.default_rng(0), progress=False)
    p = pop.theta_true.size
    assert isinstance(res, ReplicationResult)
    assert res.params.shape == (n_rep, p)
    assert res.Vs.shape == (n_rep, p, p)
    assert res.theta_true.shape == (p,)
    assert res.parameter_names == ["Intercept", "x1", "x2"]


def test_replication_result_summary_columns():
    pop = _linear_pop(intra=0.0)
    adapter = WLSAdapter("y ~ x1 + x2")
    res = replicate(pop, _factory, adapter, 10,
                    rng=np.random.default_rng(0), progress=False)
    summary = res.summary()
    expected = {"true", "mean", "bias", "se_emp", "se_ana", "ratio", "coverage"}
    assert expected.issubset(set(summary.columns))
    assert list(summary.index) == ["Intercept", "x1", "x2"]


def test_replication_result_z_scores_match_manual():
    pop = _linear_pop(intra=0.0)
    res = replicate(pop, _factory, WLSAdapter("y ~ x1 + x2"), 4,
                    rng=np.random.default_rng(0), progress=False)
    z = res.z_scores()
    se = np.sqrt(np.diagonal(res.Vs, axis1=1, axis2=2))
    assert np.allclose(z, (res.params - res.theta_true) / se)


# ---------------------------------------------------------------------------
# Slow recovery tests (gated behind --runslow)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_recovery_iid_linear_coverage():
    pop = _linear_pop(intra=0.0, seed=42)
    res = replicate(pop, _factory, WLSAdapter("y ~ x1 + x2"), 600,
                    rng=np.random.default_rng(1), progress=False)
    cov = res.coverage()
    assert np.all(np.abs(cov - 0.95) < 0.04)
    assert 0.92 < res.se_ratio().mean() < 1.08
    assert np.max(np.abs(res.bias())) < 0.02


@pytest.mark.slow
def test_recovery_clustered_linear_se_ratio():
    pop = _linear_pop(intra=0.4, seed=42)
    res = replicate(pop, _factory, WLSAdapter("y ~ x1 + x2"), 600,
                    rng=np.random.default_rng(0), progress=False)
    # With G=30 PSUs the cluster-robust sandwich has known small-sample bias,
    # so coverage runs slightly under nominal but the calibration metric
    # (empirical / analytical SE) must be near 1.
    assert 0.92 < res.se_ratio().mean() < 1.08


@pytest.mark.slow
def test_recovery_iid_binomial_coverage():
    pop = _binomial_pop(intra=0.0, seed=13)
    adapter = GLMAdapter("y ~ x1 + x2",
                         family=__import__("pystatsm.pyglm2.families",
                                           fromlist=["Binomial"]).Binomial,
                         opt_kws={"options": {"verbose": 0}})
    res = replicate(pop, _factory, adapter, 200,
                    rng=np.random.default_rng(99), progress=False)
    assert np.all(np.abs(res.coverage() - 0.95) < 0.06)
    assert 0.90 < res.se_ratio().mean() < 1.10
