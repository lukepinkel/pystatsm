import numpy as np
import pandas as pd

from pystatsm.utilities.complexdesign.sample_design import (
    SampleDesign, design_sandwich)
from pystatsm.utilities.complexdesign.repweights import (
    jackknife_multipliers, bootstrap_multipliers, replicate_variance)
from pystatsm.utilities.complexdesign.descriptives import (
    survey_total, survey_mean, survey_mean_by, survey_total_by)
from pystatsm.utilities.complexdesign.survey_glm import SurveyGLM
from pystatsm.pyglm2.families import Gaussian, Binomial

N_PSU_PER_STR = np.array([2, 3, 4, 2, 5, 3, 2, 4])


def make_survey_frame(seed=1234):
    # Unbalanced stratified-cluster frame with a cluster random effect and
    # weights correlated with it, so model-based and design-based SEs
    # separate.  psu holds within-stratum labels (R svydesign nest=TRUE).
    # The same frame, byte-for-byte, backs the R golden values in
    # complexdesign_rcheck.py --- keep the generator in sync with it.
    rng = np.random.default_rng(seed)
    n_str = N_PSU_PER_STR.size
    n_grp = N_PSU_PER_STR.sum()
    psu_sizes = rng.integers(15, 40, size=n_grp)
    n = int(psu_sizes.sum())
    stratum_of_psu = np.repeat(np.arange(n_str), N_PSU_PER_STR)
    strata = np.repeat(stratum_of_psu, psu_sizes)
    psu_g = np.repeat(np.arange(n_grp), psu_sizes)
    local = np.concatenate([np.arange(k) for k in N_PSU_PER_STR])
    psu = np.repeat(local, psu_sizes)
    u = rng.normal(scale=0.6, size=n_grp)
    x1 = rng.normal(size=n)
    x2 = rng.integers(0, 3, size=n)
    b2 = np.array([0.0, 0.7, -0.4])
    eta = 0.5 + 0.5 * x1 + b2[x2] + u[psu_g]
    yg = eta + rng.normal(scale=1.0, size=n)
    pr = 1.0 / (1.0 + np.exp(-(eta - 0.5)))
    yb = rng.binomial(1, pr).astype(float)
    w = rng.gamma(2.0, 25.0, size=n) * np.exp(0.2 * u[psu_g])
    dom = (rng.random(n) < 0.55).astype(int)
    return pd.DataFrame(dict(strata=strata, psu=psu, w=w, x1=x1, x2=x2,
                             yg=yg, yb=yb, dom=dom))


def _design(seed=1234, **kws):
    df = make_survey_frame(seed)
    return SampleDesign(df, "strata", "psu", "w", **kws), df


def test_layout_and_index_arrays():
    des, df = _design()
    L = des.layout
    assert L.n_str == N_PSU_PER_STR.size
    assert L.n_grp == N_PSU_PER_STR.sum()
    assert np.array_equal(L.n_psu_per_str, N_PSU_PER_STR)
    assert des.row_psu.shape == (des.n,)
    # row_psu labels each sorted row with its PSU slot; the pointer arrays
    # must recover the observed group sizes exactly
    assert np.array_equal(np.bincount(des.row_psu), np.diff(L.ind_psu))
    assert np.array_equal(np.bincount(des.psu_stratum), N_PSU_PER_STR)
    key = des.df["strata"].to_numpy() * 1000 + des.df["psu"].to_numpy()
    _, inv = np.unique(key, return_inverse=True)
    assert np.array_equal(inv, des.row_psu)
    assert des.degf() == L.n_grp - L.n_str


def test_meat_matches_two_level_loop():
    des, df = _design()
    rng = np.random.default_rng(0)
    U = rng.normal(size=(des.n, 3))
    M = des.meat(U)
    strata = des.df["strata"].to_numpy()
    psu = des.df["psu"].to_numpy()
    M_ref = np.zeros((3, 3))
    for h in np.unique(strata):
        rows_h = strata == h
        labels = np.unique(psu[rows_h])
        n_h = labels.size
        T = np.stack([U[rows_h & (psu == i)].sum(axis=0) for i in labels])
        D = T - T.mean(axis=0)
        M_ref += np.dot(D.T, D) * n_h / (n_h - 1.0)
    assert np.allclose(M, M_ref, rtol=1e-12, atol=1e-12)


def test_jackknife_multiplier_structure():
    des, _ = _design()
    L = des.layout
    mult, rscales = jackknife_multipliers(L)
    assert np.allclose(np.diag(mult), 0.0)
    for h in range(L.n_str):
        g0, g1 = L.ind_str[h], L.ind_str[h + 1]
        n_h = g1 - g0
        blk = mult[g0:g1, g0:g1]
        off = blk[~np.eye(n_h, dtype=bool)]
        assert np.allclose(off, n_h / (n_h - 1.0))
        # rows for stratum h leave every other stratum untouched
        outside = np.ones(L.n_grp, dtype=bool)
        outside[g0:g1] = False
        assert np.allclose(mult[g0:g1][:, outside], 1.0)
        assert np.allclose(rscales[g0:g1], L.ssf[h] * ((n_h - 1) / n_h) ** 2)


def test_bootstrap_multiplier_structure():
    des, _ = _design()
    L = des.layout
    mult = bootstrap_multipliers(L, 200, rng=np.random.default_rng(3))
    assert mult.shape == (200, L.n_grp)
    # Rao-Wu-Yue with m_h = n_h - 1: counts sum to n_h - 1, so the PSU
    # multipliers of each stratum sum to exactly n_h in every replicate
    for h in range(L.n_str):
        g0, g1 = L.ind_str[h], L.ind_str[h + 1]
        n_h = g1 - g0
        assert np.allclose(mult[:, g0:g1].sum(axis=1), n_h)


def test_jackknife_total_equals_linearization():
    # Delete-one-PSU jackknife reproduces the stratified between-PSU
    # variance of a linear total algebraically, so the two estimators must
    # agree to rounding, not just asymptotically.
    des, _ = _design()
    _, V_lin = survey_total(des, ["yg", "x1"])
    _, V_jkn = survey_total(des, ["yg", "x1"], vcov="jackknife")
    assert np.allclose(V_lin, V_jkn, rtol=1e-10)
    _, V_jke = survey_total(des, ["yg", "x1"], vcov="jackknife",
                            center="estimate")
    assert np.allclose(V_lin, V_jke, rtol=1e-10)


def test_jackknife_total_equality_survives_fpc():
    # rscales carry (1 - f_h) through layout.ssf, so the identity holds
    # with a finite population correction too.
    fpc = np.array([0.1, 0.2, 0.4, 0.15, 0.5, 0.25, 0.1, 0.3])
    des, _ = _design(fpc=fpc)
    des0, _ = _design()
    _, V_lin = survey_total(des, "yg")
    _, V_jkn = survey_total(des, "yg", vcov="jackknife")
    assert np.allclose(V_lin, V_jkn, rtol=1e-10)
    # and each stratum contribution shrinks by exactly (1 - f_h): with a
    # single-stratum domain the ratio is recoverable in closed form
    values = des.df["strata"].to_numpy()
    for h in (0, 4):
        _, V_h = survey_total(des.subset(values == h), "yg")
        _, V_h0 = survey_total(des0.subset(values == h), "yg")
        assert np.allclose(V_h, (1.0 - fpc[h]) * V_h0, rtol=1e-10)


def test_mean_linearization_formula():
    des, _ = _design()
    x = des.df["yg"].to_numpy()
    w = des.w
    res, V = survey_mean(des, "yg")
    xbar = np.dot(w, x) / w.sum()
    assert np.allclose(res["mean"].to_numpy(), xbar, rtol=1e-12)
    z = (w * (x - xbar) / w.sum()).reshape(-1, 1)
    assert np.allclose(V, des.meat(z), rtol=1e-12)


def test_replicate_mean_vectorization():
    # The PSU-total shortcut must equal brute-force recomputation with
    # expanded replicate weights.
    des, _ = _design()
    X = des.df[["yg", "x1"]].to_numpy(dtype=np.float64)
    mult, rscales, scale = des.jackknife_replicates()
    w_rep = des.replicate_weights(mult)
    man = np.stack([np.dot(w_rep[r], X) / w_rep[r].sum()
                    for r in range(mult.shape[0])])
    starts = des.layout.ind_psu[:-1]
    tx = np.add.reduceat(X * des.w[:, None], starts, axis=0)
    tw = np.add.reduceat(des.w, starts)
    vec = mult.dot(tx) / mult.dot(tw)[:, None]
    assert np.allclose(man, vec, rtol=1e-12)
    V_man = replicate_variance(man, rscales, scale)
    _, V_jkn = survey_mean(des, ["yg", "x1"], vcov="jackknife")
    assert np.allclose(V_man, V_jkn, rtol=1e-12)


def test_bootstrap_mean_close_to_linearization():
    des, _ = _design()
    _, V_lin = survey_mean(des, ["yg", "x1"])
    _, V_boot = survey_mean(des, ["yg", "x1"], vcov="bootstrap", n_rep=4000,
                            rng=np.random.default_rng(7))
    se_l = np.sqrt(np.diag(V_lin))
    se_b = np.sqrt(np.diag(V_boot))
    assert np.all(np.abs(se_b / se_l - 1.0) < 0.05)


def test_domain_mean_point_estimate_and_layout():
    des, df = _design()
    mask = df["dom"].to_numpy() == 1
    sub = des.subset(mask)
    res, V = survey_mean(sub, "yg")
    w, x = des.w, des.df["yg"].to_numpy()
    m = des.df["dom"].to_numpy() == 1
    assert np.allclose(res["mean"].to_numpy(),
                       np.dot(w[m], x[m]) / w[m].sum(), rtol=1e-12)
    # the subset shares the parent layout so PSU/stratum structure is intact
    assert sub.layout is des.layout
    assert sub.degf() <= des.degf()
    assert np.sqrt(V[0, 0]) > 0


def test_by_group_totals_sum_to_full_total():
    des, _ = _design()
    res_by = survey_total_by(des, "yg", "x2")
    res, _ = survey_total(des, "yg")
    assert np.allclose(res_by["total"].sum(), res["total"].iloc[0],
                       rtol=1e-12)


def test_degf_domain_semantics():
    des, df = _design()
    # restrict to a single stratum: only its PSUs and 1 stratum remain
    values = df["strata"].to_numpy()
    sub = des.subset(values == 4)
    assert sub.degf() == N_PSU_PER_STR[4] - 1
    assert des.degf() == N_PSU_PER_STR.sum() - N_PSU_PER_STR.size


def test_replicate_variance_centering():
    rng = np.random.default_rng(11)
    theta = rng.normal(size=(40, 3))
    rscales = rng.uniform(0.5, 1.5, size=40)
    center = rng.normal(size=3)
    V0 = replicate_variance(theta, rscales, 0.5)
    d = theta - theta.mean(axis=0)
    assert np.allclose(V0, 0.5 * np.dot(d.T * rscales, d), rtol=1e-12)
    V1 = replicate_variance(theta, rscales, 0.5, center=center)
    d = theta - center
    assert np.allclose(V1, 0.5 * np.dot(d.T * rscales, d), rtol=1e-12)


def test_survey_glm_gaussian_equals_wls_sandwich():
    # Gaussian identity SurveyGLM is exactly the R-validated WLS pipeline:
    # beta = (X'WX)^{-1} X'Wy and V = A M A with U = X w r.
    des, _ = _design()
    m = SurveyGLM("yg ~ x1 + C(x2)", des, family=Gaussian).fit()
    X, y, w = m.model.X, m.model.y, des.w
    XtWX = np.dot(X.T, X * w[:, None])
    beta = np.linalg.solve(XtWX, np.dot(X.T, w * y))
    assert np.allclose(beta, m.params, rtol=1e-8, atol=1e-10)
    A = np.linalg.inv(XtWX)
    U = X * (w * (y - np.dot(X, beta)))[:, None]
    V = design_sandwich(A, U, des)
    assert np.allclose(V, m.vcov_linearized, rtol=1e-7, atol=1e-12)


def test_survey_glm_weight_scale_invariance():
    des, df = _design()
    df2 = df.copy()
    df2["w"] = df2["w"] * 1e-4
    des2 = SampleDesign(df2, "strata", "psu", "w")
    m1 = SurveyGLM("yb ~ x1 + C(x2)", des, family=Binomial).fit()
    m2 = SurveyGLM("yb ~ x1 + C(x2)", des2, family=Binomial).fit()
    assert np.allclose(m1.params, m2.params, rtol=1e-7, atol=1e-9)
    assert np.allclose(m1.vcov_linearized, m2.vcov_linearized,
                       rtol=1e-6, atol=1e-12)


def test_survey_glm_jackknife_and_bootstrap_close_to_linearized():
    des, _ = _design()
    m = SurveyGLM("yg ~ x1 + C(x2)", des, family=Gaussian)
    m.fit(vcov="jackknife", progress=False)
    se_j, se_l = m.params_se, np.sqrt(np.diag(m.vcov_linearized))
    assert np.all(np.abs(se_j / se_l - 1.0) < 0.05)
    mb = SurveyGLM("yg ~ x1 + C(x2)", des, family=Gaussian)
    mb.fit(vcov="bootstrap", n_rep=200, rng=np.random.default_rng(5),
           progress=False)
    se_b = mb.params_se
    assert np.all(np.abs(se_b / se_l - 1.0) < 0.15)
    assert np.all(np.linalg.eigvalsh(mb.params_cov) > 0)


def test_survey_glm_domain_runs_all_families():
    des, df = _design()
    sub = des.subset(df["dom"].to_numpy() == 1)
    for formula, fam in (("yg ~ x1", Gaussian), ("yb ~ x1", Binomial)):
        m = SurveyGLM(formula, sub, family=fam).fit(vcov="jackknife",
                                                    progress=False)
        assert np.all(np.isfinite(m.params_se))
        assert np.all(m.params_se > 0)
        assert m.ddf == sub.degf() - m.n_params + 1


def test_singleton_policies():
    des, df = _design()
    dfs = df.copy()
    # make strata 6 and 7 single-PSU strata (two of them, so the aggregate
    # policy has something to pool them into)
    dfs = dfs[~(dfs["strata"].isin([6, 7]) & (dfs["psu"] > 0))]
    dfs = dfs.reset_index(drop=True)
    try:
        SampleDesign(dfs, "strata", "psu", "w", singleton="error")
        raised = False
    except ValueError:
        raised = True
    assert raised
    agg = SampleDesign(dfs, "strata", "psu", "w", singleton="aggregate")
    assert np.all(agg.layout.n_psu_per_str > 1)
    keep = SampleDesign(dfs, "strata", "psu", "w", singleton="none")
    n_h = keep.layout.n_psu_per_str
    assert np.any(n_h == 1)
    assert np.all(keep.layout.ssf[n_h == 1] == 0.0)
    mult, rscales = jackknife_multipliers(keep.layout)
    lone = np.flatnonzero(np.repeat(n_h == 1, n_h))
    assert np.allclose(mult[lone], 1.0)
    assert np.allclose(rscales[lone], 0.0)


# ---------------------------------------------------------------------------
# Golden values from R survey 4.2.1 (R 4.3.3), generated on the seed-1234
# frame by complexdesign_rcheck.py.  Linearized quantities agreed with R to
# ~1e-15 (they are the same algebra), the binomial svyglm to 1.1e-11 in
# coefficients / 2.1e-7 in SEs, and the JKn svyglm SEs to 6.1e-9 (replicate
# optimizer noise); tolerances below leave headroom over the 12-digit
# storage quantization and cross-platform BLAS variation.
# ---------------------------------------------------------------------------

R_TOTAL_EST = np.array([20172.9897842, 2375.09970263])
R_TOTAL_SE = np.array([5451.18839896, 1029.88310707])
R_MEAN_EST = np.array([0.582472381151, 0.0685783314254])
R_MEAN_SE = np.array([0.138054709815, 0.028170602008])
R_DOMAIN_MEAN = np.array([0.615715477601, 0.134985206385])
R_BY_MEAN = np.array([0.603968518398, 1.20990349733, -0.0161257210311])
R_BY_SE = np.array([0.188053650408, 0.134188090716, 0.139526634464])
R_GLM_GAUSS_COEF = np.array([0.579980681812, 0.59319093888,
                             -0.613004986711, 0.373648715837])
R_GLM_GAUSS_SE = np.array([0.167985518722, 0.115342282103,
                           0.123208712415, 0.0512219034595])
R_GLM_BINOM_COEF = np.array([0.0793007761437, 0.459253734257,
                             -0.546831722987, 0.484297497095])
R_GLM_BINOM_SE = np.array([0.20728440929, 0.200115283891,
                           0.174263862811, 0.123721424928])
R_GLM_GAUSS_JKN_SE = np.array([0.16860946297, 0.116391663168,
                               0.124375757998, 0.051565094043])
R_MEAN_JKN_SE = 0.13832967954
R_FPC_TOTAL_SE = 4937.29333067
R_BOOT_MEAN_SE = 0.139709927727
R_BOOT_TOTAL_SE = 5467.42179882
R_GLM_DF_RESID = 14
GLM_LABELS = ["Intercept", "C(x2)[T.1]", "C(x2)[T.2]", "x1"]


def test_r_golden_descriptives():
    des, df = _design()
    rt, _ = survey_total(des, ["yg", "x1"])
    assert np.allclose(rt["total"].to_numpy(), R_TOTAL_EST, rtol=1e-9)
    assert np.allclose(rt["SE"].to_numpy(), R_TOTAL_SE, rtol=1e-9)
    rm, _ = survey_mean(des, ["yg", "x1"])
    assert np.allclose(rm["mean"].to_numpy(), R_MEAN_EST, rtol=1e-9)
    assert np.allclose(rm["SE"].to_numpy(), R_MEAN_SE, rtol=1e-9)
    rd, _ = survey_mean(des.subset(df["dom"].to_numpy() == 1), "yg")
    assert np.allclose([rd["mean"].iloc[0], rd["SE"].iloc[0]],
                       R_DOMAIN_MEAN, rtol=1e-9)
    rby = survey_mean_by(des, "yg", "x2")
    assert np.allclose(rby["mean"].to_numpy(), R_BY_MEAN, rtol=1e-9)
    assert np.allclose(rby["SE"].to_numpy(), R_BY_SE, rtol=1e-9)
    rj, _ = survey_mean(des, "yg", vcov="jackknife")
    assert np.allclose(rj["SE"].iloc[0], R_MEAN_JKN_SE, rtol=1e-9)


def test_r_golden_fpc_total():
    fpc = N_PSU_PER_STR / np.array([20, 15, 10, 8, 25, 12, 6, 40])
    des, _ = _design(fpc=fpc)
    rt, _ = survey_total(des, "yg")
    assert np.allclose(rt["SE"].iloc[0], R_FPC_TOTAL_SE, rtol=1e-9)


def test_r_golden_bootstrap_assembly():
    # Same Generator(42) multinomial stream as the replicate weights that
    # were exported to R survey by complexdesign_rcheck.py; pins both the
    # draw reproducibility and the svrVar-equivalent assembly.
    des, _ = _design()
    mult = bootstrap_multipliers(des.layout, 100,
                                 rng=np.random.default_rng(42))
    starts = des.layout.ind_psu[:-1]
    x = des.df["yg"].to_numpy()
    tx = np.add.reduceat(des.w * x, starts)
    tw = np.add.reduceat(des.w, starts)
    se_tot = np.sqrt(replicate_variance(mult.dot(tx), np.ones(100),
                                        0.01))[0, 0]
    se_mean = np.sqrt(replicate_variance(mult.dot(tx) / mult.dot(tw),
                                         np.ones(100), 0.01))[0, 0]
    assert np.allclose(se_tot, R_BOOT_TOTAL_SE, rtol=1e-9)
    assert np.allclose(se_mean, R_BOOT_MEAN_SE, rtol=1e-9)


def test_r_golden_glm_gaussian():
    des, _ = _design()
    m = SurveyGLM("yg ~ x1 + C(x2)", des, family=Gaussian).fit()
    assert m.param_labels == GLM_LABELS
    assert np.allclose(m.params, R_GLM_GAUSS_COEF, rtol=1e-8, atol=1e-10)
    assert np.allclose(m.params_se, R_GLM_GAUSS_SE, rtol=1e-8)
    assert m.ddf == R_GLM_DF_RESID
    mj = SurveyGLM("yg ~ x1 + C(x2)", des, family=Gaussian)
    mj.fit(vcov="jackknife", progress=False)
    assert np.allclose(mj.params_se, R_GLM_GAUSS_JKN_SE, rtol=1e-6)


def test_r_golden_glm_binomial():
    des, _ = _design()
    m = SurveyGLM("yb ~ x1 + C(x2)", des, family=Binomial).fit()
    assert np.allclose(m.params, R_GLM_BINOM_COEF, rtol=1e-6, atol=1e-8)
    assert np.allclose(m.params_se, R_GLM_BINOM_SE, rtol=1e-5)


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
