"""Test suite for pyfa2. Each test is a plain `def test_*()` function with
asserts; covers layout, cayley, criterion, rotation, solvers, identification,
likelihood (concentrated-psi and full-theta paths), workspace caching,
inference, alignment, and simulation."""

import numpy as np

from pystatsm.pyfa2.layout import ParamLayout
from pystatsm.pyfa2 import cayley as _cay
from pystatsm.pyfa2.criterion import GCFCriterion, TargetCriterion
from pystatsm.pyfa2.rotation import OrthoRotation, ObliqueRotation
from pystatsm.pyfa2.solvers import GPA, CayleySolver, gpa, cayley_solve
from pystatsm.pyfa2.identification import RotationIdentification, CanonicalIdentification
from pystatsm.pyfa2.likelihood import MLEstimator
from pystatsm.pyfa2.inference import (augmented_hessian, bread, param_cov,
                                      sandwich_cov, se_from_cov, empirical_cov_vech_S)
from pystatsm.pyfa2.alignment import align, apply, apply_phi, align_model
from pystatsm.pyfa2.simulation import (FactorModel, generate_loadings,
                                       generate_factor_corr, generate_uniquenesses)


# ---------- finite-difference helpers --------------------------------------

def fd_grad(f, x, eps=1e-6):
    x = np.asarray(x, dtype=float)
    f0 = np.atleast_1d(f(x))
    g = np.zeros((f0.size, x.size))
    for i in range(x.size):
        xp = x.copy(); xp[i] += eps
        xm = x.copy(); xm[i] -= eps
        g[:, i] = (f(xp) - f(xm)) / (2 * eps)
    return g


def fd_jac_mat(f, X, eps=1e-6):
    X = np.asarray(X, dtype=float)
    f0 = np.atleast_1d(f(X))
    J = np.zeros((f0.size, X.size))
    flat = X.reshape(-1, order='F')
    for i in range(X.size):
        Xp = flat.copy(); Xp[i] += eps
        Xm = flat.copy(); Xm[i] -= eps
        J[:, i] = (f(Xp.reshape(X.shape, order='F'))
                   - f(Xm.reshape(X.shape, order='F'))) / (2 * eps)
    return J


# ---------- layout ----------------------------------------------------------

def test_layout_dims():
    layout = ParamLayout(p=10, m=3)
    assert layout.nl == 30
    assert layout.ns == 3
    assert layout.nr == 10
    assert layout.nt == 43
    assert np.array_equal(layout.ixl, np.arange(30))
    assert np.array_equal(layout.ixs, np.arange(30, 33))
    assert np.array_equal(layout.ixr, np.arange(33, 43))


def test_layout_pack_unpack_roundtrip():
    rng = np.random.default_rng(0)
    p, m = 8, 3
    layout = ParamLayout(p, m)
    L = rng.standard_normal((p, m))
    Phi = np.eye(m)
    ri, ci = layout._row_i, layout._col_j
    Phi[ri, ci] = rng.standard_normal(layout.ns)
    Phi[ci, ri] = Phi[ri, ci]
    psi = 0.3 + rng.random(p)
    theta = layout.pack(L, Phi, psi)
    L2, Phi2, psi2 = layout.unpack(theta)
    assert np.allclose(L, L2)
    assert np.allclose(Phi, Phi2)
    assert np.allclose(psi, psi2)
    # Diag of unpacked Phi is 1 (correlation convention)
    assert np.allclose(np.diag(Phi2), 1.0)


# ---------- cayley ----------------------------------------------------------

def test_cayley_in_SO_and_roundtrip():
    rng = np.random.default_rng(1)
    for m in (2, 3, 5, 8):
        theta = 0.2 * rng.standard_normal(m * (m - 1) // 2)
        Q = _cay.vec_to_rot(theta, m)
        assert np.allclose(Q.T @ Q, np.eye(m), atol=1e-10)
        assert np.isclose(np.linalg.det(Q), 1.0, atol=1e-10)
        theta2 = _cay.rot_to_vec(Q)
        assert np.allclose(theta, theta2, atol=1e-10)


# ---------- criterion -------------------------------------------------------

def test_gcf_grad_and_hessian_vs_fd():
    rng = np.random.default_rng(2)
    p, m = 12, 3
    L = rng.standard_normal((p, m)) * 0.3
    Q_flat = lambda v, c: c.Q(v.reshape(p, m, order='F'))
    for method in ('varimax', 'quartimax', 'equamax', 'parsimax'):
        crit = GCFCriterion(method, p, m)
        g_fd = fd_grad(lambda v: Q_flat(v, crit), L.reshape(-1, order='F')).ravel()
        g_an = crit.dQ(L).reshape(-1, order='F')
        assert np.allclose(g_an, g_fd, atol=1e-6)
        H_an = crit.d2Q(L)
        H_fd = fd_jac_mat(lambda X: crit.dQ(X).reshape(-1, order='F'), L)
        assert np.allclose(H_an, H_fd, atol=1e-6)


def test_gcf_d2Q_apply_matches_dense():
    rng = np.random.default_rng(3)
    p, m = 10, 3
    L = rng.standard_normal((p, m)) * 0.4
    crit = GCFCriterion('varimax', p, m)
    Y = rng.standard_normal((p * m, 5))
    HY = crit.d2Q_apply(L, Y)
    H = crit.d2Q(L)
    assert np.allclose(HY, H @ Y, atol=1e-10)


def test_target_criterion():
    rng = np.random.default_rng(4)
    p, m = 8, 3
    L = rng.standard_normal((p, m)) * 0.3
    H = rng.standard_normal((p, m))
    W = np.abs(rng.standard_normal((p, m))) + 0.1
    crit = TargetCriterion(H, W)
    g_fd = fd_grad(lambda v: crit.Q(v.reshape(p, m, order='F')),
                   L.reshape(-1, order='F')).ravel()
    g_an = crit.dQ(L).reshape(-1, order='F')
    assert np.allclose(g_an, g_fd, atol=1e-6)


# ---------- rotation --------------------------------------------------------

def test_ortho_constraint_retract_in_SO():
    # Stress with multiple seeds — for some inputs the bare SVD product lands
    # at det == -1 (in O(m) but not SO(m)); the retract must enforce det == +1.
    for seed in range(20):
        rng = np.random.default_rng(seed)
        m = 4
        rot = OrthoRotation(m)
        X = rng.standard_normal((m, m))
        T = rot.constraint_retract(X)
        assert np.allclose(T.T @ T, np.eye(m), atol=1e-10)
        assert np.isclose(np.linalg.det(T), 1.0, atol=1e-10)


def test_ortho_constraint_project_is_tangent():
    rng = np.random.default_rng(6)
    m = 4
    rot = OrthoRotation(m)
    T = rot.constraint_retract(rng.standard_normal((m, m)))
    G = rng.standard_normal((m, m))
    Gp = rot.constraint_project(T, G)
    # Tangent at T to SO(m): T'·Gp must be skew-symmetric.
    M = T.T @ Gp
    assert np.allclose(M + M.T, 0.0, atol=1e-10)


def test_oblique_constraint_retract_unit_cols():
    rng = np.random.default_rng(7)
    m = 4
    rot = ObliqueRotation(m)
    T = rot.constraint_retract(rng.standard_normal((m, m)))
    assert np.allclose(np.linalg.norm(T, axis=0), 1.0, atol=1e-10)


def test_rotation_d_constraint_vs_fd():
    rng = np.random.default_rng(8)
    p, m = 10, 3
    crit = GCFCriterion('varimax', p, m)
    L = rng.standard_normal((p, m)) * 0.3
    for cls in (OrthoRotation, ObliqueRotation):
        rot = cls(m)
        Phi = rot.implied_corr(rot.constraint_retract(rng.standard_normal((m, m))))
        dCdL, _ = rot.d_constraint(L, Phi, crit)
        dCdL_fd = fd_jac_mat(lambda X: rot.constraint(X, Phi, crit), L)
        assert np.allclose(dCdL, dCdL_fd, atol=1e-6)


# ---------- solvers ---------------------------------------------------------

def test_gpa_class_and_function_agree():
    rng = np.random.default_rng(9)
    p, m = 10, 3
    A = rng.standard_normal((p, m))
    crit = GCFCriterion('varimax', p, m)
    rot = OrthoRotation(m)
    T1, info1 = GPA(crit, rot).solve(A)
    T2, info2 = gpa(crit, rot, A)
    assert np.allclose(T1, T2)
    assert info1['f'] == info2['f']
    assert info1['grad_norm'] < 1e-5


def test_cayley_class_with_warm_start():
    rng = np.random.default_rng(10)
    p, m = 10, 3
    A = rng.standard_normal((p, m))
    crit = GCFCriterion('varimax', p, m)
    rot = OrthoRotation(m)
    T_cold, info_cold = CayleySolver(crit, rot).solve(A)
    T0 = rot.constraint_retract(rng.standard_normal((m, m)))
    T_warm, info_warm = CayleySolver(crit, rot).solve(A, T0=T0)
    # Each should produce a critical point in SO(m); varimax has multiple local
    # minima so we don't require Q-equality, just convergence + valid rotation.
    assert np.allclose(T_cold.T @ T_cold, np.eye(m), atol=1e-8)
    assert np.allclose(T_warm.T @ T_warm, np.eye(m), atol=1e-8)
    assert info_cold['opt'].success
    assert info_warm['opt'].success


# ---------- identification --------------------------------------------------

def test_rotation_identification_constraint_at_fit():
    rng = np.random.default_rng(11)
    p, m = 12, 3
    A = rng.standard_normal((p, m))
    crit = GCFCriterion('varimax', p, m)
    layout = ParamLayout(p, m)
    psi = 0.3 + rng.random(p)
    for cls in (OrthoRotation, ObliqueRotation):
        ident = RotationIdentification(cls(m), crit, solver='gpa')
        out = ident.fit(A)
        theta = layout.pack(out['L'], out['Phi'], psi)
        assert np.linalg.norm(ident.constraint(theta)) < 1e-5


def test_canonical_identification_constraint_at_fit():
    rng = np.random.default_rng(12)
    p, m = 12, 3
    A = rng.standard_normal((p, m))
    psi = 0.3 + rng.random(p)
    cid = CanonicalIdentification(p, m)
    out = cid.fit(A, Psi=psi)
    layout = ParamLayout(p, m)
    theta = layout.pack(out['L'], out['Phi'], psi)
    assert np.linalg.norm(cid.constraint(theta)) < 1e-10


def test_identification_d_constraint_vs_fd():
    rng = np.random.default_rng(13)
    p, m = 10, 3
    A = rng.standard_normal((p, m))
    psi = 0.3 + rng.random(p)
    crit = GCFCriterion('varimax', p, m)
    layout = ParamLayout(p, m)
    cases = [
        ('canon',   CanonicalIdentification(p, m)),
        ('ortho',   RotationIdentification(OrthoRotation(m), crit)),
        ('oblique', RotationIdentification(ObliqueRotation(m), crit)),
    ]
    for tag, ident in cases:
        if isinstance(ident, CanonicalIdentification):
            out = ident.fit(A, Psi=psi)
        else:
            out = ident.fit(A)
        theta = layout.pack(out['L'], out['Phi'], psi)
        dC = ident.d_constraint(theta)
        dC_fd = fd_grad(ident.constraint, theta)
        assert np.allclose(dC, dC_fd, atol=1e-6), tag


def test_identification_free_mask_shapes():
    p, m = 10, 3
    crit = GCFCriterion('varimax', p, m)
    nt = ParamLayout(p, m).nt
    canon = CanonicalIdentification(p, m).free_mask()
    ortho = RotationIdentification(OrthoRotation(m), crit).free_mask()
    oblique = RotationIdentification(ObliqueRotation(m), crit).free_mask()
    assert canon.size == nt and ortho.size == nt and oblique.size == nt
    assert canon.sum() == p * m + p           # Phi fixed
    assert ortho.sum() == p * m + p           # Phi fixed
    assert oblique.sum() == p * m + m * (m - 1) // 2 + p  # Phi free


# ---------- likelihood ------------------------------------------------------

def _make_data(p=15, m=3, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    L_true = rng.standard_normal((p, m)) * 0.6
    psi_true = 0.3 + rng.random(p)
    Sigma = L_true @ L_true.T + np.diag(psi_true)
    X = rng.multivariate_normal(np.zeros(p), Sigma, size=n)
    return X, np.cov(X, rowvar=False, bias=True), n, L_true, psi_true


def test_concentrated_psi_grad_hess_vs_fd():
    X, S, n, L_true, psi_true = _make_data()
    est = MLEstimator(S, m=3)
    psi = est._psi_init()
    g_fd = fd_grad(est.loglike_psi, psi).ravel()
    assert np.allclose(est.grad_psi(psi), g_fd, atol=1e-6)
    H_fd = fd_grad(est.grad_psi, psi)
    assert np.allclose(est.hess_psi(psi), H_fd, atol=1e-5)


def test_fit_psi_converges():
    X, S, n, L_true, psi_true = _make_data(n=5000)
    est = MLEstimator(S, m=3)
    out = est.fit_psi()
    assert out['opt'].success
    assert np.linalg.norm(est.grad_psi(out['psi'])) < 1e-4


def test_full_theta_grad_hessian_dsigma_score_vs_fd():
    rng = np.random.default_rng(14)
    p, m = 12, 3
    L = rng.standard_normal((p, m)) * 0.5
    psi = 0.4 + rng.random(p)
    S_arr = L @ L.T + np.diag(psi)
    est = MLEstimator(S_arr, m)
    theta = est.layout.pack(L, np.eye(m), psi) + 0.05 * rng.standard_normal(est.layout.nt)
    g_fd = fd_grad(est.F, theta).ravel()
    assert np.allclose(est.grad(theta), g_fd, atol=1e-6)
    H_fd = fd_grad(est.grad, theta)
    assert np.allclose(est.hessian(theta), 0.5 * (H_fd + H_fd.T), atol=1e-6)
    G3 = est.dsigma(theta)
    DGp = G3.transpose(1, 2, 0).reshape(p * p, est.layout.nt, order='F')
    DGp_fd = fd_grad(lambda th: est.sigma(th).reshape(-1, order='F'), theta)
    assert np.allclose(DGp, DGp_fd, atol=1e-7)


def test_score_jacobian_vs_fd_vech():
    from pystatsm.utilities.linalg_operations import _vech, _invech
    rng = np.random.default_rng(15)
    p, m = 10, 3
    L = rng.standard_normal((p, m)) * 0.5
    psi = 0.4 + rng.random(p)
    S_arr = L @ L.T + np.diag(psi)
    S_arr = 0.5 * (S_arr + S_arr.T)
    est = MLEstimator(S_arr, m)
    theta = est.layout.pack(L, np.eye(m), psi)
    B = est.score_jacobian(theta)
    q = p * (p + 1) // 2
    B_fd = np.zeros_like(B)
    eps = 1e-6
    vech_S = _vech(S_arr)
    for l in range(q):
        vp = vech_S.copy(); vp[l] += eps
        vm = vech_S.copy(); vm[l] -= eps
        Sp, Sm = _invech(vp), _invech(vm)
        gp = MLEstimator(Sp, m).grad(theta)
        gm = MLEstimator(Sm, m).grad(theta)
        B_fd[:, l] = (gp - gm) / (2 * eps)
    assert np.allclose(B, B_fd, atol=1e-7)


# ---------- workspace -------------------------------------------------------

def test_workspace_results_match_uncached():
    rng = np.random.default_rng(16)
    p, m = 10, 3
    L = rng.standard_normal((p, m)) * 0.5
    psi = 0.4 + rng.random(p)
    S_arr = L @ L.T + np.diag(psi)
    est = MLEstimator(S_arr, m)
    theta = est.layout.pack(L, np.eye(m), psi)
    ws = est.workspace(theta)
    assert np.allclose(ws.hessian(), est.hessian(theta))
    assert np.allclose(ws.grad(), est.grad(theta))
    assert np.allclose(ws.score_jacobian(), est.score_jacobian(theta))
    assert np.allclose(ws.meat(n_obs=1000), est.meat(theta, n_obs=1000))


# ---------- inference -------------------------------------------------------

def test_bread_param_cov_shapes():
    rng = np.random.default_rng(17)
    p, m = 10, 3
    L = rng.standard_normal((p, m)) * 0.5
    psi = 0.4 + rng.random(p)
    S_arr = L @ L.T + np.diag(psi)
    est = MLEstimator(S_arr, m)
    theta = est.layout.pack(L, np.eye(m), psi)
    crit = GCFCriterion('varimax', p, m)
    ident = RotationIdentification(OrthoRotation(m), crit)
    H = est.hessian(theta)
    C = ident.d_constraint(theta)
    fm = ident.free_mask()
    n_free = int(fm.sum())
    b = bread(H, C, fm)
    V = param_cov(H, C, fm, n_obs=500)
    assert b.shape == (n_free, n_free)
    assert V.shape == (n_free, n_free)
    assert np.all(np.diag(V) > 0)


def test_sandwich_matches_ml_at_gaussian():
    X, S, n, L_true, psi_true = _make_data()
    est = MLEstimator(S, m=3)
    out = est.fit_psi()
    psi, Lambda = out['psi'], out['Lambda']
    crit = GCFCriterion('varimax', est.p, est.m)
    ident = RotationIdentification(OrthoRotation(est.m), crit)
    f = ident.fit(Lambda)
    theta = est.layout.pack(f['L'], f['Phi'], psi)
    H = est.hessian(theta); C = ident.d_constraint(theta); fm = ident.free_mask()
    V_ml = param_cov(H, C, fm, n)
    J = est.meat(theta, n_obs=n)
    V_sw = sandwich_cov(H, C, J, fm)
    se_ml = se_from_cov(V_ml)
    se_sw = se_from_cov(V_sw)
    assert np.max(np.abs(se_ml - se_sw)) / np.max(se_ml) < 0.02


def test_empirical_cov_vech_S_symmetric_and_positive():
    rng = np.random.default_rng(18)
    p, n = 6, 500
    X = rng.standard_normal((n, p))
    V = empirical_cov_vech_S(X)
    q = p * (p + 1) // 2
    assert V.shape == (q, q)
    assert np.allclose(V, V.T)
    assert np.all(np.linalg.eigvalsh(V) > -1e-10)


# ---------- alignment -------------------------------------------------------

def test_align_identity_when_equal():
    rng = np.random.default_rng(19)
    L = rng.standard_normal((10, 3))
    perm, signs = align(L, L)
    assert np.array_equal(perm, np.arange(3))
    assert np.allclose(signs, 1.0)


def test_align_recovers_permutation_and_signs():
    rng = np.random.default_rng(20)
    L = rng.standard_normal((10, 4))
    perm_true = np.array([2, 0, 3, 1])
    signs_true = np.array([1.0, -1.0, 1.0, -1.0])
    L_perm = L[:, perm_true] * signs_true
    perm, signs = align(L_perm, L)
    L_back = apply(L_perm, perm, signs)
    assert np.allclose(L_back, L, atol=1e-12)


def test_align_model_propagates_through_phi():
    rng = np.random.default_rng(21)
    L = rng.standard_normal((10, 3))
    Phi = np.eye(3); Phi[0, 1] = Phi[1, 0] = 0.4
    perm_true = np.array([1, 0, 2])
    signs_true = np.array([-1.0, 1.0, 1.0])
    L_perm = L[:, perm_true] * signs_true
    Phi_perm = (signs_true.reshape(-1, 1) * Phi[np.ix_(perm_true, perm_true)]
                * signs_true.reshape(1, -1))
    out = align_model(L_perm, L, Phi=Phi_perm)
    assert np.allclose(out['L'], L, atol=1e-12)
    assert np.allclose(out['Phi'], Phi, atol=1e-12)


# ---------- simulation ------------------------------------------------------

def test_factor_model_sigma_is_lphi_lt_plus_psi():
    rng = np.random.default_rng(22)
    p, m = 8, 3
    L = generate_loadings(p, m, rng=rng)
    Phi = generate_factor_corr(m)
    psi = generate_uniquenesses(p, rng=rng)
    model = FactorModel(L, Phi, psi)
    expected = L @ Phi @ L.T + np.diag(psi)
    assert np.allclose(model.Sigma, expected)
    assert np.allclose(model.Sigma, model.Sigma.T)


def test_factor_model_sample_shape_and_mean():
    rng = np.random.default_rng(23)
    p, m, n = 6, 2, 5000
    L = generate_loadings(p, m, rng=rng)
    Phi = np.eye(m)
    psi = generate_uniquenesses(p, rng=rng)
    model = FactorModel(L, Phi, psi)
    X = model.sample(n, rng=rng)
    assert X.shape == (n, p)
    assert np.max(np.abs(X.mean(axis=0))) < 0.1
    S = np.cov(X, rowvar=False, bias=True)
    assert np.linalg.norm(S - model.Sigma) / np.linalg.norm(model.Sigma) < 0.1


# ---------- runner ----------------------------------------------------------

def main():
    tests = [v for k, v in globals().items() if k.startswith('test_') and callable(v)]
    fails = []
    for t in tests:
        try:
            t()
            print(f"  ok   {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL {t.__name__}: {e}")
            fails.append(t.__name__)
        except Exception as e:
            print(f"  ERR  {t.__name__}: {type(e).__name__}: {e}")
            fails.append(t.__name__)
    print(f"\n{len(tests) - len(fails)}/{len(tests)} passed")
    return 0 if not fails else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
