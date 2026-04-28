import numpy as np
import tqdm
from ..utilities.linalg_operations import _invec
from .simulation import (FactorModel, generate_loadings,
                         generate_factor_corr, generate_uniquenesses)
from .likelihood import MLEstimator
from .identification import RotationIdentification
from .criterion import GCFCriterion
from .rotation import OrthoRotation, ObliqueRotation
from .inference import param_cov
from .alignment import align


def unpack_se(se, free_mask, layout, phi_free):
    se_full = np.zeros(layout.nt)
    se_full[free_mask] = se
    L_se = _invec(se_full[layout.ixl],layout.p, layout.m)
    Phi_se = np.zeros((layout.m, layout.m))
    if phi_free:
        Phi_se[layout._row_i, layout._col_j] = se_full[layout.ixs]
        Phi_se[layout._col_j, layout._row_i] = se_full[layout.ixs]
    psi_se = se_full[layout.ixr]
    return L_se, Phi_se, psi_se


def alignment_matrix(layout, free_mask, perm, signs):
    free_idx = np.where(free_mask)[0]
    n_free = free_idx.size
    M = np.zeros((n_free, n_free))
    for j in range(n_free):
        e = np.zeros(layout.nt)
        e[free_idx[j]] = 1.0
        L, Phi, psi = layout.unpack(e)
        L_a = L[:, perm] * signs
        s_col = signs.reshape(-1, 1)
        s_row = signs.reshape(1, -1)
        Phi_a = s_col * Phi[np.ix_(perm, perm)] * s_row
        M[:, j] = layout.pack(L_a, Phi_a, psi)[free_mask]
    return M


def run(p, m, n_obs, n_rep, rotation_kind='ortho', seed=0):
    rng = np.random.default_rng(seed)
    L_true = generate_loadings(p, m, rng=rng)
    L_true = L_true + L_true[:, np.r_[-1, np.arange(m - 1)]] / 8
    Phi_true = np.eye(m) if rotation_kind == 'ortho' else generate_factor_corr(m)
    psi_true = generate_uniquenesses(p, rng=rng)
    model = FactorModel(L_true, Phi_true, psi_true)

    crit = GCFCriterion('varimax', p, m)
    rot = OrthoRotation(m) if rotation_kind == 'ortho' else ObliqueRotation(m)
    ident = RotationIdentification(rot, crit, solver='gpa')
    free_mask = ident.free_mask()
    layout = ident.layout
    n_free = int(free_mask.sum())

    est_pop = MLEstimator(model.Sigma, m)
    Lambda_pop = est_pop.loadings_from_psi(psi_true)
    fit_pop = ident.fit(Lambda_pop)
    L_true, Phi_true = fit_pop['L'], fit_pop['Phi']
    theta_true = layout.pack(L_true, Phi_true, psi_true)[free_mask]

    thetas = np.zeros((n_rep, n_free))
    Vs = np.zeros((n_rep, n_free, n_free))
    L_ests = np.zeros((n_rep, p, m))
    Phi_ests = np.zeros((n_rep, m, m))
    psi_ests = np.zeros((n_rep, p))

    for r in tqdm.tqdm(range(n_rep)):
        X = model.sample(n_obs, rng=rng)
        S = np.cov(X, rowvar=False, bias=True)
        est = MLEstimator(S, m)
        out = est.fit_psi()
        psi = out['psi']
        fit = ident.fit(out['Lambda'])
        L_fit, Phi_fit = fit['L'], fit['Phi']
        theta = est.layout.pack(L_fit, Phi_fit, psi)
        H = est.hessian(theta)
        C = ident.d_constraint(theta)
        V = param_cov(H, C, free_mask, n_obs)
        # align: determine perm/signs from L, then transform both theta and V
        perm, signs = align(L_fit, L_true)
        L_a = L_fit[:, perm] * signs
        s_col = signs.reshape(-1, 1)
        s_row = signs.reshape(1, -1)
        Phi_a = s_col * Phi_fit[np.ix_(perm, perm)] * s_row
        theta_a = est.layout.pack(L_a, Phi_a, psi)
        M = alignment_matrix(est.layout, free_mask, perm, signs)
        thetas[r] = theta_a[free_mask]
        Vs[r] = np.matmul(np.matmul(M, V, M.T))
        L_ests[r], Phi_ests[r], psi_ests[r] = L_a, Phi_a, psi

    # ---- theta-level diagnostics --------------------------------------------
    theta_dev = thetas - theta_true                 # (n_rep, n_free)
    theta_bias = theta_dev.mean(axis=0)             # (n_free,)
    V_emp = np.cov(thetas.T, ddof=1)                # (n_free, n_free)
    V_ase = Vs.mean(axis=0)                         # (n_free, n_free)
    se_emp = np.sqrt(np.diag(V_emp))
    se_ase = np.sqrt(np.diag(V_ase))
    se_ratio = se_emp / se_ase                      # ~ 1 if SE is calibrated

    # z-statistics per rep (using that rep's analytical SE)
    se_per_rep = np.sqrt(np.diagonal(Vs, axis1=1, axis2=2))   # (n_rep, n_free)
    z = theta_dev / se_per_rep                      # should be ~ N(0, 1)
    coverage = np.mean(np.abs(z) < 1.96, axis=0)    # per-param 95% coverage

    # Mahalanobis / chi^2 using mean analytical cov
    V_ase_inv = np.linalg.inv(V_ase)
    mahal = np.einsum('ri,ij,rj->r', theta_dev, V_ase_inv, theta_dev)

    # Full-matrix discrepancy
    fro_ratio = np.linalg.norm(V_emp - V_ase) / np.linalg.norm(V_ase)

    def block_summary(est, true, name):
        bias = est.mean(axis=0) - true
        emp = est.std(axis=0, ddof=1)
        return (np.max(np.abs(bias)), np.mean(emp))


    if rotation_kind == 'oblique':
        ri, ci = layout._row_i, layout._col_j
        Phi_est_stril = Phi_ests[:, ri, ci]
        Phi_true_stril = Phi_true[ri, ci]



    return {'thetas': thetas, 'Vs': Vs, 'theta_true': theta_true,
            'V_emp': V_emp, 'V_ase': V_ase, 'z': z, 'coverage': coverage,
            'mahal': mahal, "L_ests":L_ests, "Phi_ests":Phi_ests, "psi_ests":psi_ests}

