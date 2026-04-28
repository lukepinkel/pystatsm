import numpy as np
from scipy.optimize import linear_sum_assignment


def align(L_est, L_target, normalize=False):
    M = np.matmul(L_target.T, L_est)
    if normalize:
        nt = np.linalg.norm(L_target, axis=0).reshape(-1, 1)
        ne = np.linalg.norm(L_est, axis=0).reshape(1, -1)
        M = M / (nt * ne + 1e-30)
    _, perm = linear_sum_assignment(-np.abs(M))
    signs = np.sign(np.sum(L_target * L_est[:, perm], axis=0))
    signs[signs == 0] = 1.0
    return perm, signs


def apply(L, perm, signs):
    return L[:, perm] * signs


def apply_phi(Phi, perm, signs):
    s_col = signs.reshape(-1, 1)
    s_row = signs.reshape(1, -1)
    return s_col * Phi[np.ix_(perm, perm)] * s_row


def apply_se(L_se, perm):
    return L_se[:, perm]


def apply_phi_se(Phi_se, perm):
    return Phi_se[np.ix_(perm, perm)]


def align_model(L_est, L_target, Phi=None, L_se=None, Phi_se=None, normalize=False):
    perm, signs = align(L_est, L_target, normalize=normalize)
    out = {'L': apply(L_est, perm, signs), 'perm': perm, 'signs': signs}
    if Phi is not None:
        out['Phi'] = apply_phi(Phi, perm, signs)
    if L_se is not None:
        out['L_se'] = apply_se(L_se, perm)
    if Phi_se is not None:
        out['Phi_se'] = apply_phi_se(Phi_se, perm)
    return out
