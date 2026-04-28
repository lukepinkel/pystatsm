import numpy as np
from ..utilities.linalg_operations import _vech

def augmented_hessian(H, dCdTheta):
    nt = H.shape[0]
    n_c = dCdTheta.shape[0]
    A = np.zeros((nt + n_c, nt + n_c))
    A[:nt, :nt] = H
    A[:nt, nt:] = dCdTheta.T
    A[nt:, :nt] = dCdTheta
    return A


def bread(H, dCdTheta, free_mask):
    nt = H.shape[0]
    n_c = dCdTheta.shape[0]
    A = augmented_hessian(H, dCdTheta)
    free = np.r_[np.where(free_mask)[0], nt + np.arange(n_c)]
    Ainv = np.linalg.inv(A[np.ix_(free, free)])
    n_free = free.size - n_c
    return Ainv[:n_free, :n_free]


def param_cov(H, dCdTheta, free_mask, n_obs):
    return bread(H, dCdTheta, free_mask) * (2.0 / n_obs)


def sandwich_cov(H, dCdTheta, J, free_mask):
    b = bread(H, dCdTheta, free_mask)
    free_idx = np.where(free_mask)[0]
    return np.matmul(np.matmul(b, J[np.ix_(free_idx, free_idx)]), b.T)


def se_from_cov(V):
    return np.sqrt(np.diag(V))


def empirical_cov_vech_S(X):
    # Skip the (n, p, p) outer: vech(x x')[(jx, ix)] = x[jx] * x[ix].
    n, p = X.shape
    ix, jx = np.triu_indices(p, k=0)
    Xc = X - X.mean(axis=0, keepdims=True)
    vechs = Xc[:, jx] * Xc[:, ix]
    centered = vechs - vechs.mean(axis=0, keepdims=True)
    return np.matmul(centered.T, centered) / (n * (n - 1))
