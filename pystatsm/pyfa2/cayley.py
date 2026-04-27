import numpy as np
import scipy as sp
from ..utilities.indexing_utils import tril_indices


def vec_to_skew(theta, m):
    rows, cols = tril_indices(m, k=-1)
    S = np.zeros(theta.shape[:-1] + (m, m), dtype=theta.dtype)
    S[..., rows, cols] = theta
    S[..., cols, rows] = -theta
    return S


def skew_to_vec(S):
    rows, cols = tril_indices(S.shape[-1], k=-1)
    return S[..., rows, cols]


def cayley(S):
    m = S.shape[-1]
    I = np.eye(m, dtype=S.dtype)
    M = I + S
    return 2.0 * np.linalg.solve(M, np.broadcast_to(I, M.shape).copy()) - I

cayley_inverse = cayley


def vec_to_rot(theta, m):
    return cayley(vec_to_skew(theta, m))


def rot_to_vec(Q):
    return skew_to_vec(cayley_inverse(Q))


def constraint_factor(theta, m):
    return sp.linalg.lu_factor(np.eye(m) + S)
