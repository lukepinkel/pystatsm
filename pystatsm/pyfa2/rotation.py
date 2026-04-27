import numpy as np
import scipy as sp
from . import cayley as _cayley

from ..utilities.linalg_operations import _vecl, _vec_ndg_mask, _vec_tril_mask, _vec_ndg
from ..utilities.indexing_utils import tril_indices

class OrthoRotation:
    phi_free = False

    def __init__(self, m):
        self.m = m
        self.n_c = m * (m - 1) // 2
        self._vec_tril_mask = _vec_tril_mask(m, k=-1)

    def rotated_loadings(self, A, T):
        return np.matmul(A, T)

    def implied_corr(self, T):
        return np.eye(self.m)

    def grad(self, A, T, dQL):
        return np.matmul(A.T, dQL)

    def constraint_project(self, T, G):
        M = np.matmul(T.T , G)
        Gp =  G - np.matmul(T, ((M + M.T) * 0.5))
        return Gp

    def constraint_retract(self, X):
        U, _, V = np.linalg.svd(X, full_matrices=False)
        UV = np.matmul(U, V)
        return UV

    @property
    def constraint_dim(self):
        return self.n_c

    def unconstrained_to_rotation(self, theta):
        return _cayley.vec_to_rot(theta, self.m)

    def rotation_to_unconstrained(self, T):
        return _cayley.rot_to_vec(T)

    def d_rotation(self, theta, H):
        lu = _cayley.constraint_factor(theta, self.m)
        X = sp.linalg.lu_solve(lu, H.T)
        K = sp.linalg.lu_solve(lu, X.T, trans=1).T
        return 2.0 * _cayley.skew_to_vec(K - K.T)

    def n_constraints(self):
        return self.n_c

    def constraint(self, L, Phi, crit):
        M = np.matmul(L.T, crit.dQ(L))
        return _vecl(M - M.T)

    def d_constraint(self, L, Phi, crit):
        p, m = L.shape[0], self.m
        G = crit.dQ(L)
        rr, cc = np.triu_indices(m, 1)
        row_i, col_j = cc, rr
        n_c = row_i.size
        U = np.zeros((p * m, 2 * n_c))
        for k in range(n_c):
            i, j = row_i[k], col_j[k]
            U[j * p:(j + 1) * p, 2 * k]     = L[:, i]
            U[i * p:(i + 1) * p, 2 * k + 1] = L[:, j]
        Y = crit.d2Q_apply(L, U)
        dCdL = np.empty((n_c, p * m))
        for k in range(n_c):
            i, j = row_i[k], col_j[k]
            T = Y[:, 2 * k].reshape(p, m, order='F') - Y[:, 2 * k + 1].reshape(p, m, order='F')
            T[:, i] += G[:, j]
            T[:, j] -= G[:, i]
            dCdL[k, :] = T.reshape(-1, order='F')
        dCdPhi = np.zeros((n_c, m * (m - 1) // 2))
        return dCdL, dCdPhi


class ObliqueRotation:
    phi_free = True

    def __init__(self, m):
        self.m = m
        self.n_c = m * (m - 1)
        self._odiag_mask = _vec_ndg_mask(m)

    def rotated_loadings(self, A, T):
        return np.matmul(A, np.linalg.inv(T).T)

    def implied_corr(self, T):
        return np.matmul(T.T, T)

    def grad(self, A, T, dQL):
        Tinv = np.linalg.inv(T)
        L = np.matmul(A, Tinv.T)
        return -np.linalg.multi_dot([L.T, dQL, Tinv]).T

    def constraint_project(self, T, G):
        return G - np.matmul(T, np.diag((T * G).sum(axis=0)))

    def constraint_retract(self, X):
        return X / np.sqrt(np.sum(X * X, axis=0, keepdims=True))

    @property
    def constraint_dim(self):
        raise NotImplementedError("oblique parameterization not implemented yet")

    def unconstrained_to_rotation(self, theta):
        raise NotImplementedError("oblique parameterization not implemented yet")

    def d_rotation(self, theta, H):
        raise NotImplementedError("oblique parameterization not implemented yet")

    def n_constraints(self):
        return self.n_c

    def constraint(self, L, Phi, crit):
        V = np.linalg.inv(Phi)
        C = np.matmul(np.matmul(L.T, crit.dQ(L)), V)
        return _vec_ndg(C)

    def d_constraint(self, L, Phi, crit):
        p, m = L.shape[0], self.m
        V = np.linalg.inv(Phi)
        G = crit.dQ(L)
        GV = np.matmul(G, V)
        ij = [(i, j) for j in range(m) for i in range(m) if i != j]
        n_c = len(ij)
        U = np.empty((p * m, n_c))
        for k, (i, j) in enumerate(ij):
            U[:, k] = np.outer(L[:, i], V[:, j]).reshape(-1, order='F')
        Y = crit.d2Q_apply(L, U)
        dCdL = np.empty((n_c, p * m))
        for k, (i, j) in enumerate(ij):
            T = Y[:, k].reshape(p, m, order='F')
            T[:, i] += GV[:, j]
            dCdL[k, :] = T.reshape(-1, order='F')
        dCdPhi = self._dM_dPhi_stril(L, G, V)[self._odiag_mask]
        return dCdL, dCdPhi

    def _dM_dPhi_stril(self, L, G, V):
        m = self.m
        M = np.matmul(np.matmul(L.T, G), V)
        ri, ci = tril_indices(m, -1)
        cols = np.empty((m * m, ri.size))
        for k, (x, y) in enumerate(zip(ri, ci)):
            J = -np.outer(M[:, x], V[y, :]) - np.outer(M[:, y], V[x, :])
            cols[:, k] = J.reshape(-1, order='F')
        return cols
