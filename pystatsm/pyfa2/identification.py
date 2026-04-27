import numpy as np
from . import solvers
from .layout import ParamLayout
from ..utilities.indexing_utils import tril_indices


_SOLVERS = {'gpa': solvers.gpa, 'cayley': solvers.cayley_solve}


class RotationIdentification:

    def __init__(self, rot, crit, solver='gpa', **solver_kw):
        self.rot = rot
        self.crit = crit
        self.solver = solver
        self.solver_kw = solver_kw
        self._solve = _SOLVERS[solver]
        self.p, self.m = crit.p, crit.m
        self.layout = ParamLayout(self.p, self.m)

    def fit(self, A, T0=None):
        T, info = self._solve(self.crit, self.rot, A, T0=T0, **self.solver_kw)
        L = self.rot.rotated_loadings(A, T)
        Phi = self.rot.implied_corr(T)
        return {'L': L, 'Phi': Phi, 'T': T, 'info': info}

    def n_constraints(self):
        return self.rot.n_constraints()

    def free_mask(self):
        mask = np.ones(self.layout.nt, dtype=bool)
        if not self.rot.phi_free:
            mask[self.layout.ixs] = False
        return mask

    def constraint(self, theta):
        L, Phi, _ = self.layout.unpack(theta)
        return self.rot.constraint(L, Phi, self.crit)

    def d_constraint(self, theta):
        L, Phi, _ = self.layout.unpack(theta)
        dCdL, dCdPhi = self.rot.d_constraint(L, Phi, self.crit)
        dCdPsi = np.zeros((dCdL.shape[0], self.layout.nr))
        return np.concatenate([dCdL, dCdPhi, dCdPsi], axis=1)


class CanonicalIdentification:
    def __init__(self, p, m):
        self.p, self.m = p, m
        self.n_c = m * (m - 1) // 2
        self.layout = ParamLayout(p, m)
        self._row_i, self._col_j = tril_indices(m, -1)

    def fit(self, A, Psi=None):
        psi_col = None if Psi is None else np.asarray(Psi).reshape(-1, 1)
        AD = A if psi_col is None else A / psi_col
        M = np.matmul(AD.T, A)
        vals, vecs = np.linalg.eigh(M)
        order = np.argsort(vals)[::-1]
        T = vecs[:, order]
        L = np.matmul(A, T)
        Phi = np.eye(self.m)
        return {'L': L, 'Phi': Phi, 'T': T, 'info': {'eigvals': vals[order]}}

    def n_constraints(self):
        return self.n_c

    def free_mask(self):
        mask = np.ones(self.layout.nt, dtype=bool)
        mask[self.layout.ixs] = False
        return mask

    def constraint(self, theta):
        L, _, psi = self.layout.unpack(theta)
        LD = L / psi[:, None]
        M = np.matmul(LD.T, L)
        return M[self._row_i, self._col_j]

    def d_constraint(self, theta):
        L, _, psi = self.layout.unpack(theta)
        nl, ns = self.layout.nl, self.layout.ns
        row_i, col_j, n_c = self._row_i, self._col_j, self.n_c
        p, m = self.p, self.m
        LD = L / psi[:, None]
        J = np.zeros((n_c, p, m))
        ks = np.arange(n_c)
        J[ks, :, row_i] = LD[:, col_j].T
        J[ks, :, col_j] = LD[:, row_i].T
        dCdL = J.transpose(0, 2, 1).reshape(n_c, nl)
        dCdPhi = np.zeros((n_c, ns))
        L_prod = L[:, row_i] * L[:, col_j]
        dCdPsi = -(L_prod / psi[:, None] ** 2).T
        return np.concatenate([dCdL, dCdPhi, dCdPsi], axis=1)
