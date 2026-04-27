import numpy as np
from ..utilities.indexing_utils import tril_indices


class ParamLayout:
    def __init__(self, p, m):
        self.p, self.m = p, m
        self.nl = p * m
        self.ns = m * (m - 1) // 2
        self.nr = p
        self.nt = self.nl + self.ns + self.nr
        self.ixl = np.arange(self.nl)
        self.ixs = np.arange(self.nl, self.nl + self.ns)
        self.ixr = np.arange(self.nl + self.ns, self.nt)
        self._row_i, self._col_j = tril_indices(m, -1)

    def pack(self, L, Phi, Psi):
        theta = np.empty(self.nt)
        theta[self.ixl] = L.reshape(-1, order='F')
        theta[self.ixs] = Phi[self._row_i, self._col_j]
        theta[self.ixr] = np.diag(Psi) if np.ndim(Psi) == 2 else np.asarray(Psi)
        return theta

    def unpack(self, theta):
        L = theta[self.ixl].reshape(self.p, self.m, order='F')
        Phi = np.eye(self.m)
        Phi[self._row_i, self._col_j] = theta[self.ixs]
        Phi[self._col_j, self._row_i] = theta[self.ixs]
        psi = theta[self.ixr]
        return L, Phi, psi
