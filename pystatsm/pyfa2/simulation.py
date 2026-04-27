import numpy as np
import scipy as sp
from ..utilities.random import r_lkj


class FactorModel:

    def __init__(self, L, Phi, psi):
        self.L = np.asarray(L)
        self.Phi = np.asarray(Phi)
        self.psi = np.diag(psi) if np.ndim(psi) == 2 else np.asarray(psi)
        self.p, self.m = self.L.shape
        self.Sigma = np.matmul(np.matmul(self.L, self.Phi), self.L.T)
        self.Sigma[np.diag_indices(self.p)] += self.psi

    def sample(self, n, rng=None):
        rng = np.random.default_rng() if rng is None else rng
        return rng.multivariate_normal(np.zeros(self.p), self.Sigma, size=n)


def generate_loadings(p, m, low=0.4, high=0.9, random=False, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    L = np.zeros((p, m))
    for j, rows in enumerate(np.array_split(np.arange(p), m)):
        if random:
            u = np.sort(rng.uniform(low, high, size=len(rows)))[::-1]
        else:
            u = np.linspace(high, low, len(rows))
        L[rows, j] = u
    return L


def generate_factor_corr(m, rho=0.5, random=False, eta=2.0, rng=None):
    if random:
        return r_lkj(n=1, dim=m, eta=eta, rng=rng)
    Phi = rho ** sp.linalg.toeplitz(np.arange(m))
    s = (-1) ** np.arange(m).reshape(-1, 1)
    return s * Phi * s.T


def generate_uniquenesses(p, low=0.3, high=0.7, random=True, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    if random:
        return rng.uniform(low, high, size=p)
    return np.linspace(low, high, p)
