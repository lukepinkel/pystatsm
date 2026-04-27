import numpy as np
import scipy as sp
from .layout import ParamLayout
from ..utilities.linalg_operations import _vec, _vecl, _vech


class MLEstimator:

    def __init__(self, S, m):
        self.S = np.asarray(S)
        self.p = self.S.shape[0]
        self.m = m
        self.r = self.p - m
        self.layout = ParamLayout(self.p, m)
        self._Svals, self._Svecs = np.linalg.eigh(self.S)

    def loglike_psi(self, psi):
        s = 1.0 / np.sqrt(psi[:, None])
        u = np.linalg.eigvalsh(s.T * self.S * s)[:self.r]
        f = np.sum(u - np.log(u) - 1)
        return f

    def grad_psi(self, psi):
        s = 1.0 / np.sqrt(psi[:, None])
        u, V = np.linalg.eigh(s.T * self.S * s)
        g = ((1 - u[:self.r]) * V[:, :self.r] ** 2).sum(axis=1) / psi
        return g

    def hess_psi(self, psi):
        p, r = self.p, self.r
        s = 1.0 / np.sqrt(psi[:, None])
        W = s.T * self.S * s
        u, V = np.linalg.eigh(W)
        B = np.zeros((p, p))
        for i in range(r):
            A = np.outer(V[:, i], V[:, i])
            b1 = (2 * u[i] - 1) * A
            diffs = u[i] - u
            inv = np.zeros_like(diffs)
            np.divide(1.0, diffs, out=inv, where=np.abs(diffs) > 1e-12)
            pinv_term = np.matmul(V * inv, V.T)
            b2 = 2 * u[i] * (u[i] - 1) * pinv_term
            B += (b1 + b2) * A
        inv_psi = 1.0 / psi
        return inv_psi[:, None] * B * inv_psi[None, :]

    def loadings_from_psi(self, psi):
        s = 1.0 / np.sqrt(psi[:, None])
        u, V = np.linalg.eigh(s.T * self.S * s)
        w = np.sqrt(u[-self.m:] - 1)
        A = np.sqrt(psi[:, None]) * V[:, -self.m:] * w
        return A

    def _psi_init(self):
        L0 = self._Svecs[:, -self.m:] * np.sqrt(self._Svals[-self.m:])
        return np.diag(self.S - np.matmul(L0, L0.T))

    def fit_psi(self, psi_init=None, **opt_kws):
        psi_init = self._psi_init() if psi_init is None else psi_init
        rho_init = np.log(psi_init)
        f = lambda r: self.loglike_psi(np.exp(r))
        g = lambda r: np.exp(r) * self.grad_psi(np.exp(r))
        def h(r):
            psi = np.exp(r)
            dF = self.grad_psi(psi)
            H = self.hess_psi(psi)
            return psi[:, None].T * H * psi[:, None] + np.diag(psi * dF)
        opt_kws.setdefault('method', 'trust-constr')
        opt = sp.optimize.minimize(f, rho_init, jac=g, hess=h, **opt_kws)
        psi = np.exp(opt.x)
        return {'psi': psi, 'Lambda': self.loadings_from_psi(psi), 'opt': opt}

    # ---- full-theta path ---------------------------------------------------

    def _sigma_parts(self, theta):
        L, Phi, psi = self.layout.unpack(theta)
        LPhi = np.matmul(L, Phi)
        Sigma = np.matmul(LPhi, L.T)
        Sigma[np.diag_indices(self.p)] += psi
        return L, Phi, psi, LPhi, Sigma

    def sigma(self, theta):
        return self._sigma_parts(theta)[4]

    def F(self, theta):
        Sigma = self.sigma(theta)
        _, lnd = np.linalg.slogdet(Sigma)
        tr = np.trace(np.linalg.solve(Sigma, self.S))
        return lnd + tr

    def grad(self, theta):
        L, Phi, psi, LPhi, Sigma = self._sigma_parts(theta)
        V = np.linalg.inv(Sigma)
        VRV = np.matmul(np.matmul(V, Sigma - self.S), V)
        gL = 2 * _vec(np.matmul(VRV, LPhi))
        gPhi = 2 * _vecl(np.matmul(np.matmul(L.T, VRV), L))
        gPsi = np.diag(VRV)
        g = np.empty(self.layout.nt)
        g[self.layout.ixl] = gL
        g[self.layout.ixs] = gPhi
        g[self.layout.ixr] = gPsi
        return g

    def dsigma(self, theta):
        L, Phi, psi, LPhi, Sigma = self._sigma_parts(theta)
        p, m = self.p, self.m
        ixl, ixs, ixr = self.layout.ixl, self.layout.ixs, self.layout.ixr
        row_i, col_j = self.layout._row_i, self.layout._col_j
        G3 = np.zeros((p, p, self.layout.nt))

        I_p = np.eye(p)
        for i in range(m):
            G3[:, :, p * i:p * (i + 1)] = (np.einsum('ba,c->bca', I_p, LPhi[:, i])
                                           + np.einsum('ca,b->bca', I_p, LPhi[:, i]))
        G3[:, :, ixs] = (np.einsum('bk,ck->bck', L[:, row_i], L[:, col_j])
                        + np.einsum('bk,ck->bck', L[:, col_j], L[:, row_i]))
        a_idx = np.arange(p)
        G3[a_idx, a_idx, ixr] = 1.0
        return G3

    def hessian(self, theta):
        L, Phi, psi, LPhi, Sigma = self._sigma_parts(theta)
        p, m, nt = self.p, self.m, self.layout.nt
        ixl, ixs = self.layout.ixl, self.layout.ixs
        row_i, col_j = self.layout._row_i, self.layout._col_j

        V = np.linalg.inv(Sigma)
        Sdiff = self.S - Sigma
        VSdiffV = np.matmul(np.matmul(V, Sdiff), V)

        G3 = self.dsigma(theta)
        G3T = np.transpose(G3, (2, 0, 1))

        W1G = np.matmul(np.matmul(V, G3T), V)
        W2G = np.matmul(np.matmul(VSdiffV, G3T), V)

        G_flat = G3T.reshape(nt, -1)
        W1G_flat = W1G.reshape(nt, -1)
        W2G_flat = W2G.reshape(nt, -1)
        H1 = 0.5 * np.matmul(G_flat, W1G_flat.T)
        H2 = np.matmul(G_flat, W2G_flat.T)

        Y = VSdiffV
        nl, ns = self.layout.nl, self.layout.ns

        H3_LL = np.kron(Phi, 2.0 * Y)

        YL = np.matmul(Y, L)
        H3_LPhi_3d = np.zeros((p, m, ns))
        ks = np.arange(ns)
        H3_LPhi_3d[:, row_i, ks] = 2.0 * YL[:, col_j]
        H3_LPhi_3d[:, col_j, ks] = 2.0 * YL[:, row_i]
        H3_LPhi = H3_LPhi_3d.reshape(nl, ns, order='F')

        H3 = np.zeros((nt, nt))
        ixl_col = ixl[:, None]
        ixs_col = ixs[:, None]
        H3[ixl_col, ixl] = H3_LL
        H3[ixl_col, ixs] = H3_LPhi
        H3[ixs_col, ixl] = H3_LPhi.T

        return 2.0 * (H1 + H2) - H3

    def score_jacobian(self, theta):
        L, Phi, psi, LPhi, Sigma = self._sigma_parts(theta)
        V = np.linalg.inv(Sigma)
        G3T = np.transpose(self.dsigma(theta), (2, 0, 1))
        M = np.matmul(np.matmul(V, G3T), V)
        factor = 2.0 - np.eye(self.p)  # 2 off-diag, 1 on diag
        return _vech(-M * factor)

    def meat(self, theta, V_S=None, n_obs=None):
        if V_S is None:
            L, Phi, psi, LPhi, Sigma = self._sigma_parts(theta)
            V = np.linalg.inv(Sigma)
            G3T = np.transpose(self.dsigma(theta), (2, 0, 1))
            W1G = np.matmul(np.matmul(V, G3T), V)
            G_flat = G3T.reshape(self.layout.nt, -1)
            W1G_flat = W1G.reshape(self.layout.nt, -1)
            return (2.0 / n_obs) * np.matmul(G_flat, W1G_flat.T)
        B = self.score_jacobian(theta)
        return np.matmul(np.matmul(B, V_S), B.T)
