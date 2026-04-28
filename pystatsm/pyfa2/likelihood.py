import numpy as np
import scipy as sp
from functools import cached_property
from .layout import ParamLayout
from ..utilities.linalg_operations import _vec, _vecl, _vech


class _Workspace:
    """Lazy cache of derived quantities at a fixed theta. Lifetime is the
    caller's scope; nothing here is shared across thetas. Only what gets
    accessed is materialized."""

    def __init__(self, est, theta):
        self.est = est
        self.theta = theta

    @cached_property
    def parts(self):
        return self.est._sigma_parts(self.theta)

    @cached_property
    def L(self):
        return self.parts[0]

    @cached_property
    def Phi(self):
        return self.parts[1]

    @cached_property
    def psi(self):
        return self.parts[2]

    @cached_property
    def LPhi(self):
        return self.parts[3]

    @cached_property
    def Sigma(self):
        return self.parts[4]

    @cached_property
    def Sinv(self):
        return np.linalg.inv(self.Sigma)

    @cached_property
    def dsigma(self):
        return self.est._dsigma_inner(self)

    @cached_property
    def M(self):
        return np.matmul(np.matmul(self.Sinv, self.dsigma), self.Sinv)

    def F(self):
        return self.est._F_inner(self)

    def grad(self):
        return self.est._grad_inner(self)

    def hessian(self):
        return self.est._hessian_inner(self)

    def score_jacobian(self):
        return self.est._score_jacobian_inner(self)

    def meat(self, V_S=None, n_obs=None):
        return self.est._meat_inner(self, V_S, n_obs)


class MLEstimator:

    def __init__(self, S, m):
        self.S = np.asarray(S)
        self.p = self.S.shape[0]
        self.m = m
        self.r = self.p - m
        self.layout = ParamLayout(self.p, m)
        self._Svals, self._Svecs = np.linalg.eigh(self.S)

    def workspace(self, theta):
        return _Workspace(self, theta)

    # ---- concentrated-psi path ---------------------------------------------

    def loglike_psi(self, psi):
        s = 1.0 / np.sqrt(psi.reshape(-1,1))
        u = np.linalg.eigvalsh(s.T * self.S * s)[:self.r]
        return np.sum(u - np.log(u) - 1)

    def grad_psi(self, psi):
        s = 1.0 / np.sqrt(psi.reshape(-1,1))
        u, V = np.linalg.eigh(s.T * self.S * s)
        return ((1 - u[:self.r]) * V[:, :self.r] ** 2).sum(axis=1) / psi

    def hess_psi(self, psi):
        p, r = self.p, self.r
        s = 1.0 / np.sqrt(psi.reshape(-1, 1))
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
        inv_psi = (1.0 / psi).reshape(-1, 1)
        return inv_psi * B * inv_psi.T

    def loadings_from_psi(self, psi):
        s = 1.0 / np.sqrt(psi.reshape(-1, 1))
        u, V = np.linalg.eigh(s.T * self.S * s)
        w = np.sqrt(u[-self.m:] - 1)
        return np.sqrt(psi.reshape(-1, 1)) * V[:, -self.m:] * w

    def _psi_init(self):
        L0 = self._Svecs[:, -self.m:] * np.sqrt(self._Svals[-self.m:])
        return np.diag(self.S - np.matmul(L0, L0.T))

    def loglike_exp(self, rho):
        return self.loglike_psi(np.exp(rho))

    def grad_exp(self, rho):
        psi = np.exp(rho)
        return psi * self.grad_psi(psi)

    def hess_exp(self, rho):
        psi = np.exp(rho)
        dF = self.grad_psi(psi)
        H = self.hess_psi(psi)
        psi_col = psi.reshape(-1, 1)
        out = psi_col.T * H * psi_col
        out[np.diag_indices_from(out)] += psi * dF
        return out

    def fit_psi(self, psi_init=None, **opt_kws):
        psi_init = self._psi_init() if psi_init is None else psi_init
        rho_init = np.log(psi_init)
        opt_kws.setdefault('method', 'trust-constr')
        opt = sp.optimize.minimize(self.loglike_exp, rho_init,
                                   jac=self.grad_exp, hess=self.hess_exp, **opt_kws)
        psi = np.exp(opt.x)
        return {'psi': psi, 'Lambda': self.loadings_from_psi(psi), 'opt': opt}

    # ---- full-theta path: thin wrappers around _Workspace ------------------

    def _sigma_parts(self, theta):
        L, Phi, psi = self.layout.unpack(theta)
        LPhi = np.matmul(L, Phi)
        Sigma = np.matmul(LPhi, L.T)
        Sigma[np.diag_indices(self.p)] += psi
        return L, Phi, psi, LPhi, Sigma

    def sigma(self, theta):
        return self.workspace(theta).Sigma

    def F(self, theta):
        return self.workspace(theta).F()

    def grad(self, theta):
        return self.workspace(theta).grad()

    def hessian(self, theta):
        return self.workspace(theta).hessian()

    def dsigma(self, theta):
        return self.workspace(theta).dsigma

    def score_jacobian(self, theta):
        return self.workspace(theta).score_jacobian()

    def meat(self, theta, V_S=None, n_obs=None):
        return self.workspace(theta).meat(V_S, n_obs)

    def _F_inner(self, ws):
        _, lnd = np.linalg.slogdet(ws.Sigma)
        tr = np.trace(np.linalg.solve(ws.Sigma, self.S))
        return lnd + tr

    def _grad_inner(self, ws):
        VRV = np.matmul(np.matmul(ws.Sinv, ws.Sigma - self.S), ws.Sinv)
        gL = 2 * _vec(np.matmul(VRV, ws.LPhi))
        gPhi = 2 * _vecl(np.matmul(np.matmul(ws.L.T, VRV), ws.L))
        gPsi = np.diag(VRV)
        g = np.empty(self.layout.nt)
        g[self.layout.ixl] = gL
        g[self.layout.ixs] = gPhi
        g[self.layout.ixr] = gPsi
        return g

    def _dsigma_inner(self, ws):
        p, m, nt = self.p, self.m, self.layout.nt
        ixs, ixr = self.layout.ixs, self.layout.ixr
        row_i, col_j = self.layout._row_i, self.layout._col_j
        L, LPhi = ws.L, ws.LPhi
        a_idx = np.arange(p)
        G3 = np.zeros((nt, p, p))
        for i in range(m):
            slab = G3[p * i:p * (i + 1)]
            slab[a_idx, a_idx, :] = LPhi[:, i]
            slab[a_idx, :, a_idx] += LPhi[:, i]
        ns = self.layout.ns
        Lr = L[:, row_i].T
        Lc = L[:, col_j].T
        Lr_col = Lr.reshape(ns, p, 1)
        Lc_col = Lc.reshape(ns, p, 1)
        Lr_row = Lr.reshape(ns, 1, p)
        Lc_row = Lc.reshape(ns, 1, p)
        G3[ixs] = Lr_col * Lc_row + Lc_col * Lr_row
        G3[ixr, a_idx, a_idx] = 1.0
        return G3

    def _hessian_inner(self, ws):
        p, m, nt = self.p, self.m, self.layout.nt
        ixl, ixs = self.layout.ixl, self.layout.ixs
        nl, ns = self.layout.nl, self.layout.ns
        row_i, col_j = self.layout._row_i, self.layout._col_j

        Sdiff = self.S - ws.Sigma
        VSdiff = np.matmul(ws.Sinv, Sdiff)
        W1G = ws.M
        W2G = np.matmul(VSdiff, W1G)
        W_comb = W1G + 2.0 * W2G

        G_flat = ws.dsigma.reshape(nt, -1)
        W_flat = W_comb.reshape(nt, -1)
        H_main = np.matmul(G_flat, W_flat.T)

        Y = np.matmul(VSdiff, ws.Sinv)
        H3_LL = np.kron(ws.Phi, 2.0 * Y)
        YL = np.matmul(Y, ws.L)
        H3_LPhi_3d = np.zeros((p, m, ns))
        ks = np.arange(ns)
        H3_LPhi_3d[:, row_i, ks] = 2.0 * YL[:, col_j]
        H3_LPhi_3d[:, col_j, ks] = 2.0 * YL[:, row_i]
        H3_LPhi = H3_LPhi_3d.reshape(nl, ns, order='F')

        H = H_main.copy()
        ixl_col = ixl.reshape(-1, 1)
        ixs_col = ixs.reshape(-1, 1)
        H[ixl_col, ixl] -= H3_LL
        H[ixl_col, ixs] -= H3_LPhi
        H[ixs_col, ixl] -= H3_LPhi.T
        return H

    def _score_jacobian_inner(self, ws):
        factor = 2.0 - np.eye(self.p)
        return _vech(-ws.M * factor)

    def _meat_inner(self, ws, V_S, n_obs):
        if V_S is None:
            if n_obs is None:
                raise ValueError("supply V_S or n_obs (for analytical Gaussian)")
            G_flat = ws.dsigma.reshape(self.layout.nt, -1)
            W1G_flat = ws.M.reshape(self.layout.nt, -1)
            return (2.0 / n_obs) * np.matmul(G_flat, W1G_flat.T)
        B = self._score_jacobian_inner(ws)
        return np.matmul(np.matmul(B, V_S), B.T)
