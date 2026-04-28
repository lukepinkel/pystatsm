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
    def parts(self):    return self.est._sigma_parts(self.theta)
    @cached_property
    def L(self):        return self.parts[0]
    @cached_property
    def Phi(self):      return self.parts[1]
    @cached_property
    def psi(self):      return self.parts[2]
    @cached_property
    def LPhi(self):     return self.parts[3]
    @cached_property
    def Sigma(self):    return self.parts[4]
    @cached_property
    def Sinv(self):     return np.linalg.inv(self.Sigma)
    @cached_property
    def dsigma(self):   return self.est._dsigma_inner(self)   # (nt, p, p)
    @cached_property
    def M(self):
        # Sinv @ dSigma_k @ Sinv, batched. (p, p) broadcasts against (nt, p, p).
        return np.matmul(np.matmul(self.Sinv, self.dsigma), self.Sinv)

    def F(self):                            return self.est._F_inner(self)
    def grad(self):                         return self.est._grad_inner(self)
    def hessian(self):                      return self.est._hessian_inner(self)
    def score_jacobian(self):               return self.est._score_jacobian_inner(self)
    def meat(self, V_S=None, n_obs=None):   return self.est._meat_inner(self, V_S, n_obs)


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
        s = 1.0 / np.sqrt(psi[:, None])
        u = np.linalg.eigvalsh(s.T * self.S * s)[:self.r]
        return np.sum(u - np.log(u) - 1)

    def grad_psi(self, psi):
        s = 1.0 / np.sqrt(psi[:, None])
        u, V = np.linalg.eigh(s.T * self.S * s)
        return ((1 - u[:self.r]) * V[:, :self.r] ** 2).sum(axis=1) / psi

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
        return np.sqrt(psi[:, None]) * V[:, -self.m:] * w

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

    # ---- full-theta path: thin wrappers around _Workspace ------------------

    def _sigma_parts(self, theta):
        L, Phi, psi = self.layout.unpack(theta)
        LPhi = np.matmul(L, Phi)
        Sigma = np.matmul(LPhi, L.T)
        Sigma[np.diag_indices(self.p)] += psi
        return L, Phi, psi, LPhi, Sigma

    def sigma(self, theta):                 return self.workspace(theta).Sigma
    def F(self, theta):                     return self.workspace(theta).F()
    def grad(self, theta):                  return self.workspace(theta).grad()
    def hessian(self, theta):               return self.workspace(theta).hessian()
    def dsigma(self, theta):                return self.workspace(theta).dsigma
    def score_jacobian(self, theta):        return self.workspace(theta).score_jacobian()
    def meat(self, theta, V_S=None, n_obs=None):
        return self.workspace(theta).meat(V_S, n_obs)

    # ---- _*_inner helpers consume a workspace ------------------------------

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
        # (nt, p, p) layout: G3[k] = dSigma/dtheta_k. Building directly in this
        # layout makes the (nt, p^2) reshape later a free view.
        p, m, nt = self.p, self.m, self.layout.nt
        ixs, ixr = self.layout.ixs, self.layout.ixr
        row_i, col_j = self.layout._row_i, self.layout._col_j
        L, LPhi = ws.L, ws.LPhi
        G3 = np.zeros((nt, p, p))
        I_p = np.eye(p)
        for i in range(m):
            # k = p*i + a, a in [0, p): G3[k, b, c] = δ_{b=a} LPhi[c, i] + δ_{c=a} LPhi[b, i].
            G3[p * i:p * (i + 1)] = (np.einsum('ab,c->abc', I_p, LPhi[:, i])
                                     + np.einsum('ac,b->abc', I_p, LPhi[:, i]))
        # Phi block: G3[k_phi, b, c] = L[b, i] L[c, j] + L[b, j] L[c, i].
        G3[ixs] = (np.einsum('bk,ck->kbc', L[:, row_i], L[:, col_j])
                   + np.einsum('bk,ck->kbc', L[:, col_j], L[:, row_i]))
        # Psi block: G3[k_psi, a, a] = 1.
        a_idx = np.arange(p)
        G3[ixr, a_idx, a_idx] = 1.0
        return G3

    def _hessian_inner(self, ws):
        # H = 2(H1 + H2) - H3 with W1 = kron(V, V), W2 = kron(V, V Sdiff V).
        # Algebraic refactor: W2G_k = V Sdiff V G_k V = V Sdiff (V G_k V) = V Sdiff W1G_k,
        # so once W1G is in hand W2G is one batched matmul. Then 2(H1+H2) = G^T (W1+2W2) G
        # collapses into a single (nt, p^2) x (p^2, nt) inner product on (W1G + 2 W2G).
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

        # H3 block: only LL and LPhi nonzero. Y = V Sdiff V; D_T = 2 Y.
        Y = np.matmul(VSdiff, ws.Sinv)
        H3_LL = np.kron(ws.Phi, 2.0 * Y)
        YL = np.matmul(Y, ws.L)
        H3_LPhi_3d = np.zeros((p, m, ns))
        ks = np.arange(ns)
        H3_LPhi_3d[:, row_i, ks] = 2.0 * YL[:, col_j]
        H3_LPhi_3d[:, col_j, ks] = 2.0 * YL[:, row_i]
        H3_LPhi = H3_LPhi_3d.reshape(nl, ns, order='F')

        H = H_main.copy()
        ixl_col = ixl[:, None]
        ixs_col = ixs[:, None]
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
