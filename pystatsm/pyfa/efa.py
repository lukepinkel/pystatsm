# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 03:35:53 2022

@author: lukepinkel
"""

import numpy as np
import scipy as sp
import scipy.stats
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ..utilities.linalg_operations import vec, invec, vech, vecl, invecl, eighs, mat_sqrt, inv_sqrt
from ..utilities.indexing_utils import vecl_inds
from ..utilities.func_utils import handle_default_kws
from ..utilities.special_mats import nmat, dmat, lmat
from ..utilities.data_utils import _check_type, cov, flip_signs, corr
from .fit_measures import srmr, lr_test, gfi, agfi
from .rotation import GeneralizedCrawfordFerguson, get_gcf_constants
from ..utilities.numerical_derivs import so_gc_cd


class FactorAnalysis(object):
   
    def __init__(self, X=None, S=None, n_obs=None, n_factors=None, 
                 rotation_method=None, rotation_type=None, **n_factor_kws):
        self._process_data(X, S, n_obs)
        self._get_n_factors(n_factors, **n_factor_kws)
        self._init_psi()
        self.p, self.m = self.n_vars, self.n_facs
        self.E = np.eye(self.n_vars)
        self.Im = np.eye(self.n_facs)
        self.Nm = nmat(self.n_facs).A
        self.Lm = lmat(self.n_facs).A
        self.Dm = dmat(self.n_facs).A
        self.Ip = np.eye(self.n_vars)
        self.Dp = dmat(self.n_vars).A
        self.Lp = lmat(self.n_vars).A
        self.Np = nmat(self.n_vars).A
        self.LpNp = np.dot(self.Lp, self.Np)
        self.d_inds = vec(self.Ip)==1
        self.l_inds = vecl_inds(self.n_facs)
        self.ndg_inds = vech(np.eye(self.m))!=1
        if rotation_method is not None:
            rotation_type = 'ortho' if rotation_type is None else rotation_type
            self._rotate = GeneralizedCrawfordFerguson(A=np.zeros((self.p, self.m)),
                                                       rotation_method=rotation_method,
                                                       rotation_type=rotation_type)
        else:
            self._rotate = None
        self.rotation_method, self.rotation_type = rotation_method, rotation_type
        rc = np.arange(self.p)*self.p+np.arange(self.p), np.arange(self.p)
        self.Dpsi = sp.sparse.csc_matrix((np.ones(self.p), rc), shape=(self.p**2, self.p))
        self._make_param_indices()
        
    def _process_data(self, X, S, n_obs):
        given_x = X is not None
        given_s = S is not None
        given_n = n_obs is not None
        if given_x and not given_s:
            X, cols, inds, _is_pd = _check_type(X)
            S = cov(X)
            n_obs, n_vars = X.shape
        if not given_x and given_s and given_n:
            S, cols, _, _is_pd = _check_type(S)
            n_vars = S.shape[0]
            inds = np.arange(n_obs)
        u, V = eighs(S)
        V = flip_signs(V)
        self.X, self.cols, self.inds, self._is_pd = X, cols, inds, _is_pd
        self.S, self.V, self.u = S, V, u
        self.cols, self.inds, self._is_pd = cols, inds, _is_pd
        self.n_obs, self.n_vars = n_obs, n_vars
        
    def _get_n_factors(self, n_factors, proportion=0.6, eigmin=1.0):
        if type(n_factors) is str:
            if n_factors == 'proportion':
                n_factors = np.sum((self.u.cumsum()/self.u.sum())<proportion)+1
            elif n_factors == 'eigmin':
                n_factors = np.sum(self.u>eigmin)
        elif type(n_factors) in [float, int]:
            n_factors = int(n_factors)
        self.n_facs = self.n_factors = n_factors
        
    def _init_psi(self):
        self.pix = np.arange(self.n_vars*self.n_facs, self.n_vars*self.n_facs+self.n_vars)
        L = self.V[:, :self.n_facs]
        psi = np.diag(self.S - np.dot(L, L.T))
        self.psi_init = psi
        self.rho_init = np.log(psi)

    def _make_param_indices(self):
        p, m = self.p, self.m
        nl = p * m
        ns = m * (m - 1) // 2
        nr = p
        if self.rotation_type == "oblique":
            nc = m * (m - 1) 
            ixa = np.arange(nl+ns+nr+nc)
        else:
            nc = m * (m - 1) //2
            ixa = np.r_[np.arange(nl), np.arange(nl+ns, nl+ns+nr+nc)]
        nt = nl + ns + nr
        ixl = np.arange(nl)
        ixs = np.arange(nl, nl + ns)
        ixr = np.arange(nl + ns, nl + ns + nr)
        self.ixa, self.ixl, self.ixs, self.ixr = ixa, ixl, ixs, ixr
        self.nl, self.ns, self.nr, self.nc, self.nt = nl, ns, nr, nc, nt
        self.n_params = len(ixa) - nc
        
    def model_matrices_to_params(self, L, Phi, Psi):
        """
        Parameters
        ----------
        L : (p, m) array
            Loadings matrix.
        Phi : (m, m) array
            Factor correlation matrix.
        Psi : (p, p) array
            Diagonal matrix of unique variances.

        Returns
        -------
        params : (nt, ) array 
            Vector of parameters.

        """
        if Psi.ndim==2:
            psi = np.diag(Psi)
        else:
            psi = Psi
        params = np.zeros(self.nt)
        params[self.ixl] = vec(L.copy())
        params[self.ixs] = vecl(Phi.copy())
        params[self.ixr] = psi.copy()
        return params

    def loglike(self, psi):
        """

        Parameters
        ----------
        psi : (p,) array
            Vector of unique variances.

        Returns
        -------
        f : float
            -2 times log likelihood minus constants.

        """
        S, q = self.S, self.n_vars - self.n_facs
        s = 1.0 / np.sqrt(psi[:, None])
        u = np.linalg.eigvalsh(s.T * S * s)[:q]
        f = np.sum(u - np.log(u) - 1)
        return f
    
    def loglike_exp(self, rho):
        """

        Parameters
        ----------
        rho : (p,) array
            Vector of log unique variances.

        Returns
        -------
        f : float
            -2 times log likelihood minus constants.

        """
        psi = np.exp(rho)
        f = self.loglike(psi)
        return f
    
    def gradient(self, psi):
        """
        Parameters
        ----------
        psi : (p,) array
            Vector of unique variances.

        Returns
        -------
        g : (p,) array
            Derivative of -2 times log likelihood with respect to unique variances.

        """
        S, q = self.S,  self.n_vars - self.n_facs
        s = 1.0 / np.sqrt(psi[:, None])
        u, V = np.linalg.eigh(s.T * S * s)
        g = ((1-u[:q]) * V[:, :q]**2).sum(axis=1)/psi
        return g
    
    def gradient_exp(self, rho):
        """
        Parameters
        ----------
        rho : (p,) array
            Vector of log unique variances.

        Returns
        -------
        dF_dRho : (p,) array
            Derivative of -2 times log likelihood with respect to log unique variances.

        """
        psi = np.exp(rho)
        dF_dPsi = self.gradient(psi)
        dF_dRho = psi * dF_dPsi
        return dF_dRho
    
    def hessian(self, psi):
        """

        Parameters
        ----------
        psi : (p,) array
            Vector of unique variances.
        Returns
        -------
        H : (p, p) array
            Hessian of -2 log likelihood with respect to unique variances.

        """
        S, q, p= self.S, self.n_vars - self.n_facs, self.n_vars
        s = 1.0 / np.sqrt(psi[:, None])
        Psi_inv = np.diag(1 / psi)
        W = s.T * S * s
        u, V = np.linalg.eigh(W)
        B = np.zeros((p, p))
        Ip = np.eye(p)
        for i in range(q):
            A = np.outer(V[:, i], V[:, i])
            b1 = (2 * u[i] - 1) * A
            b2 = 2 * u[i] * (u[i] - 1) * np.linalg.pinv(u[i] * Ip - W)
            B+= (b1 + b2) * A
        H = Psi_inv.dot(B).dot(Psi_inv)
        return H
    
    def hessian_exp(self, rho):
        """

        Parameters
        ----------
        rho : (p,) array
            Vector of unique variances.
        Returns
        -------
        H : (p, p) array
            Hessian of -2 log likelihood with respect to log unique variances.

        """
        psi = np.exp(rho)
        dF_dPsi = self.gradient(psi)
        d2F_dPsi2 = self.hessian(psi)
        H = psi[:, None].T * d2F_dPsi2 * psi[:, None]
        H[np.diag_indices_from(H)] += psi * dF_dPsi
        return H
    
    def loadings_from_psi(self, psi):
        """

        Parameters
        ----------
        psi : (p,) array
            Vector of unique variances.

        Returns
        -------
        A : (p, m) array
            Canonical loadings with respect to psi.

        """
        S, m = self.S,  self.n_facs
        s = 1.0 / np.sqrt(psi[:, None])
        u, V = np.linalg.eigh(s.T * S * s)
        w = np.sqrt(u[-m:] - 1)
        A = np.sqrt(psi[:,None]) * V[:, -m:] * w
        return A
    
    def _optimize_psi(self, opt_kws=None):
        opt_kws = handle_default_kws(opt_kws, {"method":"trust-constr"})
        opt = sp.optimize.minimize(self.loglike_exp, self.rho_init, jac=self.gradient_exp,
                                   hess=self.hessian_exp, **opt_kws)
        rho, psi = opt.x, np.exp(opt.x)
        A = self.loadings_from_psi(psi)
        return opt, rho, psi, A
        
    def _rotate_loadings(self, A=None, opt_kws=None):
        A = self.A.copy() if A is None else A
        opt_kws = handle_default_kws(opt_kws, {})
        if self._rotate is not None:
            self._rotate.A = A
            self._rotate.fit(opt_kws=opt_kws)
            T = self._rotate.T
            L = self._rotate.rotate(T)
        else:
            L = A
            T = np.eye(self.m)
        Phi = np.dot(T.T, T)
        return L, T, Phi

    def hessian_aug(self, params):
        H_aug = np.zeros((self.nt+self.nc, self.nt+self.nc))
        H = self.hessian_params(params)
        C = self.constraint_derivs(params)
        H_aug[:self.nt, :self.nt] = H
        H_aug[self.nt:, :self.nt] = C
        H_aug[:self.nt, self.nt:] = C.T
        H_aug  = H_aug[self.ixa][:, self.ixa]
        return H_aug
    
    def _fit(self, sort_factors=False, loglike_opt_kws=None, rotation_opt_kws=None):
        opt, rho, psi, A = self._optimize_psi(opt_kws=loglike_opt_kws)
        L, T, Phi = self._rotate_loadings(A.copy(), opt_kws=rotation_opt_kws)
        if sort_factors:
            L, T, Phi, factor_perm = self.order_loadings(L, T, Phi)
        else:
            factor_perm = np.eye(self.m)
        params = self.model_matrices_to_params(L, Phi, psi)
        H = self.hessian_aug(params)
        Acov = np.linalg.inv(H)
        se_params = np.sqrt(np.diag(Acov)[:-self.nc]/self.n_obs * 2.0) 
        L_se = invec(se_params[self.ixl], self.n_vars, self.n_facs)
        if self.rotation_type == "oblique":
            Phi_se = invecl(se_params[self.ixs])
            psi_se = se_params[self.ixr]
        else:
            Phi_se = invecl(np.ones(self.nc))
            psi_se = se_params[self.nl:]
        self.opt, self.rho, self.psi, self.A = opt, rho, psi, A
        self.L, self.T, self.Phi, self.Psi = L, T, Phi, np.diag(psi)
        self.H_aug = H
        self.Acov = Acov
        self.se_params = se_params
        self.params = params
        self.L_se, self.Phi_se, self.psi_se = L_se, Phi_se, psi_se
        self.factor_perm = factor_perm
        
            
    def fit(self):
        self._fit()
        z = self.params[self.ixa[:-self.nc]] / self.se_params
        p = sp.stats.norm(0, 1).sf(np.abs(z)) * 2.0
        
        param_labels = []
        for j in range(self.n_facs):
            for i in range(self.n_vars):
                param_labels.append(f"L[{i}][{j}]")
        if self.rotation_type == "oblique":
            for i in range(self.n_facs):
                for j in range(i):
                    param_labels.append(f"Phi[{i}][{j}]")
        for i in range(self.n_vars):
            param_labels.append(f"Psi[{i}]")
        res_cols = ["param", "SE", "z", "p"]
        fcols = [f"Factor{i}" for i in range(self.n_facs)]
        self.res = pd.DataFrame(np.vstack((self.params[self.ixa[:-self.nc]], self.se_params, z, p)).T,
                                columns=res_cols, index=param_labels)
        self.Loadings = pd.DataFrame(self.L, index=self.cols, columns=fcols)
        self.FactorCorr = pd.DataFrame(self.Phi, index=fcols, columns=fcols)
        self.ResidualCov = pd.DataFrame(self.Psi, index=self.cols, columns=self.cols)
        self.Sigma = self.sigma_params(self.params)
        chi2_table, incrimental_fit_indices, misc_fit_indices = self.fit_indices()
        self.chi2_table = chi2_table 
        self.incrimental_fit_indices = incrimental_fit_indices
        self.misc_fit_indices=  misc_fit_indices
        
        
    def order_loadings(self, L, T, Phi):
        order = np.argsort(np.sum(L**2, axis=0))
        sign = np.sign(np.sum(L, axis=0))
        perm_mat = np.diag(sign)[:, order]
        L = L[:, order] * sign
        T = T[:, order] * sign
        Phi = Phi[order, order[None].T]
        Phi = sign[:,None] * Phi * sign[: None].T
        return L, T, Phi, perm_mat
        
    def params_to_model_matrices(self, params):
        L = invec(params[self.ixl], self.n_vars, self.n_facs)
        Phi = invecl(params[self.ixs])            
        Psi = np.diag(params[self.ixr])
        return L, Phi, Psi
    
    def sigma_params(self, params):
        L, Phi, Psi = self.params_to_model_matrices(params)
        Sigma = L.dot(Phi).dot(L.T)+Psi
        return Sigma
    
    def dsigma_params(self, params):
        L, Phi, Psi = self.params_to_model_matrices(params)
        DLambda = np.dot(self.LpNp, np.kron(L.dot(Phi), self.Ip))
        DPhi = self.Lp.dot(np.kron(L, L)).dot(self.Dm)[:, self.ndg_inds]
        DPsi = np.dot(self.Lp, np.diag(vec(np.eye(self.n_vars))))[:, self.d_inds]
        G= np.block([DLambda, DPhi, DPsi])
        return G
    
    def d2sigma_params(self, params):
        L, Phi, Psi = self.params_to_model_matrices(params)
        Hpp = []
        Im, E = self.Im, self.E*0.0
        Hij = np.zeros((self.nt, self.nt))
        for i in range(self.n_vars):
            for j in range(i, self.n_vars):
                E[i, j] = 1.0
                T = E + E.T
                H11 = np.kron(Phi, T)
                H22 = np.kron(Im, T.dot(L)).dot(self.Dm)[:, self.ndg_inds]
                Hij[self.ixl, self.ixl[:, None]] = H11
                Hij[self.ixl, self.ixs[:, None]] = H22.T
                Hij[self.ixs, self.ixl[:, None]] = H22
                E[i, j] = 0.0
                Hpp.append(Hij[None])
                Hij = Hij*0.0
        D2Sigma = np.concatenate(Hpp,axis=0)
        return D2Sigma
    
    def hessian_params(self, params):
        L, Phi, Psi = self.params_to_model_matrices(params)
        Sigma = L.dot(Phi).dot(L.T) + Psi
        Sigma_inv = np.linalg.inv(Sigma)
        Sdiff = self.S - Sigma
        d = vech(Sdiff)
        DLambda = np.dot(self.LpNp, np.kron(L.dot(Phi), self.Ip))
        DPhi = self.Lp.dot(np.kron(L, L)).dot(self.Dm)[:, self.ndg_inds]
        DPsi = np.dot(self.Lp, np.diag(vec(np.eye(self.n_vars))))[:, self.d_inds]
        G= np.block([DLambda, DPhi, DPsi])
        G = self.dsigma_params(params)
        DGp = self.Dp.dot(G)
        W1 = np.kron(Sigma_inv, Sigma_inv)
        W2 = np.kron(Sigma_inv, Sigma_inv.dot(Sdiff).dot(Sigma_inv))
        H1 = 0.5 * DGp.T.dot(W1).dot(DGp)
        H2 = 1.0 * DGp.T.dot(W2).dot(DGp)
        Hpp = []
        Dp, Im, E = self.Dp, self.Im, self.E*0.0
        Hij = np.zeros((self.nt, self.nt))
        for i in range(self.n_vars):
            for j in range(i, self.n_vars):
                E[i, j] = 1.0
                T = E + E.T
                H11 = np.kron(Phi, T)
                H22 = np.kron(Im, T.dot(L)).dot(self.Dm)[:, self.ndg_inds]
                Hij[self.ixl, self.ixl[:, None]] = H11
                Hij[self.ixl, self.ixs[:, None]] = H22.T
                Hij[self.ixs, self.ixl[:, None]] = H22
                E[i, j] = 0.0
                Hpp.append(Hij[:, :, None])
                Hij = Hij*0.0
        W = np.linalg.multi_dot([Dp.T, W1, Dp])
        dW = np.dot(d, W)
        Hp = np.concatenate(Hpp, axis=2) 
        H3 = np.einsum('k,ijk ->ij', dW, Hp)      
        H = (H1 + H2 - H3 / 2.0)*2.0 
        return H     
    
    def implied_cov_params(self, params):
        L, Phi, Psi = self.params_to_model_matrices(params)
        Sigma = L.dot(Phi).dot(L.T) + Psi
        return Sigma
    
    def loglike_params(self, params):
        Sigma = self.implied_cov_params(params)
        _, lndS = np.linalg.slogdet(Sigma)
        trSV = np.trace(np.linalg.solve(Sigma, self.S))
        ll = lndS + trSV
        return ll
    
    def gradient_params(self, params):
        L, Phi, Psi = self.params_to_model_matrices(params)
        Sigma = L.dot(Phi).dot(L.T) + Psi
        V = np.linalg.pinv(Sigma)
        VRV = V.dot(Sigma - self.S).dot(V)
        gL = 2 * vec(VRV.dot(L.dot(Phi)))
        gPhi = 2*vecl(L.T.dot(VRV).dot(L))
        gPsi = np.diag(VRV)
        g = np.zeros(self.nt)
        g[self.ixl] = gL
        g[self.ixs] = gPhi
        g[self.ixr] = gPsi
        return g
    
    def _canonical_constraint(self, L, Psi):
        C = vecl(np.dot(L.T, np.diag(1/np.diag(Psi))).dot(L))
        return C
    
    def canonical_constraint(self, params):
        L, Phi, Psi = self.params_to_model_matrices(params)    
        return self._canonical_constraint(L, Psi)
    
    def constraints(self, params):
        L, Phi, Psi = self.params_to_model_matrices(params)  
        if self._rotate is not None:
            C = self._rotate.constraints(L, Phi)
        else:
            C = self._canonical_constraint(L, Psi)
        return C

    
    def constraint_derivs(self, params):
        L, Phi, Psi = self.params_to_model_matrices(params)
        if self.rotation_type == "ortho":
            dCdL = self._rotate.dC_dL_Ortho(L, Phi)[self._rotate.lix]
            dCdP = np.zeros((dCdL.shape[0], self.ns))
            dCdPsi = np.zeros((dCdL.shape[0], self.nr))
        elif self.rotation_type == "oblique":
            dCdL = self._rotate.dC_dL_Obl(L, Phi)
            dCdP = self._rotate.dC_dP_Obl(L, Phi)
            dCdL = dCdL.reshape(self.m * self.m, self.p * self.m, order='F')
            dCdP = dCdP.reshape(self.m**2, self.m**2, order='F')
            dCdL = dCdL[self._rotate.lix]
            dCdP = dCdP[self._rotate.lix][:, self._rotate.cix]
            dCdPsi = np.zeros((dCdL.shape[0], self.nr))

        else:
            Psi_inv = np.diag(1.0 / np.diag(Psi))
            A = L.T.dot(Psi_inv)
            Dpsi = self.Dpsi
            #Dpsi.data = np.diag(Psi)
            dCdL = self.Nm.dot(np.kron(self.Im, A))[self.l_inds]
            dCdP = np.zeros((dCdL.shape[0], self.ns))
            dCdPsi =-np.kron(A, A)
            dCdPsi = (Dpsi.T.dot(dCdPsi.T)).T[self.l_inds]
        dC = np.concatenate([dCdL, dCdP, dCdPsi], axis=1)
        return dC
    
    def estimate_factors(self,  X=None, method="tenBerge", L=None,
                         Phi=None, Psi=None, S=None):
        X = self.X if X is None else X
        L = self.L if L is None else L
        Phi = self.Phi if Phi is None else Phi
        Psi = self.Psi if Psi is None else Psi
        S = self.S if S is None else S
        Psiinv = np.diag(1.0/np.diag(Psi))
        LPhi = L.dot(Phi)
        LtPsi_inv = L.T.dot(Psiinv)
        Gamma = LtPsi_inv.dot(L)
        if method == "thurstone":
            B = np.linalg.solve(S, LPhi)
        elif method == "bartlett":
            B = np.linalg.solve(Gamma, LtPsi_inv).T
        elif method == "mcdonald":
            Phi_sq = mat_sqrt(Phi)
            temp = mat_sqrt(S).dot(LtPsi_inv.T.dot(Phi_sq))
            G, Delta, Mt = np.linalg.svd(temp, full_matrices=False)
            C = G.dot(Mt)
            B = inv_sqrt(S).dot(C).dot(Phi_sq)
        elif method == "tenBerge":
            Phi_sq = mat_sqrt(Phi)
            S_isq = inv_sqrt(S)
            R = S_isq.dot(L.dot(Phi_sq))
            C = R.dot(inv_sqrt(np.dot(R.T, R)))
            B = S_isq.dot(C).dot(Phi_sq)
        Z = (X - np.mean(X, axis=0)).dot(B)
        return Z, B 
    
    def _full_loglike(self, params=None, Sigma=None, nscale=True):
        n, p, S = self.n_obs, self.p, self.S
        if Sigma is None and params is not None:
            Sigma = self.sigma_params(params)
        else:
            Sigma = Sigma
        trSigmaInvS = np.trace(np.linalg.solve(Sigma, S))
        _, lndSigma = np.linalg.slogdet(Sigma)
        _, lndS = np.linalg.slogdet(S)
        s = -n / 2 if nscale else 1.0
        ll = s * (lndSigma + trSigmaInvS - lndS - p)
        return ll
    
    def fit_indices(self, Sigma_null=None, df_null=None):
        Sigma = self.sigma_params(self.params)
        Sigma_null = np.diag(np.diag(self.S)) if Sigma_null is None else Sigma_null
        df_null = int(self.p * (self.p - 1) // 2) if df_null is None else df_null
        df_full = int(self.p * (self.p + 1) // 2) - self.n_params
        
        f_null = self._full_loglike(Sigma=Sigma_null, nscale=False)
        f_full = self._full_loglike(Sigma=Sigma, nscale=False)
        ll_full = -f_full * self.n_obs / 2
        chi2_null = self.n_obs * f_null
        chi2_full = self.n_obs * f_full
        chi2_full_pval = sp.stats.chi2(df=df_full).sf(chi2_full)
        chi2_null_pval = sp.stats.chi2(df=df_null).sf(chi2_null)

        tli = ((chi2_null / df_null) - (chi2_full / df_full)) / ((chi2_null / df_null) - 1)
        nfi = (chi2_null - chi2_full) / chi2_null
        ifi = (chi2_null - chi2_full) / (chi2_null - df_full)
        rni = ((chi2_null - df_null) - (chi2_full - df_full)) / (chi2_null - df_null) 
        cfi = (np.maximum(chi2_null - df_null, 0) - np.maximum(chi2_full - df_full, 0))\
             /np.maximum(chi2_null - df_null, 0) 
        rmsea = np.sqrt(np.max(chi2_full - df_full, 0) / (df_null * (self.n_obs - 1)))
        gfi_ = gfi(Sigma, self.S)
        agfi_ = agfi(Sigma, self.S, df_full)
        srmr_ = srmr(Sigma, self.S, df_full)
        BIC = self.n_params * np.log(self.n_obs) -2 * ll_full
        AIC = 2 * self.n_params - 2*ll_full

        chi2_table = pd.DataFrame([[chi2_full, df_full, chi2_full_pval],
                                   [chi2_null, df_null, chi2_null_pval]],
                                  index=["Full", "Null"],
                                  columns=["Chi2", "df", "pval"])
        incrimental_fit_indices = pd.DataFrame(dict(TLI=tli, 
                                                    NFI=nfi,
                                                    IFI=ifi,
                                                    RNI=rni, 
                                                    CFI=cfi),
                                               index=["Value"]).T
        misc_fit_indices = pd.DataFrame(dict(GFI=gfi_, 
                                             AGFI=agfi_, 
                                             RMSEA=rmsea,
                                             SRMR=srmr_,
                                             loglikelihood=ll_full,
                                             AIC=AIC,
                                             BIC=BIC,
                        ),
                                        index=["Value"]).T
        return chi2_table, incrimental_fit_indices, misc_fit_indices
        
        
    
    
    
        
            
            
        
        

        
        
        