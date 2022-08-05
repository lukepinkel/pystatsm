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
        self._init_params()
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
        if rotation_method is not None:
            rotation_type = 'ortho' if rotation_type is None else rotation_type
            consts = get_gcf_constants(rotation_method, self.p, self.m)
            self._rotate = GeneralizedCrawfordFerguson(A=np.zeros((self.p, self.m)),
                                                       rotation_type=rotation_type, 
                                                       **consts)
        else:
            self._rotate = None
        self.rotation_method, self.rotation_type = rotation_method, rotation_type
        rc = np.arange(self.p)*self.p+np.arange(self.p), np.arange(self.p)
        self.Dpsi = sp.sparse.csc_matrix((np.ones(self.p), rc), shape=(self.p**2, self.p))

        
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
        
    def _init_params(self):
        self.pix = np.arange(self.n_vars*self.n_facs, self.n_vars*self.n_facs+self.n_vars)
        L = self.V[:, :self.n_facs]
        psi = np.diag(self.S - np.dot(L, L.T))
        self.psi_init = psi
        self.rho_init = np.log(psi)

    def loglike(self, psi):
        S, q = self.S, self.n_vars - self.n_facs
        s = 1.0 / np.sqrt(psi[:, None])
        u = np.linalg.eigvalsh(s.T * S * s)[:q]
        f = np.sum(u - np.log(u) - 1)
        return f
    
    def loglike_exp(self, rho):
        psi = np.exp(rho)
        return self.loglike(psi)
    
    def gradient(self, psi):
        S, q = self.S,  self.n_vars - self.n_facs
        s = 1.0 / np.sqrt(psi[:, None])
        u, V = np.linalg.eigh(s.T * S * s)
        g = ((1-u[:q]) * V[:, :q]**2).sum(axis=1)/psi
        return g
    
    def gradient_exp(self, rho):
        psi = np.exp(rho)
        dF_dPsi = self.gradient(psi)
        dF_dRho = psi * dF_dPsi
        return dF_dRho
    
    def hessian(self, psi):
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
        psi = np.exp(rho)
        dF_dPsi = self.gradient(psi)
        d2F_dPsi2 = self.hessian(psi)
        H = psi[:, None].T * d2F_dPsi2 * psi[:, None]
        H[np.diag_indices_from(H)] += psi * dF_dPsi
        return H
    
    def loadings_from_psi(self, psi):
        S, m = self.S,  self.n_facs
        s = 1.0 / np.sqrt(psi[:, None])
        u, V = np.linalg.eigh(s.T * S * s)
        w = np.sqrt(u[-m:] - 1)
        A = np.sqrt(psi[:,None]) * V[:, -m:] * w
        return A
    
    def _optimize_psi(self):
        opt = sp.optimize.minimize(self.loglike_exp, self.rho_init, jac=self.gradient_exp,
                                   hess=self.hessian_exp, method='trust-constr')
        self.opt, self.rho = opt, opt.x
        self.psi = np.exp(self.rho)
        self.A = self.loadings_from_psi(self.psi)
        
    def _rotate_loadings(self):
        if self._rotate is not None:
            self._rotate.A = self.A
            self._rotate.fit()
            self.T = self._rotate.T
            self.L = self._rotate.rotate(self.T)
        else:
            self.L = self.A
            self.T = np.eye(self.m)
        self.Phi = np.dot(self.T.T, self.T)
        self.Psi = np.diag(self.psi)
        self._make_params(self.L, self.Phi, self.Psi)
    
    def hessian_aug(self, params):
        H_aug = np.zeros((self.nt+self.nc, self.nt+self.nc))
        H = self.hessian_params(params)
        C = self.constraint_derivs(params)
        H_aug[:self.nt, :self.nt] = H
        H_aug[self.nt:, :self.nt] = C
        H_aug[:self.nt, self.nt:] = C.T
        H_aug  = H_aug[self.ixa][:, self.ixa]
        return H_aug
    
    def _fit(self):
        self._optimize_psi()
        self._rotate_loadings()
        self.H = self.hessian_aug(self.params)
        self.acov = np.linalg.inv(self.H)
        self.se_params = np.sqrt(np.diag(self.acov)[:-self.nc]/self.n_obs * 2.0) 
        self.L_se = invec(self.se_params[self.ixl], self.n_vars, self.n_facs)
        if self.rotation_type == "oblique":
            self.Phi_se = invecl(self.se_params[self.ixs])
            self.psi_se = self.se_params[self.ixr]
        else:
            self.Phi_se = invecl(np.ones(self.nc))
            self.psi_se = self.se_params[self.nl:]
            
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
        
    def _make_params(self, L, Phi, Psi):
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
        params = np.zeros(nt)
        ixl = np.arange(nl)
        ixs = np.arange(nl, nl + ns)
        ixr = np.arange(nl + ns, nl + ns + nr)
        params[ixl] = vec(L)
        params[ixs] = vecl(Phi)
        params[ixr] = np.diag(Psi)
        self.ixl, self.ixs, self.ixr = ixl, ixs, ixr
        self.nl, self.ns, self.nr, self.nc = nl, ns, nr, nc
        self.nt = nl + ns + nr
        self.params = params
        self.ixa = ixa
        
    def model_matrices_params(self, params):
        L = invec(params[self.ixl], self.n_vars, self.n_facs)
        Phi = invecl(params[self.ixs])            
        Psi = np.diag(params[self.ixr])
        return L, Phi, Psi
    
    def sigma_params(self, params):
        L, Phi, Psi = self.model_matrices_params(params)
        Sigma = L.dot(Phi).dot(L.T)+Psi
        return Sigma
    
    def dsigma_params(self, params):
        L, Phi, Psi = self.model_matrices_params(params)
        DLambda = np.dot(self.LpNp, np.kron(L.dot(Phi), self.Ip))
        ix = vech(np.eye(L.shape[1]))!=1
        DPhi = self.Lp.dot(np.kron(L, L)).dot(self.Dm)[:, ix]
        DPsi = np.dot(self.Lp, np.diag(vec(np.eye(self.n_vars))))[:, self.d_inds]
        G= np.block([DLambda, DPhi, DPsi])
        return G
    
    def d2sigma_params(self, params):
        L, Phi, Psi = self.model_matrices_params(params)
        Hpp = []
        Im, E = self.Im, self.E
        ix = vech(np.eye(L.shape[1]))!=1
        Hij = np.zeros((self.nt, self.nt))
        for i in range(self.n_vars):
            for j in range(i, self.n_vars):
                E[i, j] = 1.0
                T = E + E.T
                H11 = np.kron(Phi, T)
                H22 = np.kron(Im, T.dot(L)).dot(self.Dm)[:, ix]
                Hij[self.ixl, self.ixl[:, None]] = H11
                Hij[self.ixl, self.ixs[:, None]] = H22.T
                Hij[self.ixs, self.ixl[:, None]] = H22
                E[i, j] = 0.0
                Hpp.append(Hij[None])
                Hij = Hij*0.0
        D2Sigma = np.concatenate(Hpp,axis=0)
        return D2Sigma
    
    def hessian_params(self, params):
        L, Phi, Psi = self.model_matrices_params(params)
        Sigma = L.dot(Phi).dot(L.T) + Psi
        Sigma_inv = np.linalg.inv(Sigma)
        Sdiff = self.S - Sigma
        d = vech(Sdiff)
        G = self.dsigma_params(params)
        DGp = self.Dp.dot(G)
        W1 = np.kron(Sigma_inv, Sigma_inv)
        W2 = np.kron(Sigma_inv, Sigma_inv.dot(Sdiff).dot(Sigma_inv))
        H1 = 0.5 * DGp.T.dot(W1).dot(DGp)
        H2 = 1.0 * DGp.T.dot(W2).dot(DGp)
        ix = vech(np.eye(L.shape[1]))!=1
        Hp = self.d2sigma_params(params)
        W = np.linalg.multi_dot([self.Dp.T, W1, self.Dp])
        dW = np.dot(d, W)
        H3 = np.einsum('i,ijk ->jk', dW, Hp)      
        H = (H1 + H2 - H3 / 2.0)*2.0
        return H     
    
    def implied_cov_params(self, params):
        L, Phi, Psi = self.model_matrices_params(params)
        Sigma = L.dot(Phi).dot(L.T) + Psi
        return Sigma
    
    def loglike_params(self, params):
        Sigma = self.implied_cov_params(params)
        _, lndS = np.linalg.slogdet(Sigma)
        trSV = np.trace(np.linalg.solve(Sigma, self.S))
        ll = lndS + trSV
        return ll
    
    def gradient_params(self, params):
        L, Phi, Psi = self.model_matrices_params(params)
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
        L, Phi, Psi = self.model_matrices_params(params)    
        return self._canonical_constraint(L, Psi)
    
    def constraints(self, params):
        L, Phi, Psi = self.model_matrices_params(params)  
        if self._rotate is not None:
            C = self._rotate.constraints(L, Phi)
        else:
            C = self._canonical_constraint(L, Psi)
        return C

    
    def constraint_derivs(self, params):
        L, Phi, Psi = self.model_matrices_params(params)
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

        
        
        