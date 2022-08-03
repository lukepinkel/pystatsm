#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 16:02:21 2022

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
        self._make_params()
        self.p, self.m = self.n_vars, self.n_facs
        self.E = np.eye(self.n_vars)
        self.Ik = np.eye(self.n_facs)
        self.Nk = nmat(self.n_facs).A
        self.Lk = lmat(self.n_facs).A
        self.Dk = dmat(self.n_facs).A
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
        
    def _make_params(self):
        self.n_pars = self.n_vars*self.n_facs + self.n_vars
        self.theta = np.zeros(self.n_pars)
        self.lix = np.arange(self.n_vars*self.n_facs)
        self.pix = np.arange(self.n_vars*self.n_facs, self.n_vars*self.n_facs+self.n_vars)
        L = self.V[:, :self.n_facs]
        psi = np.diag(self.S - np.dot(L, L.T))
        self.theta[self.lix] = vec(L)
        self.theta[self.pix] = np.log(psi)
    
    def model_matrices(self, theta):
        """

        Parameters
        ----------
        theta : ndarray
            ndarray of length t containing model parameters.

        Returns
        -------
        L : ndarray
            (p x q) matrix of loadings.
        Psi : ndarray
            (p x p) diagonal matrix of residual covariances.

        """
        L = invec(theta[self.lix], self.n_vars, self.n_facs)
        Psi = np.diag(np.exp(theta[self.pix]))
        return L, Psi
    
    def implied_cov(self, theta):
        """
        
        Parameters
        ----------
        theta : ndarray
             ndarray of length t containing model parameters.

        Returns
        -------
        Sigma : ndarray
            (p x p) implied covariance matrix.

        """
        L, Psi = self.model_matrices(theta)
        Sigma = L.dot(L.T) + Psi
        return Sigma
    
    def loglike(self, theta):
        """
        

        Parameters
        ----------
        theta : ndarray
             ndarray of length t containing model parameters.

        Returns
        -------
        ll : float
            Loglikelihood of the model.

        """
        Sigma = self.implied_cov(theta)
        _, lndS = np.linalg.slogdet(Sigma)
        trSV = np.trace(np.linalg.solve(Sigma, self.S))
        ll = lndS + trSV
        return ll
    
    def gradient(self, theta):
        """
        
        Parameters
        ----------
        theta : ndarray
             ndarray of length t containing model parameters.

        Returns
        -------
        g : ndarray
             ndarray of length t containing the derivatives of the 
             loglikelihood with respect to theta.

        """
        L, Psi = self.model_matrices(theta)
        Sigma = L.dot(L.T) + Psi
        V = np.linalg.pinv(Sigma)
        VRV = V.dot(Sigma - self.S).dot(V)
        g1 = 2 * vec(VRV.dot(L))
        g2 = np.diag(VRV.dot(Psi))
        g = np.zeros(self.n_pars)
        g[self.lix] = g1
        g[self.pix] = g2
        return g
    
        
    def dsigma(self, theta):
        L, Psi = self.model_matrices(theta)
        DLambda = np.dot(self.LpNp, np.kron(L, self.Ip))
        DPsi = np.dot(self.Lp, np.diag(vec(Psi)))[:, self.d_inds]
        G = np.block([DLambda, DPsi])
        return G
    
    def hessian(self, theta):
        L, Psi = self.model_matrices(theta)
        Sigma = L.dot(L.T) + Psi
        Sigma_inv = np.linalg.inv(Sigma)
        Sdiff = self.S - Sigma
        d = vech(Sdiff)
        G = self.dsigma(theta)
        DGp = self.Dp.dot(G)
        W1 = np.kron(Sigma_inv, Sigma_inv)
        W2 = np.kron(Sigma_inv, Sigma_inv.dot(Sdiff).dot(Sigma_inv))
        H1 = 0.5 * DGp.T.dot(W1).dot(DGp)
        H2 = 1.0 * DGp.T.dot(W2).dot(DGp)

        Hpp = []
        Dp, Ik, E = self.Dp, self.Ik, self.E
        Hij = np.zeros((self.n_pars, self.n_pars))
        for i in range(self.n_vars):
            for j in range(i, self.n_vars):
                eij = np.zeros(self.n_vars)
                if i==j:
                    eij[i] = 1.0
                E[i, j] = 1.0
                T = E + E.T
                H11 = np.kron(Ik, T)
                H22 = np.diag(Psi) * eij
                Hij[self.lix, self.lix[:, None]] = H11
                Hij[self.pix, self.pix] = H22
                E[i, j] = 0.0
                Hpp.append(Hij[:, :, None])
                Hij = Hij*0.0
        W = np.linalg.multi_dot([Dp.T, W1, Dp])
        dW = np.dot(d, W)
        Hp = np.concatenate(Hpp, axis=2) 
        H3 = np.einsum('k,ijk ->ij', dW, Hp)      
        H = (H1 + H2 - H3 / 2.0)*2.0
        return H
    
    def _make_augmented_params(self, L, Phi, Psi):
        p, q = self.n_vars, self.n_facs
        nl = p * q
        ns = q * (q - 1) // 2
        nr = p
        if self.rotation_type == "oblique":
            nc = q * (q - 1) 
        else:
            nc = q * (q - 1) //2
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
    
    def model_matrices_augmented(self, params):
        L = invec(params[self.ixl], self.n_vars, self.n_facs)
        Phi = invecl(params[self.ixs])            
        Psi = np.diag(params[self.ixr])
        return L, Phi, Psi
    
    def dsigma_augmented(self, params):
        """
        

        Parameters
        ----------
        params : ndarray
            ndarray of length m containing model parameters.

        Returns
        -------
        G : ndarray
            (k x m) matrix of derivatives of the implied covariance with 
            respect to parameters

        """
        L, Phi, Psi = self.model_matrices_augmented(params)
        DLambda = np.dot(self.LpNp, np.kron(L.dot(Phi), self.Ip))
        ix = vech(np.eye(L.shape[1]))!=1
        DPhi = self.Lp.dot(np.kron(L, L)).dot(self.Dk)[:, ix]
        DPsi = np.dot(self.Lp, np.diag(vec(np.eye(self.n_vars))))[:, self.d_inds]
        G= np.block([DLambda, DPhi, DPsi])
        return G
    
    def d2sigma_augmented(self, params):
        L, Phi, Psi = self.model_matrices_augmented(params)
        Hpp = []
        Ik, E = self.Ik, self.E
        ix = vech(np.eye(L.shape[1]))!=1
        Hij = np.zeros((self.nt, self.nt))
        for i in range(self.n_vars):
            for j in range(i, self.n_vars):
                E[i, j] = 1.0
                T = E + E.T
                H11 = np.kron(Phi, T)
                H22 = np.kron(Ik, T.dot(L)).dot(self.Dk)[:, ix]
                Hij[self.ixl, self.ixl[:, None]] = H11
                Hij[self.ixl, self.ixs[:, None]] = H22.T
                Hij[self.ixs, self.ixl[:, None]] = H22
                E[i, j] = 0.0
                Hpp.append(Hij[None])
                Hij = Hij*0.0
        D2Sigma = np.concatenate(Hpp,axis=0)
        return D2Sigma
            
    
    def implied_cov_augmented(self, params):
        """
        

        Parameters
        ----------
        params : ndarray
            ndarray of length m containing model parameters.


        Returns
        -------
        Sigma : ndarray
            (p x p) implied covariance matrix.

        """
        L, Phi, Psi = self.model_matrices_augmented(params)
        Sigma = L.dot(Phi).dot(L.T) + Psi
        return Sigma
    
    def loglike_augmented(self, params):
        """
        

        Parameters
        ----------
        params : ndarray
            ndarray of length m containing model parameters.

        Returns
        -------
        ll : float
            Loglikelihood of the model.

        """
        Sigma = self.implied_cov_augmented(params)
        _, lndS = np.linalg.slogdet(Sigma)
        trSV = np.trace(np.linalg.solve(Sigma, self.S))
        ll = lndS + trSV
        return ll
    
    def gradient_augmented(self, params):
        """
        
        Parameters
        ----------
        params : ndarray
             ndarray of length m containing model parameters.

        Returns
        -------
        g : ndarray
             ndarray of length m containing the derivatives of the 
             loglikelihood with respect to params.

        """
        L, Phi, Psi = self.model_matrices_augmented(params)
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
    
    def hessian_augmented(self, params):
        """
        

        Parameters
        ----------
        params : ndarray
             ndarray of length m containing model parameters.

        Returns
        -------
        H : ndarray
             (m x m) matrix of second derivative of the log likelihood with 
             respect to params

        """
        L, Phi, Psi = self.model_matrices_augmented(params)
        Sigma = L.dot(Phi).dot(L.T) + Psi
        Sigma_inv = np.linalg.inv(Sigma)
        Sdiff = self.S - Sigma
        d = vech(Sdiff)
        G = self.dsigma_augmented(params)
        DGp = self.Dp.dot(G)
        W1 = np.kron(Sigma_inv, Sigma_inv)
        W2 = np.kron(Sigma_inv, Sigma_inv.dot(Sdiff).dot(Sigma_inv))
        H1 = 0.5 * DGp.T.dot(W1).dot(DGp)
        H2 = 1.0 * DGp.T.dot(W2).dot(DGp)
        ix = vech(np.eye(L.shape[1]))!=1
        Hpp = []
        Dp, Ik, E = self.Dp, self.Ik, self.E
        Hij = np.zeros((self.nt, self.nt))
        for i in range(self.n_vars):
            for j in range(i, self.n_vars):
                E[i, j] = 1.0
                T = E + E.T
                H11 = np.kron(Phi, T)
                H22 = np.kron(Ik, T.dot(L)).dot(self.Dk)[:, ix]
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
    
    def loglike_canonical(self, psi):
        S, q = self.S, self.n_vars - self.n_facs
        s = 1.0 / np.sqrt(psi[:, None])
        u = np.linalg.eigvalsh(s.T * S * s)[:q]
        f = np.sum(u - np.log(u) - 1)
        return f
    
    def gradient_canonical(self, psi):
        S, q = self.S,  self.n_vars - self.n_facs
        s = 1.0 / np.sqrt(psi[:, None])
        u, V = np.linalg.eigh(s.T * S * s)
        g = ((1-u[:q]) * V[:, :q]**2).sum(axis=1)/psi
        return g
    
    def hessian_canonical(self, psi):
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
        H = -Psi_inv.dot(B).dot(Psi_inv)
        return H
    
    def loadings_from_psi(self, psi):
        S, m = self.S,  self.n_facs
        s = 1.0 / np.sqrt(psi[:, None])
        u, V = np.linalg.eigh(s.T * S * s)
        w = np.sqrt(u[-m:] - 1)
        A = np.sqrt(psi[:,None]) * V[:, -m:] * w
        return A
        
    def _unrotated_constraint_dervs(self, L, Psi):
        """
        

        Parameters
        ----------
        L : ndarray
            (p x q) matrix of loadings.
        Psi : ndarray
            (p x p) diagonal matrix of residual covariances.

        Returns
        -------
        J : ndarray
            (q * (q - 1) // 2 x t) matrix of derivatives of the constraints for
            the orthogonal model.

        """
        Psi_inv = np.diag(1.0 / np.diag(Psi))
        A = L.T.dot(Psi_inv)
        J1 = self.Nk.dot(np.kron(self.Ik, A))
        J2 =-np.kron(A, A)[:, self.d_inds]
        J = np.concatenate([J1, J2], axis=1)
        return J
        
    def constraint_derivs(self, theta):
        """
        

        Parameters
        ----------
        theta : ndarray
             ndarray of length t containing model parameters.

        Returns
        -------
        J : ndarray
            (q * (q - 1) // 2 x t) matrix of derivatives of the constraints for
            the orthogonal model.

        """
        L, Psi = self.model_matrices(theta)            
        J = self._unrotated_constraint_dervs(L, Psi)
        return J
    
    def _reorder_factors(self, L, T, theta):
        """
        

        Parameters
        ----------
        L : ndarray
            (p x q) matrix of loadings.
        T : ndarray
            (q x q)  rotation matrix

        theta :  ndarray
             ndarray of length t containing model parameters.

        Returns
        -------
        L : ndarray
            (p x q) matrix of loadings.
        T : ndarray
            (q x q)  rotation matrix

        theta :  ndarray
             ndarray of length t containing model parameters.


        """
        v = np.sum(L**2, axis=0)
        order = np.argsort(v)[::-1]
        if type(L) is pd.DataFrame:
            cols = L.columns
            L = L.iloc[:, order]
            j = np.argmax(np.abs(L), axis=0)#np.sum(L, axis=0)
            s = np.sign(L[j, np.arange(L.shape[1])])#np.sign(j)
            L = s * L
            L.columns = cols
            theta[self.lix] = vec(L.values)
        else:
            L = L[:, order]
            j = np.argmax(np.abs(L), axis=0)#np.sum(L, axis=0)
            s = np.sign(L[j, np.arange(L.shape[1])])#np.sign(j)
            L = s * L
            theta[self.lix] = vec(L)
        T = T[:, order] * s[:, None]
        return L, T, theta
        
    def _fit_indices(self, Sigma):
        """
        

        Parameters
        ----------
        Sigma : ndarray
            (p x p) implied covariance matrix.

        Returns
        -------
        sumstats : dict

        """
        t = (self.n_vars + 1.0) * self.n_vars / 2.0
        k = len(self.theta)
        degfree = t  - k
        GFI = gfi(Sigma, self.S)
        AGFI = agfi(Sigma, self.S, degfree)
        chi2, chi2p = lr_test(Sigma, self.S, degfree, self.n_obs-1)
        stdchi2 = (chi2 - degfree) /  np.sqrt(2 * degfree)
        RMSEA = np.sqrt(np.maximum(chi2 -degfree, 0)/(degfree*(self.n_obs-1)))
        SRMR = srmr(Sigma, self.S, degfree)
        llval = -(self.n_obs-1)*self.loglike(self.theta)/2
        BIC = k * np.log(self.n_obs) -2 * llval
        AIC = 2 * k - 2*llval
        sumstats = dict(GFI=GFI, AGFI=AGFI, chi2=chi2, chi2_pvalue=chi2p,
                         std_chi2=stdchi2, RMSEA=RMSEA, SRMR=SRMR,
                         loglikelihood=llval,
                         AIC=AIC,
                         BIC=BIC)
        return sumstats
    
    def _fit(self, hess=True, opt_kws=None):
        """
        

        Parameters
        ----------
        hess : bool, optional
            Whether or not to use the analytic hessian. The default is True.

        Returns
        -------
        None.

        """
        hess = self.hessian if hess else None
        opt_kws = {} if opt_kws is None else opt_kws
        self.opt = sp.optimize.minimize(self.loglike, self.theta, jac=self.gradient,
                                        hess=hess, method='trust-constr', **opt_kws)
        self.theta = self.opt.x.copy()
        self.A, self.Psi = self.model_matrices(self.theta)
        
        if self.rotation_method is not None:
            self._rotate.A = self.A
            self._rotate.fit()
            self.T = self._rotate.T
            self.L = self._rotate.rotate(self.T)
            self.theta[self.lix] = vec(self.L)
        else:
            self.L = self.A
            self.T = np.eye(self.n_facs)
        #self.L, self.T, self.theta = self._reorder_factors(self.L, self.T, self.theta)
        self.Phi = np.dot(self.T.T, self.T)
        self._make_augmented_params(self.L, self.Phi, self.Psi)
        self.Sigma = self.implied_cov_augmented(self.params)
        #self.H = so_gc_cd(self.gradient_augmented, self.params)
        self.H = self.hessian_augmented(self.params)
        if self.rotation_type == "oblique":
            nl, ns, nr, nc = self.nl, self.ns, self.nr, self.nc
            nt = nl + ns + nr
            dCdL = self._rotate.dC_dL_Obl(self.L, self.Phi)
            dCdP = self._rotate.dC_dP_Obl(self.L, self.Phi)
            dCdL = dCdL.reshape(self.m * self.m, self.p * self.m, order='F')
            dCdP = dCdP.reshape(self.m**2, self.m**2, order='F')
            lix = vec(np.eye(self.m)!=1)
            cix = vec(np.tril(np.ones(self.m), -1)!=0)
            H = np.zeros((nt+nc, nt+nc))
            H[:nt, :nt] = self.H
            H[nt:, :nl] = dCdL[lix] 
            H[:nl, nt:] = dCdL[lix].T
            H[nt:, nl:nl+ns] = dCdP[lix][:, cix]
            H[nl:nl+ns, nt:] = dCdP[lix][:, cix].T
            ixp = np.arange(nt)
            
        elif self.rotation_type == "ortho":
            dCdL = self._rotate.dC_dL_Ortho(self.L, self.Phi)
            nl, nr, nc = self.nl, self.nr, self.nc
            nt, nc = nl + nr, nc 
            H = np.zeros((nt+nc, nt+nc))
            ixp = np.r_[np.arange(self.nl),
                       np.arange(self.nl+self.ns, self.nl+self.ns+self.nr)]
            lix = vec(np.tril(np.ones((self.m, self.m)), -1)!=0)
            H[:nt, :nt] = self.H[ixp, ixp[:, None]] 
            H[-nc:, :nl] = dCdL[lix]
            H[:nl, -nc:] = dCdL[lix].T 
        else:
            dCdL = self.constraint_derivs(self.theta)
            nl, nt, nc = self.nl, self.nl + self.nr, self.nc
            H = np.zeros((nt+nc, nt+nc))
            ixp = np.r_[np.arange(self.nl),
                       np.arange(self.nl+self.ns, self.nl+self.ns+self.nr)]
            lix = vec(np.tril(np.ones((self.m, self.m)), -1)!=0)
            H[:nt, :nt] = self.H[ixp, ixp[:, None]] 
            if self.nc > 0:
                H[nt:, :-nc] = dCdL[lix]
                H[:-nc, nt:] = dCdL[lix].T 

            
            
        self.ixp = ixp
        self.H_aug = H
        self.se_params = np.sqrt(np.diag(np.linalg.inv(self.H_aug))[:nt]/self.n_obs * 2.0) 
        self.L_se = invec(self.se_params[self.lix], self.n_vars, self.n_facs)
        if self.rotation_type == "oblique":
            self.Phi_se = invecl(self.se_params[self.ixs])
            self.psi_se = self.se_params[self.ixr]
        else:
            self.Phi_se = invecl(np.ones(self.nc))
            self.psi_se = self.se_params[nl:]
            
                        
    def constraint_jac(self, params):
        L, Phi, Psi = self.model_matrices_augmented(params)
        if self.rotation_type == "oblique":
            dCdL = self._rotate.dC_dL_Obl(L, Phi)
            dCdP = self._rotate.dC_dP_Obl(L, Phi)
            dCdL = dCdL.reshape(self.m * self.m, self.p * self.m, order='F')
            dCdP = dCdP.reshape(self.m**2, self.m**2, order='F')
            lix = vec(np.eye(self.m)!=1)
            cix = vec(np.tril(np.ones(self.m), -1)!=0)
            dC = np.concatenate([dCdL[lix], dCdP[lix][:, cix]], axis=1)
        elif self.rotation_type == "ortho":
            dCdL = self._rotate.dC_dL_Ortho(L, Phi)
            lix = vec(np.tril(np.ones((self.m, self.m)), -1)!=0)
            dC = dCdL[lix]
        else:
            dCdL = self.constraint_derivs(self.theta)
            lix = vec(np.tril(np.ones((self.m, self.m)), -1)!=0)
            dC = dCdL[lix]
        return dC
        
    def fit(self, compute_factors=True, factor_method='tenberge', hess=True,
            opt_kws=None):
        """
        

        Parameters
        ----------
        compute_factors : bool, optional
            Whether or not to compute factors. The default is True.
        factor_method : str, optional
            Method to compute factors. The default is 'regression'.
        hess : bool, optional
             Whether or not to use the analytic hessian. The default is True.
        **opt_kws : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self._fit(hess, opt_kws)
        self.sumstats = self._fit_indices(self.Sigma)
        z = self.params[self.ixp] / self.se_params
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
        self.res = pd.DataFrame(np.vstack((self.params[self.ixp], self.se_params, z, p)).T,
                                columns=res_cols, index=param_labels)
        self.L = pd.DataFrame(self.L, index=self.cols, columns=fcols)
        
        if compute_factors:
            factor_coefs, factors = self.compute_factors(factor_method)
            self.factor_coefs = pd.DataFrame(factor_coefs, index=self.cols, columns=fcols)
            self.factors = pd.DataFrame(factors, index=self.inds, columns=fcols)
    
    def plot_loadings(self, plot_kws=None):
        default_plot_kws = dict(vmin=-1, vmax=1, center=0,
                                cmap=plt.cm.bwr)
        plot_kws = {} if plot_kws is None else plot_kws
        plot_kws = {**default_plot_kws, **plot_kws}
        ax = sns.heatmap(self.L, **plot_kws)
        return ax
    
        
    def compute_factors(self, method="tenberge"):
        """
        

        Parameters
        ----------
        method : str, optional
            Method to compute factors. The default is "regression".

        Returns
        -------
        factor_coefs : ndarray
            Array to compute factors.
        factors : ndarray
            Array of factors.

        """
        X = self.X - np.mean(self.X, axis=0)
        R = corr(X)
        L, Phi = self.L, self.Phi
        A = L.dot(Phi)
        if method == "thurstone":
            factor_coefs = np.linalg.solve(R, A)
        elif method == "tenberge":
            Phi_sq = mat_sqrt(Phi)
            R_isq = inv_sqrt(R)
            P = L.dot(Phi_sq)
            C = R_isq.dot(P).dot(inv_sqrt(P.T.dot(np.linalg.solve(R, P))))
            factor_coefs = R_isq.dot(C).dot(Phi_sq)
        factors = X.dot(factor_coefs)
        return factor_coefs, factors



                