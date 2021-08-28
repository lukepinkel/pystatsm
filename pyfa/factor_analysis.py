# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 18:11:20 2021

@author: lukepinkel
"""


import numpy as np
import scipy as sp
import scipy.stats
import pandas as pd
from pystats.pyfa.rotation import rotate
from pystats.utilities.linalg_operations import vec, invec
from pystats.utilities.special_mats import nmat, dmat, lmat
from pystats.utilities.numerical_derivs import so_gc_cd
from pystats.utilities.data_utils import _check_type, cov, eighs

class FactorAnalysis(object):
   
    
    def __init__(self, X=None, S=None, n_obs=None, n_factors=None, 
                 rotation_method=None, **n_factor_kws):
        
        self._process_data(X, S, n_obs)
        self._get_n_factors(n_factors, **n_factor_kws)
        self._rotation_method = rotation_method
        self._make_params()
        self.E = np.eye(self.n_vars)
        self.Ik = np.eye(self.n_facs)
        self.Nk = nmat(self.n_facs).A
        self.Lk = lmat(self.n_facs).A
        self.Ip = np.eye(self.n_vars)
        self.Dp = dmat(self.n_vars).A
        self.Lp = lmat(self.n_vars).A
        self.Np = nmat(self.n_vars).A
        self.LpNp = np.dot(self.Lp, self.Np)
        self.d_inds = vec(self.Ip)==1

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
        
        self.X, self.cols, self.inds, self._is_pd = X, cols, inds, _is_pd
        self.S, self.V, self.u = S, V, u
        self.cols, self.inds, self._is_pd =cols, inds, _is_pd
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
        self.lix=np.arange(self.n_vars*self.n_facs)
        self.pix =np.arange(self.n_vars*self.n_facs, self.n_vars*self.n_facs+self.n_vars)
        L = self.V[:, :self.n_facs]
        psi = np.diag(self.S - np.dot(L, L.T))
        self.theta[self.lix] = vec(L)
        self.theta[self.pix] = np.log(psi)
    
    def model_matrices(self, theta):
        L = invec(theta[self.lix], self.n_vars, self.n_facs)
        Psi = np.diag(np.exp(theta[self.pix]))
        return L, Psi
    
    def implied_cov(self, theta):
        L, Psi = self.model_matrices(theta)
        Sigma = L.dot(L.T) + Psi
        return Sigma
    
    def loglike(self, theta):
        Sigma = self.implied_cov(theta)
        _, lndS = np.linalg.slogdet(Sigma)
        trSV = np.trace(np.linalg.solve(Sigma, self.S))
        ll = lndS + trSV
        return ll
    
    def gradient(self, theta):
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
    
    def hessian(self, theta):
        H = so_gc_cd(self.gradient, theta)
        return H
        
    def dsigma(self, theta):
        L, Psi = self.model_matrices(theta)
        DLambda = np.dot(self.LpNp, np.kron(L, self.Ip))
        DPsi = np.linalg.multi_dot([self.Lp, np.diag(vec(Psi)), self.Dp])    
        G = np.block([DLambda, DPsi])
        return G
    
    
    def _hessian(self, theta):
        L, Psi = self.model_matrices(theta)
        Sigma = L.dot(L.T) + Psi
        Sigma_inv = np.linalg.pinv(Sigma)
        Sdiff = self.S - Sigma
        d = np.diag(Sdiff)
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
                E[i, j] = 1.0
                T = E + E.T
                H11 = np.kron(Ik, T)
                Hij[self.lix, self.lix[:, None]] = H11

                E[i, j] = 0.0
                Hpp.append(Hij[:, :, None])
                Hij = Hij*0.0
        W = np.linalg.multi_dot([Dp.T, W1, Dp])
        dW = np.dot(d, W)
        Hp = np.concatenate(Hpp, axis=2) 
        H3 = np.einsum('k,ijk ->ij', dW, Hp)      
        H = (H1 + H2 - H3 / 2.0)*2.0
        return H
    
    def _unrotated_constraint_dervs(self, L, Psi):
        Psi_inv = np.diag(1.0 / np.diag(Psi))
        A = Psi_inv.dot(L)
        J1 = self.Lk.dot(self.Nk.dot(np.kron(self.Ik, A.T)))
        J2 = self.Lk.dot(((np.kron(A, A))[self.d_inds]).T)
        J = np.concatenate([J1, J2], axis=1)
        return J
    
    def _rotated_constraint_derivs(self, L, a, b, c, d):
        X = L * L
        Y = X * L

        D1 = 4.0 * a * np.sum(X) * L
        D2 = 4.0 * b * np.diag(np.sum(X, axis=1)).dot(L)
        D3 = 4.0 * c * L.dot(np.diag(np.sum(X, axis=0)))
        D4 = 2.0 * d * Y
        D = D1 + D2 + D3 +  D4
        J1 = self.Lk.dot(self.Nk.dot(np.kron(self.Ik, D.T)))
        p, k = self.n_vars, self.n_facs
        J2 = np.zeros(((k + 1) * k //2, p))
        J = np.concatenate([J1, J2], axis=1)
        return J
        
    def constraint_derivs(self, theta, gcff):
        L, Psi = self.model_matrices(theta)
        if self._rotation_method is not None:
            J = self._rotated_constraint_derivs(L, *gcff)
        else:
            J = self._unrotated_constraint_dervs(L, Psi)
        return J
        
    def _fit(self, hess=True, **opt_kws):
        hess = self.hessian if hess else None
        
        self.opt = sp.optimize.minimize(self.loglike, self.theta, jac=self.gradient,
                                        hess=hess, method='trust-constr', **opt_kws)
        self.theta = self.opt.x
        self.L, self.Psi = self.model_matrices(self.theta)
        if self._rotation_method is not None:
            self.L, self.T, self.gcff = rotate(self.L, self._rotation_method)
            self.theta[self.lix] = vec(self.L)
        else:
            self.gcff = None
        self.H = self.hessian(self.theta)
        self.J = self.constraint_derivs(self.theta, self.gcff)
        q = self.J.shape[0]
        self.Hc = np.block([[self.H, self.J.T], [self.J, np.zeros((q, q))]])
        self.se_theta = np.sqrt(1.0 / np.diag(np.linalg.pinv(self.Hc))[:-q]/self.n_obs)
        self.res = pd.DataFrame(np.vstack((self.theta, self.se_theta)).T,
                                columns=['param', 'SE'])
        self.res['z'] = self.res["param"] / self.res["SE"]
        self.res["p"] = sp.stats.norm(0, 1).sf(np.abs(self.res['z']))*2.0
        self.param_labels = []
        for j in range(self.L.shape[1]):
            for i in range(self.L.shape[0]):
                self.param_labels.append(f"L[{i}][{j}]")
        for i in range(self.L.shape[0]):
            self.param_labels.append(f"Psi[{i}]")
        self.res.index = self.param_labels
        self.L = pd.DataFrame(self.L, index=self.cols, 
                              columns=[f"Factor{i}" for i in range(self.n_facs)])





                
            
            