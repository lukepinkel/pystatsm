#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 15:25:10 2022

@author: lukepinkel
"""

import numpy as np
from ..utilities.random import r_lkj, exact_rmvnorm
from ..utilities.linalg_operations import cproject, eighs

def normalize_wrt_corr(X, S):
    w = np.einsum("ij,ik,kj->j", X, S, X, optimize=True)
    X = X / np.sqrt(w)
    return X
class SimCCA(object):
    
    def __init__(self,  n_xvars, n_yvars, rhos=None, x_corr=None, y_corr=None, 
                 x_coefs=None, y_coefs=None, rng=None, seed=None):
        rng = np.random.default_rng(seed) if rng is None else rng
        self.seed, self.rng = seed, rng
        self.n_xvars = self.p = n_xvars
        self.n_yvars = self.q = n_yvars
        rhos = 1/(np.linspace(1, n_xvars, n_xvars)+1/19) if rhos is None else rhos
        Sxx = r_lkj(eta=1.0, n=1, dim=n_xvars, rng=rng)[0, 0] if x_corr is None else x_corr
        Syy = r_lkj(eta=1.0, n=1, dim=n_yvars, rng=rng)[0, 0] if y_corr is None else y_corr
        if x_coefs is None or y_coefs is None:
            self._set_corrmats_from_rho(Sxx, Syy, rhos)
        else:
            self._set_corrmats_from_coefs(Sxx, Syy, x_coefs, y_coefs, rhos)
    
    def _set_corrmats_from_coefs(self, Sxx, Syy, x_coefs, y_coefs, rhos):
        Wx, Wy = normalize_wrt_corr(x_coefs, Sxx), normalize_wrt_corr(y_coefs, Syy)
        C = np.einsum("ij,j,kj->ik", Wx, rhos, Wy) 
        Sxy = Sxx.dot(C).dot(Syy)
        S = np.block([[Sxx, Sxy], [Sxy.T, Syy]])
        self._C = C
        self.corr = self.S = S
        self.x_corr = self.Sxx = Sxx
        self.y_corr = self.Syy = Syy
        self.xy_corr = self.Sxy = Sxy
        self.rhos = rhos
        self.x_coefs = self.Wx = Wx
        self.y_coefs = self.Wy = Wy
        self.Lx, self.Ly = self.Sxx.dot(self.Wx), self.Syy.dot(self.Wy)
        self.Px, self.Py = cproject(self.Wx), cproject(self.Wy)

        
    def _set_corrmats_from_rho(self, Sxx, Syy, rhos):
        x_eig, A0 = eighs(Sxx)
        y_eig, B0 = eighs(Syy)
        
        A = A0 * (np.sqrt(1/x_eig))
        B = B0 * (np.sqrt(1/y_eig))
        
        B_inv = np.linalg.inv(B)
        
        B1 = B_inv[:self.n_xvars]
                        
        Sxy = np.linalg.inv(A.T).dot(np.diag(rhos)).dot(B1)
        S = np.block([[Sxx, Sxy], [Sxy.T, Syy]])
        self.corr = self.S = S
        self.x_corr = self.Sxx = Sxx
        self.y_corr = self.Syy = Syy
        self.xy_corr = self.Sxy = Sxy
        self.rhos = rhos
        self.x_coefs = self.Wx = A
        self.y_coefs = self.Wy = B
        self.y_coefs_inv = self.Wy_inv = B_inv
        self.Lx, self.Ly = self.Sxx.dot(self.Wx), self.Syy.dot(self.Wy)
        self.Px, self.Py = cproject(self.Wx), cproject(self.Wy)
    
    def simulate_data(self, n_obs=1000, exact=False):
        if exact:
            data = exact_rmvnorm(self.S, n=n_obs, seed=self.seed)
        else:
            data = self.rng.multivariate_normal(mean=np.zeros(self.p+self.q),
                                                cov=self.S, size=n_obs)
            data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
        X, Y = data[:, :self.p], data[:, self.p:]
        return X, Y

        
        
    