#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 15:25:10 2022

@author: lukepinkel
"""

import numpy as np
from ..utilities.random import r_lkj, exact_rmvnorm
from ..utilities.data_utils import eighs


def inv_sqrt(arr):
    u, V  = eighs(arr)
    u[u>1e-12] = 1.0 / np.sqrt(u[u>1e-12])
    arr = (V * u).dot(V.T)
    return arr




def _cca(X, Y, n_comps=None, center=False, standardize=False):
    if center:
        X = X - np.mean(X, axis=0)
        Y = Y - np.mean(Y, axis=0)
    if standardize:
        X = X / np.std(X, axis=0)
        Y = Y / np.std(Y, axis=0)
    n_obs = X.shape[0]
    Sxy = np.dot(X.T, Y) / n_obs
    Sxx = np.dot(X.T, X) / n_obs
    Syy = np.dot(Y.T, Y) / n_obs
   
    Sxx_isq, Syy_isq = inv_sqrt(Sxx), inv_sqrt(Syy)
    U, s, Vt = np.linalg.svd(Sxx_isq.dot(Sxy).dot(Syy_isq))
    V = Vt.T
    Wx = Sxx_isq.dot(U)
    Wy = Syy_isq.dot(V)
    rhos = s
    if n_comps is not None:
        Wx, Wy, rhos = Wx[:, :n_comps], Wy[:, :n_comps], rhos[:n_comps]
    Lx, Ly = Sxx.dot(Wx), Syy.dot(Wy)
    return Lx, Ly, Wx, Wy, rhos

class SimCCA(object):
    
    def __init__(self,  n_xvars, n_yvars, rhos=None, x_corr=None, y_corr=None, 
                 rng=None, seed=None):
        rng = np.random.default_rng(seed) if rng is None else rng
        self.seed, self.rng = seed, rng
        self.n_xvars = self.p = n_xvars
        self.n_yvars = self.q = n_yvars
        rhos = 1/(np.linspace(1, n_xvars, n_xvars)+1/19) if rhos is None else rhos
        Sxx = r_lkj(eta=1.0, n=1, dim=n_xvars, rng=rng)[0, 0] if x_corr is None else x_corr
        Syy = r_lkj(eta=1.0, n=1, dim=n_yvars, rng=rng)[0, 0] if y_corr is None else y_corr
        self._set_corrmats(Sxx, Syy, rhos)
        
        
    def _set_corrmats(self, Sxx, Syy, rhos):
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
    
    def simulate_data(self, n_obs=1000, exact=False):
        if exact:
            data = exact_rmvnorm(self.S, n=n_obs, seed=self.seed)
        else:
            data = self.rng.multivariate_normal(mean=np.zeros(self.p+self.q),
                                                cov=self.S, size=n_obs)
            data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
        X, Y = data[:, :self.p], data[:, self.p:]
        return X, Y

        
        
    