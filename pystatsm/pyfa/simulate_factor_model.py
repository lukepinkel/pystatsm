#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 22:04:01 2022

@author: lukepinkel
"""

import numpy as np
import scipy as sp
import scipy.linalg
import pandas as pd
from ..utilities.random import r_lkj, exact_rmvnorm

class FactorModelSim(object):
    
    def __init__(self, n_vars=12, n_facs=3, L=None, Phi=None, 
                 Psi=None, rng=None, seed=None):
        self.rng = np.random.default_rng(seed) if rng is None else rng
        self.n_vars = n_vars
        self.n_facs = n_facs
        self.L = L
        self.Phi = Phi
        self.Psi = Psi
        
    def _generate_loadings(self, random=False, r_kws=None):
        L = np.zeros((self.n_vars, self.n_facs))
        inds = list(zip(np.array_split(np.arange(self.n_vars), self.n_facs), np.arange(self.n_facs)))
        default_r_kws = dict(low=0.4, high=0.9)
        r_kws = {} if r_kws is None else r_kws
        r_kws = {**default_r_kws, **r_kws}
        for rows, col in inds:
            s =len(rows)
            if random:
                u =self.rng.uniform(size=s, **r_kws)
                u = np.sort(u)[::-1]
            else:
                u = np.linspace(1.0, 0.4, s)
            L[rows, col] = u
        return L
    
    def _generate_factor_corr(self, random=False, r_kws=None, rho=0.5):
        default_r_kws = dict(eta=2.0)
        r_kws = {} if r_kws is None else r_kws
        r_kws = {**default_r_kws, **r_kws}
        if random:
            Phi = r_lkj(n=1, dim=self.n_facs, rng=self.rng, **r_kws)
        else:
            Phi = rho**sp.linalg.toeplitz(np.arange(self.n_facs))
            s = (-1)**np.arange(self.n_facs).reshape(-1, 1)
            Phi = s*Phi*s.T
        return Phi
    
    def _generate_residual_cov(self, random=True, dist=None):
        if random:
            if dist is None:
                dist = lambda size: self.rng.uniform(low=0.3, high=0.7, size=size)
            psi = dist(self.n_vars)
        else:
            psi = np.linspace(0.3, 0.7, self.n_vars)
        Psi = np.diag(psi)
        return Psi
    
    def simulate_cov(self, loadings_kws=None, factor_corr_kws=None,
                     residual_cov_kws=None):
        loadings_kws = {} if loadings_kws is None else loadings_kws
        factor_corr_kws = {} if factor_corr_kws is None else factor_corr_kws
        residual_cov_kws = {} if residual_cov_kws is None else residual_cov_kws
        
        self.L = self._generate_loadings(**loadings_kws) if self.L is None else self.L
        self.Phi = self._generate_factor_corr(**factor_corr_kws) if self.Phi is None else self.Phi
        self.Psi = self._generate_residual_cov(**residual_cov_kws) if self.Psi is None else self.Psi
        self.C = sp.linalg.block_diag(self.Phi, self.Psi)

        self.Sigma = self.L.dot(self.Phi).dot(self.L.T) + self.Psi
        
    def simulate_data(self, n_obs=1000, exact=True):
        if exact:
            Z = exact_rmvnorm(n=n_obs, S=self.C)
        else:
            Z = self.rng.multivariate_normal(mean=np.zeros(self.n_vars+self.n_facs),
                                             cov=self.C, size=n_obs)
        X = Z[:, :self.n_facs].dot(self.L.T) + Z[:, self.n_facs:]
        return Z, X

        
    
    
        

    
        
        
    
    
            
            
            