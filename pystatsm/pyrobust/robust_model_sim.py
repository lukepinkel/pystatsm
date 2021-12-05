# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 21:56:35 2021

@author: lukepinkel
"""
import numpy as np
import scipy as sp
import scipy.linalg
import scipy.stats
import pandas as pd
from ..utilities.random import exact_rmvnorm, students_t

class RobustRegressionModel:
    
    def __init__(self, n_obs=500, n_var=None, n_nnz=None, x_cov=None, rho=None, 
                 beta=None, beta_vals=None,  rsq=None, seed=None, 
                 rng=None, resid_dist=None):
        
        rng = np.random.default_rng() if rng is None else rng
        if resid_dist is None:
            resid_dist = lambda loc, scale, size: students_t(loc, scale, nu=2, 
                                                             size=size, rng=rng)
        if n_var is None and beta is not None:
            n_var = len(beta)
            n_nnz = np.sum(beta!=0)
            
        if x_cov is None:
            rho = 0.0 if rho is None else rho
            x_cov = rho**sp.linalg.toeplitz(np.arange(n_var))
        
        X = exact_rmvnorm(x_cov, n=np.max((n_obs, n_var+1)), seed=seed)[:n_obs]        
        if beta is None:
            if beta_vals is None:
                beta_vals = np.zeros(n_nnz)
                beta_vals[:n_nnz-n_nnz%2] = np.tile([1.0, -1.0], n_nnz//2)
                beta_vals[-(n_nnz%2)] = 1.0
        
            beta = np.zeros(n_var)
            beta[[x[0] for x in np.array_split(np.arange(n_var), n_nnz)]] = beta_vals
        
        lpred = X.dot(beta)
        
        rsq = 0.5 if rsq is None else rsq
        
        resid_scale = np.sqrt((1-rsq)/rsq * lpred.var())
        
        self.X = X
        self.beta = beta
        self.lpred = lpred
        self.rsq = rsq
        self.resid_scale = resid_scale
        self.lpred = lpred
        self.rng = rng
        self.n_obs = n_obs
        self.n_var = n_var
        self.n_nnz = n_nnz
        self.resid_dist = resid_dist
        self.data = pd.DataFrame(X, columns=[f"x{i}" for i in range(n_var)])
        self.data["y"] = 0
        
    
    def simulate_dependent(self, n_samples=1):
        size = (n_samples, self.n_obs)
        Y = self.resid_dist(loc=self.lpred, scale=self.resid_scale, size=size,
                            ).T
        if n_samples == 1:
            Y = Y[:, 0]
        return Y
