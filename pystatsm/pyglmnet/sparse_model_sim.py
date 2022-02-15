# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 08:38:04 2021

@author: lukepinkel
"""

import numpy as np
import scipy as sp
import scipy.linalg
import pandas as pd
from ..utilities.random import exact_rmvnorm

class SparseRegressionModel:
    
    def __init__(self, n_obs=500, n_var=200, n_nnz=10, x_cov=None, rho=None, 
                 beta=None, beta_vals=None, rsq=None, seed=None, rng=None, 
                 xw=None):
        rng = np.random.default_rng(seed) if rng is None else rng
        if x_cov is None:
            rho = 0.9 if rho is None else rho
            x_cov = 0.99**sp.linalg.toeplitz(np.arange(n_var))
        
        X = exact_rmvnorm(x_cov, n=np.max((n_obs, n_var+1)), seed=123)[:n_obs]
        if xw is not None:
            X = X * xw
            X = (X - X.mean(axis=0)) / X.std(axis=0)
        X/= (np.sqrt(np.einsum("ij,ij->j", X, X)) / np.sqrt(X.shape[0]))

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
        self.var_names = [f"x{i+1}" for i in range(n_var)]
        self.formula = "y~1+"+"+".join([f"x{i+1}" for i in range(n_var)])
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
        self.df = pd.DataFrame(X, columns=self.var_names)
        self.df["y"] = 0
    
    def simulate_dependent(self, response_dist="gaussian", n_samples=1, binom_n=1):
        size = (n_samples, self.n_obs)
        Y = self.rng.normal(loc=self.lpred, scale=self.resid_scale, size=size).T
        if n_samples == 1:
            Y = Y[:, 0]
        if response_dist=="binomial":
            u = np.exp(Y)
            mu = u / (1.0 + u)
            Y = self.rng.binomial(n=binom_n, p=mu) / binom_n
        return Y
    