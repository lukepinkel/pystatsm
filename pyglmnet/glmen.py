# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 01:23:16 2021

@author: lukepinkel
"""
import patsy
import numpy as np
import pandas as pd
from .binomial_eln import binom_glmnet
from .binomial_eln import cv_binom_glmnet
from .gaussian_eln import elnet as gaussian_glmnet
from .gaussian_eln import cv_glmnet as cv_gaussian_glmnet
from .eln_utils import plot_elnet_cv, process_cv
from ..utilities.linalg_operations import _check_np, _check_shape

class GLMEN:
    
    def __init__(self, formula=None, data=None, X=None, y=None, family=None):
        
        if formula is not None and data is not None:
            y, X = patsy.dmatrices(formula, data, return_type="dataframe")
            xcols, xinds = X.columns, X.index
            ycols, yinds = y.columns, y.index
            X, y = X.values, y.values[:, 0]
        elif X is not None and y is not None:
            if type(X) not in [pd.DataFrame, pd.Series]:
                xcols = [f'x{i}' for i in range(1, X.shape[1]+1)]
                xinds = np.arange(X.shape[0])
            else:
                xcols, xinds = X.columns, X.index
                X = X.values
            if type(y) not in [pd.DataFrame, pd.Series]:
                ycols = ['y']
                yinds = np.arange(y.shape[0])
            else:
                 ycols, yinds = y.columns, y.index
                 y = y.values
        
        if X.ndim==1:
            X = X[:, None]
            
        
        if family=="gaussian":
            self._fit = gaussian_glmnet
            self._fit_cv = cv_gaussian_glmnet
        elif family=="binomial":
            self._fit = binom_glmnet
            self._fit_cv = cv_binom_glmnet
        
       
        self.formula = formula
        self.data = data
        self.X, self.y = X, y
        self.xcols, self.xinds, self.ycols, self.yinds = xcols, xinds, ycols, yinds
        self.n_obs, self.n_var = self.X.shape
        self.Xinter = np.concatenate([np.ones((self.n_obs, 1)), self.X], axis=1)
        self.y = _check_shape(_check_np(y), 1)
        self.family = family
        
    def fit(self, lambda_, alpha=0.99, X=None, y=None, intercept=True, n_iters=1000, **kws):
        X = self.X if X is None else X
        y = self.y if y is None else y
        beta, active, fvals, _ = self._fit(X, y, lambda_, alpha, **kws)
        self.beta = beta 
        self.active = active
        self.fvals = fvals
    
    def fit_cv(self, cv=10, alpha=0.99, X=None, y=None, intercept=True, n_iters=1000, 
               lambdas=None, b=None, refit=True, **kws):
        X = self.X if X is None else X
        y = self.y if y is None else y
        b_path, f_path, lambdas, bfits, _ = self._fit_cv(cv, X, y, alpha, lambdas=lambdas,
                                                         n_iters=n_iters, b=b, 
                                                         refit=refit, **kws)
        self.beta_path = b_path
        self.f_path = f_path
        self.lambdas = lambdas
        self.bfits = bfits
        self.cvres, self.lambda_min = process_cv(f_path[:, :, 0], lambdas)
        self.n_nonzero = (bfits!=0).sum(axis=1)
        self.beta = bfits[self.cvres["mean"].idxmin()]
        
    
    def plot_cv(self, f_path=None, lambdas=None, bfits=None):
        f_path = self.f_path if f_path is None else f_path
        lambdas = self.lambdas if lambdas is None else lambdas
        bfits = self.bfits if bfits is None else bfits
        fig, ax = plot_elnet_cv(f_path, lambdas, bfits)
        return fig, ax
    
    def predict(self, beta=None, X=None, intercept=True):
        beta = self.beta if beta is None else beta
        X = self.X if X is None else X
        yhat = X.dot(beta) + intercept * self.y.mean()
        return yhat
    
    




        

        
        
        
        
