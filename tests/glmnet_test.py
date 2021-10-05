# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 16:39:48 2021

@author: lukepinkel
"""

import numpy as np 
import scipy as sp
import scipy.stats
from pystatsm.pyglmnet.glmen import GLMEN
from pystatsm.utilities.random import exact_rmvnorm


def test_gaussian_elnet():
    rng = np.random.default_rng(123)
    n, p, q = 500, 8, 4
    S = 0.90**sp.linalg.toeplitz(np.arange(p))
    X = exact_rmvnorm(S, np.max((n, p+1)), seed=123)[:n]
    X/= np.sqrt(np.sum(X**2, axis=0)) / np.sqrt(X.shape[0])
    beta = np.zeros(p)
    
    bvals = np.tile([-1, 1.0], q//2)
    beta[np.arange(0, p, p//q)] = bvals 
    lpred = X.dot(beta)
    rsq = 0.5
    y = rng.normal(lpred, np.sqrt((1-rsq)/rsq * lpred.var()))
    alpha = 0.99
    
    model = GLMEN(X=X, y=y, family="gaussian")
    model.fit_cv(cv=10, n_iters=2000, lmin=np.exp(-9), alpha=alpha)
    
    theta = np.array([-0.99174026, -0.        ,  0.86848155, -0.        , -0.88097313,
                      0.        ,  0.95625015,  0.02576123])
    
    assert(np.allclose(model.beta, theta))
    
    
def test_binomial_elnet():
    rng = np.random.default_rng(123)
    n, p, q = 500, 8, 4
    S = 0.90**sp.linalg.toeplitz(np.arange(p))
    X = exact_rmvnorm(S, np.max((n, p+1)), seed=123)[:n]
    X/= np.sqrt(np.sum(X**2, axis=0)) / np.sqrt(X.shape[0])
    beta = np.zeros(p)
    
    bvals = np.tile([-1, 1.0], q//2)
    beta[np.arange(0, p, p//q)] = bvals 
    lpred = X.dot(beta)
    rsq = 0.5
    eta = rng.normal(lpred, np.sqrt((1-rsq)/rsq * lpred.var()))
    mu = np.exp(eta) / (1.0 + np.exp(eta))
    y = rng.binomial(n=1, p=mu)
    alpha = 0.99
    
    model = GLMEN(X=X, y=y, family="binomial")
    model.fit_cv(cv=10, n_iters=2000, lmin=np.exp(-9), alpha=alpha)
    
    theta = np.array([-0.75989124,  0.        ,  0.61455539, -0.01698942, -0.85347493,
                       0.        ,  0.66220927,  0.23815354])
    assert(np.allclose(model.beta, theta))
    
    
    
    