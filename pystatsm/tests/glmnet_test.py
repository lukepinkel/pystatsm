# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 16:39:48 2021

@author: lukepinkel
"""

import numpy as np 
import scipy as sp
import scipy.stats
from pystatsm.pyglmnet.glmen2 import ElasticNetGLM
from pystatsm.pyglm.families import Gaussian, Binomial
from pystatsm.utilities.random import exact_rmvnorm

def test_elnet():
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
    
    model = ElasticNetGLM(X=X, y=y, family=Binomial(), alpha=alpha)
    model.fit(cv=10)
    
    theta = np.array([-0.17769525, -0.80343996,  0.        ,  0.68259049,  0.        ,
                      -0.91795782,  0.        ,  0.67142503,  0.24607844])
    assert(np.allclose(model.beta, theta))
    
    
    
    model = ElasticNetGLM(X=X, y=eta, family=Gaussian(), alpha=alpha)
    model.fit(cv=10)
    
    theta = np.array([ 0.0045532 , -1.0218214 , -0.01234576,  0.93876264,  0.        ,
                      -0.93615556,  0.        ,  0.9787038 ,  0.02896793])
    assert(np.allclose(model.beta, theta))

    
    
    
    