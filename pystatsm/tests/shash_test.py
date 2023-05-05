#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 00:02:41 2023

@author: lukepinkel
"""

import numpy as np
import scipy as sp
import scipy.special
from pystatsm.utilities.indexing_utils import fill_tensor
from pystatsm.misc import shash1, shashr
from pystatsm.utilities.numerical_derivs import jac_approx

def test_shash1():
    rng = np.random.default_rng(1234)
    params = np.array([-0.5, 0.7, 1.5, -1.2])
    m, s, t, v = params
    y = shash1.rvs(m, s, v, t, size=(10), rng=rng)
    
    L, L1, L2, L3, L4 = shash1.logpdf(y, params=params, derivs=4)
    
    L2 = fill_tensor(L2, (4,4))
    L3 = fill_tensor(L3, (4,4,4))
    L4 = fill_tensor(L4, (4,4,4,4))
    
    L1n = jac_approx(lambda x: shash1.logpdf(y, params=x, derivs=0), params)
    L2n = jac_approx(lambda x: shash1.logpdf(y, params=x, derivs=1)[1], params)
    L3n = jac_approx(lambda x: fill_tensor(shash1.logpdf(y, params=x, derivs=2)[2], (4, 4,)), params)
    L4n = jac_approx(lambda x: fill_tensor(shash1.logpdf(y, params=x, derivs=3)[3], (4, 4, 4)), params)
    
    assert(np.allclose(L1n, L1))
    assert(np.allclose(L2n, L2))
    assert(np.allclose(L3n, L3))
    assert(np.allclose(L4n, L4))
    
    y = shash1.rvs(m, s, v, t, size=(20_000), rng=rng)

    def loglike(params):
        L = -shash1.logpdf(y, params=params, derivs=0) 
        return np.sum(L)
    
    def grad(params):
        _, g = shash1.logpdf(y, params=params, derivs=1) 
        return np.sum(-g, axis=0)
    
    def hess(params):
        _, _, H = shash1.logpdf(y, params=params, derivs=2) 
        H = fill_tensor(H, (4,4))
        return np.sum(-H, axis=0)
    
    
    res = sp.optimize.minimize(loglike, np.array([0.0, 1.0, 0.1, 0.0]), 
                               jac=grad, hess=hess, method="trust-constr",
                               options=dict(verbose=3))

    assert(res.success)
    y = shash1.rvs(m, s, v, t, size=(500_000), rng=rng)
    
    assert(np.allclose(np.mean(y), shash1.mean(m,s,v,t), rtol=1e-3, atol=1e-2))
    assert(np.allclose(np.var(y), shash1.variance(m,s,v,t), rtol=1e-3, atol=1e-2))
    assert(np.allclose(sp.stats.skew(y), shash1.skew(m,s,v,t), rtol=1e-3, atol=1e-2))
    assert(np.allclose(sp.stats.kurtosis(y, fisher=False), shash1.kurt(m,s,v,t), rtol=1e-3, atol=1e-2))

    
def test_shashr():
    rng = np.random.default_rng(1234) 
    params = np.array([-0.5, 0.7, 1.5, -1.2])
    params[1] = np.exp(params[1])
    params[3] = np.exp(params[3])
    m, s, t, v = params
    y = shashr.rvs(m, s, v, t, size=(10), rng=rng)
    L, L1, L2, L3, L4 = shashr.logpdf(y, params=params, derivs=4)
    
    L2 = fill_tensor(L2, (4,4))
    L3 = fill_tensor(L3, (4,4,4))
    L4 = fill_tensor(L4, (4,4,4,4))
    
    L1n = jac_approx(lambda x: shashr.logpdf(y, params=x, derivs=0), params)
    L2n = jac_approx(lambda x: shashr.logpdf(y, params=x, derivs=1)[1], params)
    L3n = jac_approx(lambda x: fill_tensor(shashr.logpdf(y, params=x, derivs=2)[2], (4, 4,)), params)
    L4n = jac_approx(lambda x: fill_tensor(shashr.logpdf(y, params=x, derivs=3)[3], (4, 4, 4)), params)
    assert(np.allclose(L1n, L1))
    assert(np.allclose(L2n, L2))
    assert(np.allclose(L3n, L3))
    assert(np.allclose(L4n, L4))
    
    
    y = shashr.rvs(m, s, v, t, size=(20_000))
    
  
    def loglike(params):
        L = -shashr.logpdf(y, params=params, derivs=0) 
        return np.sum(L)
    
    def grad(params):
        _, g = shashr.logpdf(y, params=params, derivs=1) 
        return np.sum(-g, axis=0)
    
    def hess(params):
        _, _, H = shashr.logpdf(y, params=params, derivs=2) 
        H = fill_tensor(H, (4,4))
        return np.sum(-H, axis=0)
    
    
    res = sp.optimize.minimize(loglike, np.array([0.0, 1.0, 0.1, 0.0]), 
                               jac=grad, hess=hess, method="trust-constr",
                               options=dict(verbose=3))
    
    assert(res.success)
    y = shashr.rvs(m, s, v, t, size=(500_000), rng=rng)
    
    assert(np.allclose(np.mean(y), shashr.mean(m,s,v,t), rtol=1e-3, atol=1e-2))
    assert(np.allclose(np.var(y), shashr.variance(m,s,v,t), rtol=1e-3, atol=1e-2))
    assert(np.allclose(sp.stats.skew(y), shashr.skew(m,s,v,t), rtol=1e-3, atol=1e-2))
    assert(np.allclose(sp.stats.kurtosis(y, fisher=False), shashr.kurt(m,s,v,t), rtol=1e-3, atol=1e-2))

    


    

