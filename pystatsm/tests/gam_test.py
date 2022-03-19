# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 21:19:41 2021

@author: lukepinkel
"""

import numpy as np
import pandas as pd
from pystatsm.pygam.gam import GAM, Gamma, Gaussian
from pystatsm.pyglm.links import LogLink
from pystatsm.utilities.data_utils import dummy
from pystatsm.utilities.numerical_derivs import fo_fc_cd, so_gc_cd
rng = np.random.default_rng(123)


def test_gam():
    n_obs = 10_000
    X = pd.DataFrame(np.zeros((n_obs, 4)), columns=['x0', 'x1', 'x2', 'y'])
     
    X['x0'] = rng.choice(np.arange(5), size=n_obs, p=np.ones(5)/5)
    X['x1'] = rng.uniform(-1, 1, size=n_obs)
    X['x2'] = rng.uniform(-2, 2, size=n_obs)
    
    u0 =  dummy(X['x0']).dot(np.array([-0.2, 0.2, -0.2, 0.2, 0.0]))
    f1 = (3.0 * X['x1']**3 - 2.43 * X['x1'])
    f2 = (X['x2']**3 - X['x2']) / 10
    eta =  u0 + f1 + f2
    mu = np.exp(eta) # mu = shape * scale
    shape = mu * 2.0
    rsq = 0.5
    X['yGamma'] = rng.gamma(shape=shape, scale=1.0)
    X['yGauss'] = rng.normal(loc=eta, scale = np.sqrt(eta.var()*(1.0 - rsq) / rsq)) 
    
    
     
    model1 = GAM("yGamma~C(x0)+s(x1, kind='cr')+s(x2, kind='cr')", X, family=Gamma(link=LogLink))
    model1.fit()
    
    assert(model1.opt.success==True)
    assert((np.abs(model1.opt.grad)<1e-5).all())
    
    theta = np.array([49.617907465401451, 407.234892157726449])
    assert(np.allclose(np.exp(model1.theta)[:-1], theta))
    
    theta = model1.theta.copy()
    eps = np.finfo(float).eps**(1/4)
    grad_close = np.allclose(model1.gradient(theta), fo_fc_cd(model1.reml, theta), atol=eps, rtol=eps)
    hess_close = np.allclose(model1.hessian(theta), so_gc_cd(model1.gradient, theta), atol=eps, rtol=eps)
    assert(grad_close)
    assert(hess_close)
    
    model2 = GAM("yGauss~C(x0)+s(x1, kind='cr')+s(x2, kind='cr')", X, family=Gaussian())
    model2.fit()
    
    assert(model2.opt.success==True)
    assert((np.abs(model2.opt.grad)<1e-5).all())
    
    theta = np.array([ 38.73804598, 213.32929014])
    assert(np.allclose(np.exp(model2.theta)[:-1], theta))
    
    theta = model2.theta.copy()
    eps = np.finfo(float).eps**(1/4)
    grad_close = np.allclose(model2.gradient(theta), fo_fc_cd(model2.reml, theta), atol=eps, rtol=eps)
    hess_close = np.allclose(model2.hessian(theta), so_gc_cd(model2.gradient, theta), atol=eps, rtol=eps)
    assert(grad_close)
    assert(hess_close)
    
