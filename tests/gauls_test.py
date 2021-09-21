# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 21:25:56 2021

@author: lukepinkel
"""


import numpy as np
import pandas as pd
from pystats.pygam.gauls import GauLS
from pystats.utilities.data_utils import dummy
from pystats.utilities.numerical_derivs import fo_fc_cd, so_gc_cd


rng = np.random.default_rng(123)

def test_gauls():
    n_obs = 20000
    df = pd.DataFrame(np.zeros((n_obs, 4)), columns=['x0', 'x1', 'x2', 'y'])
     
    df['x0'] = rng.choice(np.arange(5), size=n_obs, p=np.ones(5)/5)
    df['x1'] = rng.uniform(-1, 1, size=n_obs)
    df['x2'] = rng.uniform(-1, 1, size=n_obs)
    
    u0 =  dummy(df['x0']).dot(np.array([-0.2, 0.2, -0.2, 0.2, 0.0]))
    f1 = (3.0 * df['x1']**3 - 2.43 * df['x1'])
    f2 = -(3.0 * df['x2']**3 - 2.43 * df['x2']) 
    f3 = (df['x2'] - 1.0) * (df['x2'] + 1.0)
    eta =  u0 + f1 + f2
    mu = eta.copy() 
    tau = 1.0 / (np.exp(f3) + 0.01)
    df['y'] = rng.normal(loc=mu, scale=tau)
    
    
    model = GauLS("y~C(x0)+s(x1, kind='cr')+s(x2, kind='cr')", "y~1+s(x2, kind='cr')", df)
    model.fit()
    
    assert(model.opt.success==True)
    assert((np.abs(model.opt.grad)<1e-6).all())
    
    theta = np.array([4.235378, 4.165951, 7.181457])
    assert(np.allclose(theta, model.theta))
    
    eps = np.finfo(float).eps**(1/4)
    grad_close = np.allclose(model.gradient(theta), fo_fc_cd(model.reml, theta), atol=eps, rtol=eps)
    hess_close = np.allclose(model.hessian(theta), so_gc_cd(model.gradient, theta), atol=eps, rtol=eps)
    assert(grad_close)
    assert(hess_close)