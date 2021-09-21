# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 13:28:38 2021

@author: lukepinkel
"""

import numpy as np
import pandas as pd
from pystats.utilities.random import exact_rmvnorm
from pystats.pyglm.betareg import BetaReg, LogitLink, LogLink
from pystats.pyglm.clm import CLM

from pystats.utilities.numerical_derivs import fo_fc_cd, so_gc_cd

SEED = 1234
rng = np.random.default_rng(SEED)

def test_betareg():
    n_obs = 10_000
    X = exact_rmvnorm(np.eye(4)/100, n=n_obs, seed=SEED)
    Z = exact_rmvnorm(np.eye(2)/100, n=n_obs, seed=SEED)
    betam = np.array([4.0, 1.0, -1.0, -2.0])
    betas = np.array([2.0, -2.0])
    etam, etas = X.dot(betam)+1.0, 2+Z.dot(betas)#np.tanh(Z.dot(betas))/2.0 + 2.4
    mu, phi = LogitLink().inv_link(etam), LogLink().inv_link(etas)     
    a = mu * phi
    b = (1.0 - mu) * phi
    y = rng.beta(a, b)
    
    
    xcols = [f"x{i}" for i in range(1, 4+1)]
    zcols = [f"z{i}" for i in range(1, 2+1)]
    data = pd.DataFrame(np.hstack((X, Z)), columns=xcols+zcols)
    data["y"] = y
    
    m_formula = "y~1+"+"+".join(xcols)
    s_formula = "y~1+"+"+".join(zcols)
    
    model = BetaReg(m_formula=m_formula, s_formula=s_formula, data=data)
    model.fit()
    theta = np.array([0.99819859,  3.92262116,  1.02091902, -0.98526682, -1.9795528,
                      1.98535573,  2.06533661, -2.06805411])
    assert(np.allclose(model.theta, theta))
    g1 = fo_fc_cd(model.loglike, model.theta*0.95)
    g2 = model.gradient(model.theta*0.95)
    assert(np.allclose(g1, g2))
    
    H1 = so_gc_cd(model.gradient, model.theta)
    H2 = model.hessian(model.theta)
    
    assert(np.allclose(H1, H2))
    assert(model.optimizer.success==True)
    assert((np.abs(model.optimizer.grad)<1e-5).all())
    
def test_clm():    
    n_obs, n_var, rsquared = 10_000, 8, 0.25
    S = np.eye(n_var)
    X = exact_rmvnorm(S, n=n_obs, seed=1234)
    beta = np.zeros(n_var)
    beta[np.arange(n_var//2)] = rng.choice([-1., 1., -0.5, 0.5], n_var//2)
    var_names = [f"x{i}" for i in range(1, n_var+1)]
    
    eta = X.dot(beta)
    eta_var = eta.var()
    
    scale = np.sqrt((1.0 - rsquared) / rsquared * eta_var)
    y = rng.normal(eta, scale=scale)
    
    df = pd.DataFrame(X, columns=var_names)
    df["y"] = pd.qcut(y, 7).codes
    
    formula = "y~-1+"+"+".join(var_names)
    
    model = CLM(frm=formula, data=df)
    model.fit()
    theta = np.array([-2.08417224, -1.08288221, -0.34199706,  0.34199368,  1.08217316,
                      2.08327387,  0.37275823,  0.37544884,  0.3572407 ,  0.71165265,
                      0.0086888 , -0.00846944,  0.00975741,  0.01257564])
    assert(np.allclose(theta, model.params))
    params_init, params = model.params_init.copy(),  model.params.copy()
    
    tol = np.finfo(float).eps**(1/3)
    np.allclose(model.gradient(params_init), fo_fc_cd(model.loglike, params_init))
    np.allclose(model.gradient(params), fo_fc_cd(model.loglike, params), atol=tol)
    
    
    np.allclose(model.hessian(params_init), so_gc_cd(model.gradient, params_init))
    np.allclose(model.hessian(params), so_gc_cd(model.gradient, params))
    
