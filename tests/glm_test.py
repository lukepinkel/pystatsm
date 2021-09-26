# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 13:28:38 2021

@author: lukepinkel
"""

import numpy as np
import pandas as pd
from pystats.utilities.random import exact_rmvnorm, vine_corr
from pystats.pyglm.betareg import BetaReg, LogitLink, LogLink
from pystats.pyglm.clm import CLM
from pystats.pyglm.glm import (GLM, Binomial, Poisson, Gamma, InverseGaussian, PowerLink,
                               Gaussian, IdentityLink)
from pystats.pyglm.nb2 import NegativeBinomial
from pystats.pyglm.zimodels import ZIP
from pystats.utilities.numerical_derivs import fo_fc_cd, so_gc_cd



def test_betareg():
    SEED = 1234
    rng = np.random.default_rng(SEED)
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
    SEED = 1234
    rng = np.random.default_rng(SEED)
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
    
def test_glm():
    seed = 1234
    rng = np.random.default_rng(seed)
    response_dists = ['Gaussian', 'Binomial', 'Poisson', 'Gamma', "InvGauss"]
    
    n_obs, nx, r = 2000, 10, 0.5
    n_true = nx//4
    R = vine_corr(nx, 10, seed=seed)
    X = {}
    
    X1 = exact_rmvnorm(R, n_obs, seed=seed)
    X2 = exact_rmvnorm(R, n_obs, seed=seed)
    X2 = X2 - np.min(X2, axis=0) + 0.1
    
    X['Gaussian'] = X1.copy()
    X['Binomial'] = X1.copy()
    X['Poisson'] = X1.copy()
    X['Gamma'] = X1.copy()
    X['InvGauss'] = X2.copy()
    
    beta = dict(Binomial=np.zeros(nx), Poisson=np.zeros(nx), Gamma=np.zeros(nx), 
                Gaussian=np.zeros(nx), InvGauss=np.zeros(nx))
    beta['Gaussian'][:n_true*2] = np.concatenate((0.5*np.ones(n_true), -0.5*np.ones(n_true)))
    beta['Binomial'][:n_true*2] = np.concatenate((0.5*np.ones(n_true), -0.5*np.ones(n_true)))
    beta['Poisson'][:n_true*2] = np.concatenate((0.5*np.ones(n_true), -0.5*np.ones(n_true)))
    beta['Gamma'][:n_true*2] = np.concatenate((0.1*np.ones(n_true), -0.1*np.ones(n_true)))
    beta['InvGauss'][:n_true*2] = np.concatenate((0.1*np.ones(n_true), 0.1*np.ones(n_true)))
    
    for dist in response_dists:
        beta[dist] = beta[dist][rng.choice(nx, nx, replace=False)]
    eta = {}
    eta_var = {}
    u_var = {}
    u = {}
    linpred = {}
    
    for dist in response_dists:
        eta[dist] = X[dist].dot(beta[dist])
        eta_var[dist] = eta[dist].var()
        u_var[dist] = np.sqrt(eta_var[dist] * (1.0 - r) / r)
        u[dist] = rng.normal(0, u_var[dist], size=(n_obs))
        linpred[dist] = u[dist]+eta[dist]
        if dist in ['InvGauss']:
            linpred[dist] -= linpred[dist].min()
            linpred[dist] += 0.01
    
    Y = {}
    Y['Gaussian'] = IdentityLink().inv_link(linpred['Gaussian'])
    Y['Binomial'] = rng.binomial(n=10, p=LogitLink().inv_link(linpred['Binomial']))/10.0
    Y['Poisson'] = rng.poisson(lam=LogLink().inv_link(linpred['Poisson']))
    Y['Gamma'] = rng.gamma(shape=LogLink().inv_link(linpred['Gamma']), scale=3.0)
    Y['InvGauss'] = rng.wald(mean=PowerLink(-2).inv_link(eta['InvGauss']), scale=2.0)
    
    data = {}
    formula = "y~"+"+".join([f"x{i}" for i in range(1, nx+1)])
    for dist in response_dists:
        data[dist] = pd.DataFrame(np.hstack((X[dist], Y[dist].reshape(-1, 1))), 
                                  columns=[f'x{i}' for i in range(1, nx+1)]+['y'])
        
    
    models = {}
    models['Gaussian'] = GLM(formula=formula, data=data['Gaussian'], fam=Gaussian(), scale_estimator='NR')
    models['Binomial'] = GLM(formula=formula, data=data['Binomial'], fam=Binomial(weights=np.ones(n_obs)*10.0))
    models['Poisson'] = GLM(formula=formula, data=data['Poisson'], fam=Poisson())
    models['Gamma'] = GLM(formula=formula, data=data['Gamma'], fam=Gamma())
    models['Gamma2'] = GLM(formula=formula, data=data['Gamma'], fam=Gamma(), scale_estimator='NR')
    models['InvGauss'] = GLM(formula=formula, data=data['InvGauss'], fam=InverseGaussian(), scale_estimator='NR')
    
    models['Gaussian'].fit()
    models['Binomial'].fit()
    models['Poisson'].fit()
    models['Gamma'].fit()
    models['Gamma2'].fit()
    models['InvGauss'].fit()
    
    grad_conv = {}
    grad_conv["Gaussian"] = np.mean(models['Gaussian'].optimizer.grad**2)<1e-6
    grad_conv["Binomial"] = np.mean(models['Binomial'].optimizer.grad**2)<1e-6
    grad_conv["Poisson"] = np.mean(models['Poisson'].optimizer.grad**2)<1e-6
    grad_conv["Gamma"] = models['Gamma'].optimizer['|g|'][-1]<1e-6
    grad_conv["Gamma2"] = models['Gamma2'].optimizer['|g|'][-1]<1e-6
    grad_conv["InvGauss"] = np.mean(models['InvGauss'].optimizer.grad**2)<1e-6
    
    assert(np.all(grad_conv.values()))
    
    param_vals = {}
    param_vals["Gaussian"] = np.array([0.01677157,  0.01768816,  0.03232757, -0.50586418,  0.00538817,
                                       0.01215466,  0.46273009,  0.03222982,  0.51013559, -0.00482659,
                                       -0.44925714, -0.08297647])
    param_vals["Binomial"] = np.array([-0.04811123,  0.34608258,  0.02748488,  0.02109192, -0.35403311,
                                        0.37825192, -0.46275101,  0.00668586,  0.06837819,  0.00136615,
                                        0.00321255])
    param_vals["Poisson"] = np.array([ 0.78523498, -0.52630851, -0.0407732 ,  0.02971785, -0.03919242,
                                      -0.01845692,  0.34397533, -0.55594235,  0.0257876 ,  0.42205263,
                                       0.13051603])
    param_vals["Gamma"] = np.array([ 0.33020564, -0.00496934, -0.01392126,  0.03581743, -0.01186388,
                                     0.03645015, -0.00609281, -0.01056508,  0.00163984, -0.03324063,
                                    -0.00937269])
    param_vals["Gamma2"] = np.array([ 0.33020564, -0.00496934, -0.01392126,  0.03581743, -0.01186388,
                                      0.03645015, -0.00609281, -0.01056508,  0.00163984, -0.03324063,
                                     -0.00937269,  0.09260053])
    param_vals["InvGauss"] = np.array([ 0.51658718, -0.03040851,  0.14254292,  0.10087636,  0.05071923,
                                       -0.05297573, -0.04039982, -0.04293772,  0.1251764 , -0.02370386,
                                        0.01912702, -0.66386179])
    param_close = {}
    grad_close = {}
    hess_close = {}
    for key in param_vals.keys():
        m = models[key]
        
        param_close[key] = np.allclose(param_vals[key], m.params)
        x = m.params * 0.98
        grad_close[key] = np.allclose(fo_fc_cd(m.loglike, x), m.gradient(x))
        hess_close[key] = np.allclose(so_gc_cd(m.gradient, x), m.hessian(x))
        
    assert(np.all(param_close.values()))
    assert(np.all(grad_conv.values()))
    assert(np.all(grad_close.values()))
    assert(np.all(hess_close.values()))


def test_nb2(): 
    seed = 1234
    rng = np.random.default_rng(seed)
    
    n_obs, n_var, n_nnz, rsq, k = 2000, 20, 4, 0.9**2, 4.0
    X = exact_rmvnorm(np.eye(n_var), n=n_obs, seed=seed)
    beta = np.zeros(n_var)
    
    bv = np.array([-1.0, -0.5, 0.5, 1.0])
    bvals = np.tile(bv, n_nnz//len(bv))
    if n_nnz%len(bv)>0:
        bvals = np.concatenate([bvals, bv[:n_nnz%len(bv)]])
        
    beta[:n_nnz] = bvals
    eta = X.dot(beta) / np.sqrt(np.sum(beta**2))
    lpred = rng.normal(eta, scale=np.sqrt(eta.var()*(1.0 - rsq) / rsq))
    mu = np.exp(lpred)
    var = mu + k * mu**2
    n = - mu**2 / (mu - var)
    p = mu / var
    y = rng.negative_binomial(n=n, p=p)
    
    
    xcols = [f"x{i}" for i in range(1, n_var+1)]
    data = pd.DataFrame(X, columns=xcols)
    data['y'] = y
    
    formula = "y~1+"+"+".join(xcols)
    model = NegativeBinomial(formula=formula, data=data)
    params_init = model.params.copy() + 0.01
    model.fit()
    params = model.params.copy()
    
    theta = np.array([ 0.13049303, -0.64878454, -0.30956394,  0.2903795 ,  0.58677555,
                      -0.03022705,  0.03989469,  0.01182953, -0.00498391,  0.00788808,
                      -0.04198716, -0.00162041,  0.01523861, -0.00401566, -0.02547227,
                      -0.07309814, -0.05574522,  0.00938691, -0.0034148 , -0.01254539,
                      -0.05221309,  1.41286364])
    
    g_num1 = fo_fc_cd(model.loglike, params_init)
    g_ana1 = model.gradient(params_init)
    
    g_num2 = fo_fc_cd(model.loglike, params)
    g_ana2 = model.gradient(params)
    
    H_num1 = so_gc_cd(model.gradient, params_init)
    H_ana1 = model.hessian(params_init)
    
    H_num2 = so_gc_cd(model.gradient, params)
    H_ana2 = model.hessian(params)
    
    assert(np.allclose(model.params, theta))
    assert(np.allclose(g_num1, g_ana1))
    assert(np.allclose(g_num2, g_ana2, atol=1e-4))
    assert(np.allclose(H_num1, H_ana1))
    assert(np.allclose(H_num2, H_ana2))
    assert(model.opt_full.success)
    
    
def test_zip():
    seed = 1234
    rng = np.random.default_rng(seed)
    X = exact_rmvnorm(vine_corr(3, 5, seed=seed), seed=seed)
    Z = exact_rmvnorm(vine_corr(2, 5, seed=seed), seed=seed)
    
    b, a = rng.normal(0.0, 0.5, size=(3)), rng.normal(size=(2))
    u = np.exp(Z.dot(a))
    
    prob = u / (1.0 + u)
    ybin = rng.binomial(1, p=prob)
    y = np.zeros(len(ybin), dtype=float)
    
    mu = np.exp(X.dot(b))
    q =  rng.poisson(mu[ybin==1])
    y[ybin==1] = q*1.0
    
    model = ZIP(X, y, Z)
    model.fit(opt_kws=dict(verbose=3, gtol=1e-9, xtol=1e-200))
    theta = np.array([-0.81434379,  0.02346997,  0.34179484,  0.00402158, -0.82916627])
    
    
    g1 = fo_fc_cd(model.loglike, model.params*0.98)
    g2 = model.gradient(model.params*0.98)
    H1 = so_gc_cd(model.gradient, model.params*0.98)
    H2 = model.hessian(model.params*0.98)
        
    assert(np.allclose(g1, g2))
    assert(np.allclose(H1, H2))
    assert(np.allclose(model.params, theta))
        
        