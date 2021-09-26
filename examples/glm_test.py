# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 21:39:29 2020

@author: lukepinkel
"""

import numpy as np
import pandas as pd
from pystats.utilities.random import exact_rmvnorm, vine_corr
from pystats.pyglm.glm import (GLM, Binomial, LogitLink, Poisson, LogLink, 
                               Gamma, InverseGaussian, PowerLink,
                               Gaussian, IdentityLink)


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
    for key in param_vals.keys():
        param_close[key] = np.allclose(param_vals[key], models[key].params)
        
    assert(np.all(param_close.values()))
    
    
    
    
    
    
