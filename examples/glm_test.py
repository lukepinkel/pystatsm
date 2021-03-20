# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 21:39:29 2020

@author: lukepinkel
"""

import patsy
import numpy as np
import scipy as sp
import scipy.stats
import pandas as pd
import statsmodels.api as sm
from pystats.utilities.random_corr import exact_rmvnorm, vine_corr
from pystats.utilities import numerical_derivs 

from pystats.pyglm.glm import (GLM, Binomial, LogitLink, Poisson, LogLink, 
                               Gamma, ReciprocalLink, InverseGaussian, PowerLink,
                               NegativeBinomial, Gaussian, IdentityLink)


response_dists = ['Gaussian', 'Binomial', 'Poisson', 'Gamma', "InvGauss"]

n_obs, nx, r = 1000, 4, 0.5
n_true = nx//4
R = vine_corr(nx, 10)/10
X = {}

X1 = exact_rmvnorm(R, n_obs)
X2 = exact_rmvnorm(R, n_obs)
X2 = X2 - np.min(X2) + 0.1

X['Gaussian'] = X1.copy()
X['Binomial'] = X1.copy()
X['Poisson'] = X1.copy()
X['Gamma'] = X2.copy()
X['InvGauss'] = X2.copy()

beta = dict(Binomial=np.zeros(nx), Poisson=np.zeros(nx), Gamma=np.zeros(nx),
            Gaussian=np.zeros(nx))
beta['Gaussian'][:n_true*2] = np.concatenate((0.5*np.ones(n_true), -0.5*np.ones(n_true)))
beta['Binomial'][:n_true*2] = np.concatenate((0.5*np.ones(n_true), -0.5*np.ones(n_true)))
beta['Poisson'][:n_true*2] = np.concatenate((0.5*np.ones(n_true), -0.5*np.ones(n_true)))
beta['Gamma'][:n_true*2] = np.concatenate((0.1*np.ones(n_true), 0.1*np.ones(n_true)))
beta['InvGauss'] = beta['Gamma'].copy()

for dist in response_dists:
    beta[dist] = beta[dist][np.random.choice(nx, nx, replace=False)]
eta = {}
eta_var = {}
u_var = {}
u = {}
linpred = {}

for dist in response_dists:
    eta[dist] = X[dist].dot(beta[dist])
    eta_var[dist] = eta[dist].var()
    u_var[dist] = np.sqrt(eta_var[dist] * (1.0 - r) / r)
    u[dist] = np.random.normal(0, u_var[dist], size=(n_obs))
    if dist in ['Gamma', 'InvGauss']:
        u[dist] -= u[dist].min()
        u[dist]+=0.01
    linpred[dist] = u[dist]+eta[dist]

Y = {}
Y['Gaussian'] = IdentityLink().inv_link(linpred['Gaussian'])
Y['Binomial'] = np.random.binomial(n=10, p=LogitLink().inv_link(linpred['Binomial']))/10.0
Y['Poisson'] = np.random.poisson(lam=LogLink().inv_link(linpred['Poisson']))
Y['Gamma'] = np.random.gamma(shape=LogLink().inv_link(linpred['Gamma']), scale=2.0)
Y['InvGauss'] = np.random.wald(mean=PowerLink(-2).inv_link(eta['InvGauss']), scale=2.0)

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

sm_models = {}
sm_models['Gaussian'] = sm.formula.glm(formula, data=data['Gaussian'], family=sm.families.Gaussian()).fit()
sm_models['Binomial'] = sm.formula.glm(formula, data=data['Binomial'], family=sm.families.Binomial(), n_trials=np.ones(n_obs)*10).fit()
sm_models['Poisson'] = sm.formula.glm(formula, data=data['Poisson'], family=sm.families.Poisson()).fit()
sm_models['Gamma'] = sm.formula.glm(formula, data=data['Gamma'], family=sm.families.Gamma()).fit()
sm_models['InvGauss'] = sm.formula.glm(formula, data=data['InvGauss'], family=sm.families.InverseGaussian()).fit()


comparison = {}

for dist in response_dists:
    a = sm_models[dist].params
    b = models[dist].params
    if len(a)!=len(b):
        a = np.concatenate([a, np.atleast_1d(np.log(sm_models[dist].scale))])
    comparison[dist] = pd.DataFrame(np.vstack((a, b)).T)
