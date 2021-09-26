# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 22:53:50 2021

@author: lukepinkel
"""


import arviz as az
import numpy as np
import scipy as sp
import scipy.stats
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pystats.pylmm.glmm_mcmc import MixedMCMC
from pystats.pylmm.sim_lmm import MixedModelSim
from pystats.utilities.linalg_operations import invech



def test_glmm_mcmc():
    rng = np.random.default_rng(1234)
    formula1 = "y~x1+x2+(1+x3|id1)"
    model_dict1 = {}
    n_grp1 = 200
    n_per1 = 20
    model_dict1['gcov'] = {'id1':invech(np.array([1.0, -0.5, 1.0]))}
    model_dict1['ginfo'] = {'id1':dict(n_grp=n_grp1, n_per=n_per1)} 
    model_dict1['mu'] = np.zeros(3)
    model_dict1['vcov'] = np.eye(3)
    model_dict1['beta'] = np.array([-0.2, 1.0, -1.0])
    model_dict1['n_obs'] = n_grp1 * n_per1
    r = 0.8
    v = np.sum(model_dict1['beta']**2) + np.sum([np.trace(G) for G in model_dict1['gcov'].values()])
    s = np.sqrt(v * (1 - r) / r)
    msim = MixedModelSim(formula1, model_dict1, rng=rng)
    df = msim.df
    eta = msim.simulate_linpred()
    mu = np.exp(eta) / (1.0 + np.exp(eta))
    y = rng.normal(loc=eta, scale=s)
    
    tau = sp.stats.scoreatpercentile(y, np.linspace(0, 100, 5)[1:-1])
    thresholds = np.pad(tau, ((1, 1)), mode='constant', constant_values=[-np.inf, np.inf])
    
    df['y_normal'] = y
    df['y_binom1'] = rng.binomial(n=1, p=mu)
    df['y_binom10'] = rng.binomial(n=10, p=mu) / 10
    df['y_ordinal'] = pd.cut(y, thresholds).codes.astype(float)
    
    
    model = MixedMCMC("y_binom1~1+x1+x2+(1+x3|id1)", df, response_dist='bernoulli')
    model.sample(n_samples=12_000, burnin=2_000, n_chains=8)
    print(model.summary)
    assert(np.allclose(model.summary["r_hat"], 1, atol=1e-2))
    
    
    
    model2 = MixedMCMC("y_binom10~1+x1+x2+(1+x3|id1)", df, response_dist="binomial", weights=np.ones_like(df['y_binom10'])*10.0)
    model2.sample(n_samples=12_000, burnin=2_000, n_chains=8, sampling_kws=dict(n_adapt=6000, adaption_rate=1.02))
    print(model2.summary)
    assert(np.allclose(model2.summary["r_hat"], 1, atol=1e-2))
    
    model3 = MixedMCMC("y_ordinal~1+x1+x2+(1+x3|id1)", df, response_dist='ordinal_probit')
    model3.sample(n_samples=22_000, burnin=5_000, n_chains=8, sampling_kws=dict(n_adapt=5_000, adaption_rate=1.015))
    print(model3.summary)
    assert(np.allclose(model3.summary["r_hat"], 1, atol=1e-2))
    
    
    model4 = MixedMCMC("y_normal~1+x1+x2+(1+x3|id1)", df, response_dist='normal')
    model4.sample(n_samples=12_000, burnin=2000, n_chains=8)
    print(model4.summary)
    assert(np.allclose(model4.summary["r_hat"], 1, atol=1e-2))








