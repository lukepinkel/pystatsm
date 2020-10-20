#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 04:18:35 2020

@author: lukepinkel
"""


import arviz as az # analysis:ignore
import numpy as np # analysis:ignore
import scipy as sp # analysis:ignore
import scipy.stats # analysis:ignore
import pandas as pd # analysis:ignore
from ..pylmm.glmm_mcmc import MixedMCMC # analysis:ignore
from .tests.test_data_mcmc import construct_model_matrices, vine_corr, multi_rand # analysis:ignore

def to_arviz_dict(samples, var_dict, burnin=0):
    az_dict = {}
    for key, val in var_dict.items():
        az_dict[key] = samples[:, burnin:, val]
    return az_dict    

n_grp, n_per = 300, 30
n_obs = n_grp * n_per
formula = "y~x+(1|id1)"
beta = np.array([-0.1, 0.5])
df = pd.DataFrame(np.zeros((n_obs, 6)), columns=['x', 'id1', 'u', 'eta', 'mu', 'y'])
df['id1'] = np.repeat(np.arange(n_grp), n_per)
r = np.sqrt(0.9)
_, Z, _, _ = construct_model_matrices(formula, df)
u = np.random.normal(0, 1, size=n_grp)
u = (u - u.mean()) / u.std() * np.sqrt(2.0)
df['u'] = Z.dot(u)
df['x'] = sp.stats.norm(-df['u'], np.sqrt((1-0.7)/0.7*df['u'].var())).rvs()
X, Z, _, _ = construct_model_matrices(formula, df)
lp = X.dot(beta) + Z.dot(u)
lpvar = lp.var()
rsq = r**2
np.sqrt((1-rsq)/rsq*lpvar)
df['eta'] = sp.stats.norm(lp, np.sqrt((1-rsq)/rsq*lpvar)).rvs()
df['mu'] = np.exp(df['eta']) / (1 + np.exp(df['eta']))
df['y'] = sp.stats.binom(n=1, p=df['mu']).rvs()



model = MixedMCMC(formula, df)

n_samples, n_chains = 15_000, 8
samples = np.zeros((n_chains, n_samples, model.n_params))
samples_u = np.zeros((n_chains, n_samples, model.n_re))
samples_p = np.zeros((n_chains, n_samples, model.n_ob))

model.t_init[-1] = 20
for i in range(n_chains):
    samples[i], scnd = model.sample_slice_gibbs(n_samples, save_pred=True, save_u=True)
    samples_p[i] = scnd['pred']
    samples_u[i] = scnd['u']
    
samples = samples[:, :, :-1]
az_dict = to_arviz_dict(samples, {"$\\beta$":np.arange(2), 
                                  "$\\theta$":np.arange(2, 3)}, burnin=1000)

az_data = az.from_dict(az_dict)
summary = az.summary(az_data)
summary['sampling_effeciency'] = summary['ess_mean'] / np.product(samples.shape[:2])
ranef_summary = az.summary(samples_u)


print(summary)
az.plot_trace(az_data, var_names=['$\\theta$'])
az.plot_trace(az_data, var_names=['$\\beta$'])


yhat_summary = az.summary(samples_p)














