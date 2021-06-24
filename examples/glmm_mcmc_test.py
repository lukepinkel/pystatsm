# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 19:58:18 2021

@author: lukepinkel
"""

import arviz as az
import numpy as np
import scipy as sp
import scipy.stats
import pandas as pd
from pystats.pylmm.glmm_mcmc import MixedMCMC
from pystats.pylmm.sim_lmm import MixedModelSim
from pystats.utilities.linalg_operations import invech


pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 250)
pd.set_option('display.max_columns', 50)
np.set_printoptions(suppress=True)
rng = np.random.default_rng(1234)


r = 0.5**0.5
formula = "y~x1+x2+x3+x4+x5+(1+x5|id1)"
model_dict = {}
n_grp, n_per = 200, 20
n_obs = n_grp * n_per
model_dict['gcov'] = {'id1':invech(np.array([1.0, -0.5, 1.0]))}
model_dict['ginfo'] = {'id1':dict(n_grp=n_grp, n_per=n_per)} 
model_dict['mu'] = np.zeros(5)
model_dict['vcov'] = np.eye(5)
model_dict['beta'] = np.array([0.0, 1.0, -1.0, 1.0, -2.0, 1.0])
model_dict['n_obs'] = n_obs
v = np.sum(model_dict['beta']**2) + np.sum([np.trace(G) for G in model_dict['gcov'].values()])
s = np.sqrt(v * (1 - r) / r)
msim = MixedModelSim(formula, model_dict, rng=rng)
df = msim.df
eta = msim.simulate_linpred(exact_ranefs=True)
mu = np.exp(eta) / (1.0 + np.exp(eta))
y = rng.normal(loc=eta, scale=s)

tau = sp.stats.scoreatpercentile(eta, np.linspace(0, 100, 5)[1:-1])
thresholds = np.pad(tau, ((1, 1)), mode='constant', constant_values=[-np.inf, np.inf])

df['y_normal'] = y
df['y_binom1'] = rng.binomial(n=1, p=mu)
df['y_binom10'] = rng.binomial(n=10, p=mu) / 10
df['y_ordinal'] = pd.cut(eta, thresholds).cat.codes.astype(float)
plot_kws = dict(var_names=['$\\theta$'], coords={"$\\theta$_dim_0":[0, 1, 2]})


model = MixedMCMC("y_binom1~x1+x2+x3+x4+x5+(1+x5|id1)", df, response_dist='bernoulli')
model.sample(n_samples=32_000, burnin=2_000, n_chains=8)
print(model.summary)


az.plot_rank(model.az_data, figsize=(6, 9), **plot_kws)
az.plot_trace(model.az_data, figsize=(12, 8), compact=False, **plot_kws)


model2 = MixedMCMC("y_binom10~x1+x2+x3+x4+x5+(1+x5|id1)", df, response_dist="binomial", weights=np.ones_like(df['y_binom10'])*10.0)
model2.sample(n_samples=7_000, burnin=2_000, n_chains=8, sampling_kws=dict(n_adapt=6000, adaption_rate=1.02))
print(model2.summary)

az.plot_rank(model2.az_data, figsize=(6, 9), **plot_kws)
az.plot_trace(model2.az_data, figsize=(12, 8), compact=False, **plot_kws)



model3 = MixedMCMC("y_ordinal~x1+x2+x3+x4+x5+(1+x5|id1)", df, response_dist='ordinal_probit')
model3.sample(n_samples=12_000, burnin=2_000, n_chains=8, sampling_kws=dict(n_adapt=5_000, adaption_rate=1.02))
print(model3.summary)

az.plot_rank(model3.az_data, figsize=(6, 9), **plot_kws)
az.plot_trace(model3.az_data, figsize=(12, 8), compact=False, **plot_kws)



model4 = MixedMCMC("y_normal~x1+x2+x3+x4+x5+(1+x5|id1)", df, response_dist='normal')
model4.sample(n_samples=7_000, burnin=2000, n_chains=8)
print(model4.summary)


az.plot_rank(model4.az_data, figsize=(6, 9), **plot_kws)
az.plot_trace(model4.az_data, figsize=(12, 8), compact=False, **plot_kws)









