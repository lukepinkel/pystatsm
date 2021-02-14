# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 13:51:48 2020

@author: lukepinkel
"""

import arviz as az
import numpy as np
import scipy as sp
import scipy.stats
import pandas as pd
import seaborn as sns
from pystats.pylmm.glmm import GLMM_AGQ
from pystats.pylmm.lmm import GLMM, Binomial
from pystats.pylmm.glmm_mcmc import MixedMCMC
from pystats.pylmm.test_data2 import generate_data
from pystats.utilities.random_corr import vine_corr
from pystats.utilities.linalg_operations import invech

formula = "y~x1+x2+(1|id1)"
model_dict = {}
n_grp = 50
n_per = 10
model_dict['gcov'] = {'id1':invech(np.array([2.0]))}
model_dict['ginfo'] = {'id1':dict(n_grp=n_grp, n_per=n_per)} 
model_dict['mu'] = np.zeros(2)
model_dict['vcov'] = vine_corr(2)
model_dict['beta'] = np.array([-0.2, 1.0, -1.0])
model_dict['n_obs'] = n_grp * n_per


df, formula, u, eta = generate_data(formula, model_dict, r=0.6**0.5)
mu = np.exp(eta) / (1.0 + np.exp(eta))
df['y'] = np.random.binomial(n=1, p=mu)

model1 = GLMM(formula, df, fam=Binomial())
model1.fit()

model2 = GLMM_AGQ(formula, df, family=Binomial())
model2.fit(nagq=200)

model3 = MixedMCMC(formula, df)
model3.priors['id1'] = dict(V=np.ones((1, 1)), n=1)
samples, az_data, summary, samples_a = model3.fit(n_samples=40_000, burnin=10_000, 
                                       method='Slice-Gibbs')

az.plot_rank(az_data, var_names=['$\\theta$'], coords={"$\\theta$_dim_0":[0]},
             figsize=(6, 9))

az.plot_trace(az_data, var_names=['$\\theta$'], coords={"$\\theta$_dim_0":[0]},
             figsize=(12, 8))

formula = "y~x1+x2+(1|id1)"
model_dict = {}
n_grp = 200
n_per = 10
model_dict['gcov'] = {'id1':invech(np.array([2.0]))}
model_dict['ginfo'] = {'id1':dict(n_grp=n_grp, n_per=n_per)} 
model_dict['mu'] = np.zeros(2)
model_dict['vcov'] = vine_corr(2)
model_dict['beta'] = np.array([-0.2, 1.0, -1.0])
model_dict['n_obs'] = n_grp * n_per


df, formula, u, eta = generate_data(formula, model_dict, r=0.6**0.5)
mu = np.exp(df['y']) / (1.0 + np.exp(df['y']))
df['y'] = np.random.binomial(n=10, p=mu) / 10.0

df2 = df.copy()
model3b = GLMM_AGQ(formula, df2, family=Binomial(weights=np.ones(2000)*10.0))
model3b.fit(nagq=200)





model3 = MixedMCMC(formula, df, weights=np.ones_like(df['y'])*10.0)
model3.priors['id1'] = dict(V=np.ones((1, 1)), n=1)
model3.priors['R'] = dict(V=1.0, n=1.0)
samples, az_data, summary, samples_a = model3.fit(n_samples=25_000, burnin=5_000, 
                                       n_chains=4, method='MH-Gibbs',
                                       sample_kws=dict(freeR=False, save_u=True))

samples2 = samples[:, :, :-1]
mu_c = np.mean(samples2, axis=1)
mu_t = np.mean(mu_c, axis=0)
nc, ns = samples2.shape[:-1]
Bv = ns / (nc - 1)

stat_funcs = {'min':np.min, 
              'max':np.max, 
              'skew':sp.stats.skew, 
              'kurt':sp.stats.kurtosis,
              'size': np.size,
              'pct_0.5%':lambda x: sp.stats.scoreatpercentile(x, 0.5),
              'pct_25%': lambda x: sp.stats.scoreatpercentile(x, 20),
              'pct_50%': lambda x: sp.stats.scoreatpercentile(x, 50),
              'pct_75%': lambda x: sp.stats.scoreatpercentile(x, 80),
              'pct_99.5%': lambda x: sp.stats.scoreatpercentile(x, 99.5)}

summary = az.summary(az_data, credible_interval=0.95, extend=True, round_to='none',
                     stat_funcs=stat_funcs)

summary['effeciency'] = summary['ess_mean'] / summary['size']

summary = summary.iloc[:-1]

u_samples = np.concatenate([samples_a[i]['u'][np.newaxis] for i in range(len(samples_a))])
u_stacked = u_samples.reshape(-1, u_samples.shape[-1])

u_summ = pd.DataFrame(np.vstack((np.mean(u_stacked, axis=0), np.std(u_stacked, axis=0))).T,
                      columns=['mean', 'std'])

sns.regplot(u_summ['mean'], u)

u_summ.insert(0, 'genval', u)

az.plot_rank(az_data, var_names=['$\\theta$'], coords={"$\\theta$_dim_0":[0]},
             figsize=(6, 9))

az.plot_trace(az_data, var_names=['$\\theta$'], coords={"$\\theta$_dim_0":[0]},
             figsize=(16, 8), trace_kwargs=dict(linewidth=0.1, alpha=0.2))






