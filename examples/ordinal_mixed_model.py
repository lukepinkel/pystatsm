# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 20:41:18 2021

@author: lukepinkel
"""

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


rsq = 0.8
formula = "y~1+x1+x2+(1+x2|id1)"
model_dict = {}
n_grp, n_per = 200, 20
n_obs = n_grp * n_per
model_dict['gcov'] = {'id1':invech(np.array([1.0, -0.5, 1.0]))}
model_dict['ginfo'] = {'id1':dict(n_grp=n_grp, n_per=n_per)} 
model_dict['mu'] = np.zeros(2)
model_dict['vcov'] = np.eye(2)
model_dict['beta'] = np.array([0.0, -1.0, 1.0])
model_dict['n_obs'] = n_obs

msim = MixedModelSim(formula, model_dict, rng=rng)
df = msim.df
u = msim.simulate_ranefs(exact_ranefs=True)
eta = msim.eta_fe + msim.Z.dot(u)
s = np.sqrt((1.0 - rsq) / rsq * eta.var())
mu = rng.normal(eta, scale=s)


q = np.linspace(0, 100, 8, endpoint=False)[1:]
tau_star = sp.stats.scoreatpercentile(mu, q)
thresh = np.pad(tau_star, ((1, 1)), mode='constant', constant_values=[-np.inf, np.inf])
df['y'] = pd.cut(mu, thresh).codes.astype(float)

plot_kws = dict(var_names=['$\\theta$'], coords={"$\\theta$_dim_0":[0]})
sampling_kws = dict(damping=0.99, adaption_rate=1.025, n_adapt=6000, save_u=True)

model = MixedMCMC("y~1+x1+x2+(1+x2|id1)", df, response_dist='ordinal_probit', freeR=False)
model.priors["R"] = dict(V=1, n=1)
model.sample(n_samples=22_000, burnin=6_000, n_chains=8, sampling_kws=sampling_kws)
print(model.summary)

u_hat = np.concatenate([model.samples_a[i]['u'][:, np.newaxis] for i in  model.samples_a.keys()], axis=1)
u_hat = np.mean(u_hat, axis=(0, 1))
linpred = model.X.dot(model.beta) + model.Z.dot(u_hat)
np.vstack((np.pad(model.tau, (1, 0)), (tau_star-tau_star.min())/s)).T

comp = pd.DataFrame(np.vstack((df["y"], mu, linpred)).T,  columns=['y', "mu", "linpred"])


tau_hat = np.pad(np.pad(model.tau, (1, 0)), (1, 1), constant_values=[-np.inf, np.inf])
comp["yhat"] = pd.cut(comp["linpred"], tau_hat).cat.codes*1.0
xtab = pd.crosstab(comp["y"], comp["yhat"])








