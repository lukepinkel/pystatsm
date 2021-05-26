# -*- coding: utf-8 -*-
"""
Created on Wed May 26 08:05:52 2021

@author: lukepinkel
"""

import tqdm
import numpy as np
import pandas as pd
from pystats.pylmm.lmm import LMM
from pystats.pylmm.glmm_mcmc import MixedMCMC
from pystats.pylmm.sim_lmm import MixedModelSim
from pystats.utilities.linalg_operations import invech, vech


pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 250)
pd.set_option('display.max_columns', 50)
np.set_printoptions(suppress=True)
rng = np.random.default_rng(1234)

formula = "y~x1+x2+x3+x4+(1+x5|id1)"
model_dict = {}
n_grp, n_per = 140, 5
n_obs = n_grp * n_per
model_dict['gcov'] = {'id1':invech(np.array([4.0, -2.0, 4.0]))}
model_dict['ginfo'] = {'id1':dict(n_grp=n_grp, n_per=n_per)} 
model_dict['mu'] = np.zeros(5)
model_dict['vcov'] = np.eye(5)
model_dict['beta'] = np.array([0.0, 1.0, -1.0, 2.0, -2.0])
model_dict['n_obs'] = n_obs

s = np.sqrt(18*(1-0.7)/0.7)
msim = MixedModelSim(formula, model_dict, rng=rng)
df = msim.df
df["y"] = msim.simulate_response(resid_scale=s, exact_ranefs=True, exact_resids=True)
model = LMM(formula, df)
model.fit()
theta_chol = np.zeros(4)
theta_chol[:3] = vech(np.linalg.cholesky(model_dict['gcov']['id1']))
theta_chol[3:] = s**2

n_samples = 5000
param_samples = np.zeros((n_samples, model.res.shape[0]))
se_samples = np.zeros((n_samples, model.res.shape[0]))

pbar = tqdm.tqdm(total=n_samples, smoothing=0.001)
for i in range(n_samples):
    model = msim.update_model(model, resid_scale=s, exact_ranefs=False, exact_resids=False)
    param_samples[i], se_samples[i] = msim.sim_fit(model, theta_chol.copy())
    pbar.update(1)
pbar.close()
param_samples = pd.DataFrame(param_samples, columns=model.res.index)
se_samples = pd.DataFrame(se_samples, columns=model.res.index)
comp = param_samples.agg(["mean", "std"]).T
comp["se_mean"] = se_samples.mean()

model2 = MixedMCMC(formula, df, response_dist="normal", rng=rng)
model2.sample(n_samples=15000, burnin=2500, n_chains=8)
comp["se_mcmc"] = model2.res["sd"]






