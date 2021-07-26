#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 18:24:21 2021

@author: lukepinkel
"""

import numpy as np
import pandas as pd
from pystats.pylmm.lmm import LMM
from pystats.pylmm.prf import plot_profile, profile
from pystats.pylmm.sim_lmm import MixedModelSim
from pystats.utilities.linalg_operations import invech

pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 250)
pd.set_option('display.max_columns', 50)
np.set_printoptions(suppress=True)
rng = np.random.default_rng(1234)


formula = "y~x1+x2+x3+x4+(1+x4|id1)"
model_dict = {}
n_grp, n_per = 200, 20
n_obs = n_grp * n_per
model_dict['gcov'] = {'id1':invech(np.array([4.0, -2.0, 4.0]))}
model_dict['ginfo'] = {'id1':dict(n_grp=n_grp, n_per=n_per)} 
model_dict['mu'] = np.zeros(4)
model_dict['vcov'] = np.eye(4)
model_dict['beta'] = np.array([0.0, 1.0, -1.0, 2.0, -2.0])
model_dict['n_obs'] = n_obs
group_dict={}

v = np.sum(model_dict['beta']**2) + np.sum([np.sum(np.diag(G)) for G in model_dict['gcov'].values()])
s = np.sqrt(v*(1-0.7)/0.7)
msim = MixedModelSim(formula, model_dict, group_dict=group_dict, rng=rng)
df = msim.df
df["y"] = msim.simulate_response(resid_scale=s, exact_ranefs=True, exact_resids=True)
model = LMM(formula, df)
model.fit(opt_kws=dict())


thetas, zetas, ix = profile(50, model, tb=4.5)
quantiles = np.array([60, 70, 80, 90, 95, 99, 99.9])
quantiles = np.concatenate([(100-quantiles[::-1])/2, 100-(100-quantiles)/2])
fig, axes = plot_profile(model, thetas, zetas, ix, quantiles=quantiles)

