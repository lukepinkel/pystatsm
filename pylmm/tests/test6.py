#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 03:43:15 2020

@author: lukepinkel
"""

import timeit # analysis:ignore
import tqdm# analysis:ignore
import numpy as np # analysis:ignore
import pandas as pd # analysis:ignore
import scipy as sp # analysis:ignore
from ..pylmm.lmm import LME # analysis:ignore
import scipy.sparse as sps # analysis:ignore
from .test_data import generate_data # analysis:ignore
from ..utilities.random_corr import vine_corr # analysis:ignore
from ..utilities.linalg_operations import invech, vech, scholesky # analysis:ignore

def resampled_indices(indices, resample_within=False):
    n = len(indices)
    i = np.random.choice(n, n, replace=True)
    indices_r = indices[i]
    if resample_within:
        for k, j in enumerate(indices_r):
            indices_r[k] = j[np.random.choice(len(j), len(j), replace=True)]
    return indices_r
    

formula = "y~x1+x2+(1|id1)"
model_dict = {}
model_dict['gcov'] = {'id1':invech(np.array([4.0]))}
model_dict['ginfo'] = {'id1':dict(n_grp=100, n_per=10)} 
model_dict['mu'] = np.zeros(2)
model_dict['vcov'] = vine_corr(2)
model_dict['beta'] = np.array([3.0, 2, -2])
model_dict['n_obs'] = 1_000

df, _ = generate_data(formula, model_dict, r=0.6**0.5)

model = LME(formula, df)
model._fit({'method': 'trust-constr',
          'options': {'gtol': 1e-16, 'xtol': 1e-16, 'verbose': 3}})

opt_kws = {'method': 'trust-constr',
          'options': {'gtol': 1e-6, 'xtol': 1e-9, 'verbose': 0}}

groups = df.groupby('id1').groups
groups = np.array([np.array(v, dtype=int) for k, v in groups.items()], dtype=object)

params = np.zeros((5000, 5))


for i in tqdm.tqdm(range(1000, 5000), mininterval=2, smoothing=0.01):
    model_i = LME(formula, df.iloc[np.concatenate(resampled_indices(groups))])
    opt, theta = model_i._optimize_theta(opt_kws)
    beta, XtWX_inv, _, _, _, _, _, _ = model_i._compute_effects(theta)
    params[i] = np.concatenate([beta, theta])

param_df = pd.DataFrame(params)
