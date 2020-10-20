#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 04:16:40 2020

@author: lukepinkel
"""

import arviz as az # analysis:ignore
import numpy as np # analysis:ignore
import scipy as sp # analysis:ignore
import scipy.stats # analysis:ignore
import pandas as pd # analysis:ignore
import seaborn as sns # analysis:ignore
from .tests.test_data_mcmc import SimulatedGLMM # analysis:ignore
from ..pylmm.glmm_mcmc import MixedMCMC # analysis:ignore


def to_arviz_dict(samples, var_dict, burnin=0):
    az_dict = {}
    for key, val in var_dict.items():
        az_dict[key] = samples[:, burnin:, val]
    return az_dict    
    
  
def simulate_fixed(fixed_value, n_chains, n_samples, model):
    samples = np.zeros((n_chains, n_samples, model.n_params))
    model.t_init[-1] = fixed_value
    for i in range(n_chains):
        samples[i] = model.sample_slice_gibbs5(n_samples)
    
    samples = samples[:, :, :-1]
    az_dict = to_arviz_dict(samples, {"$\\beta$":np.arange(4), 
                                      "$\\theta$":np.arange(4, 7)}, burnin=1000)
    
    az_data = az.from_dict(az_dict)
    summary = az.summary(az_data)
    summary['sampling_effeciency'] = summary['ess_mean'] / np.product(samples.shape[:2])
    return samples, summary

        
gen_model = SimulatedGLMM(n_grp=400, n_per=30, r=0.317)
model = MixedMCMC(gen_model.formula, gen_model.df)

model.priors['R'] = dict(V=model.n_ob*1.0, n=model.n_ob) #160-200 draws per second

n_samples = 15_000
n_chains = 8


samples = np.zeros((n_chains, n_samples, model.n_params))
samples_u = np.zeros((n_chains, n_samples, model.n_re))
#samples_p = np.zeros((n_chains, n_samples, model.n_ob))

model.t_init[-1] = 10
for i in range(n_chains):
    samples[i], scnd = model.sample_slice_gibbs(n_samples, save_u=True)
    #samples_p[i] = scnd['pred']
    samples_u[i] = scnd['u']
    
samples = samples[:, :, :-1]
az_dict = to_arviz_dict(samples, {"$\\beta$":np.arange(4), 
                                  "$\\theta$":np.arange(4, 7)}, burnin=1000)

az_data = az.from_dict(az_dict)
summary = az.summary(az_data, hdi_prob=0.95)
summary['sampling_effeciency'] = summary['ess_mean'] / np.product(samples.shape[:2])
summary['mean_skew'] = sp.stats.skew(samples, axis=1).mean(axis=0)
summary['sd_skew'] = sp.stats.skew(samples, axis=1).std(axis=0)
summary.insert(2, 't', summary['mean']/summary['sd'])

print(summary)
az.plot_trace(az_data, var_names=['$\\theta$'])


