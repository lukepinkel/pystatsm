#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 21:37:56 2020

@author: lukepinkel
"""
import arviz as az # analysis:ignore
import numpy as np # analysis:ignore
import scipy as sp # analysis:ignore
import scipy.stats # analysis:ignore
import pandas as pd # analysis:ignore
from .test_data_mcmc import SimulatedGLMM # analysis:ignore
from ..pylmm.glmm_mcmc import MixedMCMC # analysis:ignore


def to_arviz_dict(samples, var_dict, burnin=0):
    az_dict = {}
    for key, val in var_dict.items():
        az_dict[key] = samples[:, burnin:, val]
    return az_dict
    

gen_model = SimulatedGLMM(n_grp=250, n_per=10)
model = MixedMCMC(gen_model.formula, gen_model.df)

model.priors['R'] = dict(V=model.n_ob*1.0, n=model.n_ob) #160-200 draws per second
n_samples = 15_000
n_chains = 8
samples = np.zeros((n_chains, n_samples, model.n_params))
for i in range(n_chains):
    samples[i] = model.sample_slice_gibbs3(n_samples)
    
az_dict = to_arviz_dict(samples, {"$\\beta$":np.arange(4), 
                                  "$\\theta$":np.arange(4, 8)}, burnin=1000)

az_data = az.from_dict(az_dict)
summary = az.summary(az_data)

az.plot_trace(az_data, var_names=['$\\theta$'])
az.plot_trace(az_data, var_names=['$\\theta$'], combined=True)

az.plot_trace(az_data, var_names=['$\\beta$'])
az.plot_trace(az_data, var_names=['$\\beta$'], combined=True)


theta_means = getattr(az_data.posterior, '$\\theta$').values.mean(axis=1)
theta_vars = getattr(az_data.posterior, '$\\theta$').values.var(axis=1)


