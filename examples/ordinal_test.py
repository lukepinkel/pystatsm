# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 19:39:04 2020

@author: lukepinkel
"""
import arviz as az
import numpy as np
import scipy as sp
import scipy.stats
import pandas as pd
import matplotlib.pyplot as plt
from pystats.utilities.random_corr import vine_corr
from pystats.utilities.linalg_operations import invech
from pystats.pylmm.ordinal import OrdinalMCMC
from pystats.pylmm.test_data2 import generate_data



n_grp, n_per = 150, 10
formula = "y~x1+x2+x3+(1+x4|id1)"
model_dict = {}
model_dict['gcov'] = {'id1':invech(np.array([1.0, -0.3, 1.0]))}
model_dict['ginfo'] = {'id1':dict(n_grp=n_grp, n_per=n_per)} 
model_dict['mu'] = np.zeros(4)
model_dict['vcov'] = vine_corr(4, 100)/10
model_dict['beta'] = np.array([0.2, -0.2, 0.4, -0.4])
model_dict['n_obs'] = n_grp*n_per
df, formula, u, linpred = generate_data(formula, model_dict, r=0.6**0.5)
u = np.random.normal(0, np.sqrt(linpred.var()), size=model_dict['n_obs'])
u = (u - u.mean()) / u.std()
eta = linpred + u
tau = sp.stats.scoreatpercentile(eta, np.linspace(0, 100, 8)[1:-1])
thresholds = np.pad(tau, ((1, 1)), mode='constant', constant_values=[-np.inf, np.inf])
df['y'] = pd.cut(eta, thresholds).cat.codes.astype(float)

model = OrdinalMCMC(formula, df)
sample_kws=dict(propC=0.04, store_z=True, target_accept=0.44,  n_adapt=30_000, damping=0.99, adaption_rate=1.01)
samples, az_data, summary, accept, z_samples = model.fit(60_000, 8, 5_000, sampler_kws=sample_kws)
print(summary)
stat_funcs={"kurtosis":sp.stats.kurtosis, "skew":sp.stats.skew}
summary2 = az.summary(az_data, credible_interval=0.95, extend=True,  stat_funcs=stat_funcs, round_to='none')


z_samples_combined = z_samples.reshape(-1, 1500)
z_means = z_samples_combined.mean(axis=0)
z_sds = z_samples_combined.std(axis=0)
zlb = sp.stats.scoreatpercentile(z_samples_combined, 5, axis=0)
zub = sp.stats.scoreatpercentile(z_samples_combined, 95, axis=0)

#%matplotlib inline
az.plot_trace(az_data, var_names=["$\\tau$"], figsize=(12, 8))
az.plot_trace(az_data, var_names=["$\\theta$"], figsize=(12, 8))
az.plot_rank(az_data, var_names=["$\\tau$"], coords={"$\\tau$_dim_0":[0, 1, 2]}, figsize=(12, 8))
az.plot_rank(az_data, var_names=["$\\tau$"], coords={"$\\tau$_dim_0":[3, 4]}, figsize=(12, 8))

az.plot_rank(az_data, var_names=["$\\theta$"], figsize=(12, 8))


fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(eta, z_means, s=0.5, alpha=0.5)
ax.scatter(eta, zlb, s=0.5, alpha=0.5)
ax.scatter(eta, zub, s=0.5, alpha=0.5)

tau_samples =  (getattr(az_data.posterior, '$\\tau$').to_dataset()).to_array().values[0]

distances = np.zeros((8, 8, 5))
distances_null = np.zeros((8, 8, 5))
for k in range(5):
    for i in range(8):
        for j in range(i):
            distances[i, j, k] = distances[j, i, k] = sp.stats.energy_distance(tau_samples[i, :, k], tau_samples[j, :, k])
            distances_null[i, j, k] = distances_null[j, i, k] = sp.stats.energy_distance(tau_samples[i, :, k]+0.1, tau_samples[j, :, k])


#%matplotlib qt5
fig, ax = plt.subplots(figsize=(12, 8))
ax.errorbar(eta, z_means, yerr=np.vstack(ax.scatter(eta, z_means, s=0.5, alpha=0.5)
(z_means-zlb, zub-z_means)), fmt='none', elinewidth=0.5, alpha=0.5)




fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(z_means, eta)




