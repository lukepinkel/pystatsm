# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 11:45:33 2020

@author: lukepinkel
"""
import tqdm
import numpy as np 
import scipy as sp
import scipy.stats
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from pystats.pylvcorr import lvcorr

from pystats.utilities.random_corr import multi_rand


n_vals, n_sims = 20, 1000

R = np.eye(2)
rho_vals = np.linspace(-0.8, 0.8, n_vals)

polychor_sim_res = np.zeros((n_vals*n_sims, 3))

pbar = tqdm.tqdm(total=n_vals*n_sims)
count = 0
for rho in rho_vals:
    for i in range(n_sims):
        R[0, 1] = R[1, 0] = rho
        X = pd.DataFrame(sp.stats.multivariate_normal(np.zeros(2), R).rvs(1000))
        X = X - X.mean(axis=0)
        X[0] = pd.qcut(X[0], 3).cat.codes.astype(float)
        X[1] = pd.qcut(X[1], 5).cat.codes.astype(float)
        model_polychor = lvcorr.Polychoric(X[0], X[1])
        model_polychor.fit()
        polychor_sim_res[count] = rho, model_polychor.rho_hat, model_polychor.se_rho
        count += 1
        pbar.update(1)
pbar.close()

polychor_sim_res_df = pd.DataFrame(polychor_sim_res, columns=['rho', '$\\hat{\\rho}$', '$SE(\\hat{\\rho})$'])
grouped_df = polychor_sim_res_df.groupby("rho").agg(['mean', 'std'])
g = sns.boxplot(x='rho', y='$\\hat{\\rho}$', data=polychor_sim_res_df)

with sns.axes_style('darkgrid'):
    fig, ax = plt.subplots()
    ax.errorbar(grouped_df.index, y=grouped_df['$\\hat{\\rho}$', 'mean'],
                yerr=grouped_df[('$\\hat{\\rho}$', 'std')]*2.0,
                fmt='none', capsize=6, elinewidth=2, capthick=2,
                zorder=100)
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    ax.set_aspect('equal')
    ax.axvline(0, color='k', zorder=1)
    ax.axhline(0, color='k', zorder=1)
    ax.plot(np.linspace(-0.9, 0.9, 100), np.linspace(-0.9, 0.9, 100),
            linestyle='--', color='k', alpha=0.8)
    ax.set_xlim(-0.85, 0.85)
    ax.set_ylim(-0.85, 0.85)


polychor_sim_res_df['d'] = polychor_sim_res_df['rho'] - polychor_sim_res_df['$\\hat{\\rho}$']

R = np.array([[1.0, 0.5],
              [0.5, 1.0]])
X = pd.DataFrame(multi_rand(R))
X[1] = pd.qcut(X[1], 3).cat.codes.astype(float)

model_polyserial = lvcorr.Polyserial(X[0], X[1])
model_polyserial.fit()
