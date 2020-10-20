#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 04:46:09 2020

@author: lukepinkel
"""

import numpy as np
import scipy as sp
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ..eln_utils import plot_elnet_cv
from ..binomial_eln import cv_binom_glmnet, binom_glmnet, inv_logit
from ...utilities.random_corr import vine_corr, multi_rand # analysis:ignore

n, p, q = 1000, 200, 20
S = 0.9**sp.linalg.toeplitz(np.arange(p))
X = multi_rand(S, np.max((n, p+1)))[:n]
X/= np.sqrt(np.sum(X**2, axis=0)) / np.sqrt(X.shape[0])
beta = np.zeros(p)
#bvals = np.random.choice([-1, -0.5, 0.5, 1.0], q, replace=True)
bvals = np.tile([-1, -0.5, 0.5, 1.0], q//4)
beta[np.arange(0, p, p//q)] = bvals 
lpred = X.dot(beta)
rsq = 0.99
eta_gen = sp.stats.norm(lpred, np.sqrt((1-rsq)/rsq * lpred.var())).rvs()
mu = inv_logit(eta_gen)
y = sp.stats.binom(n=1, p=mu).rvs().astype(float)
alpha = 0.99

betas, f_path, lambdas, n_its, bfits = cv_binom_glmnet(10, X, y, alpha, lambdas=50, 
                                                 btol=1e-7, dtol=1e-7, 
                                                 n_iters=5_000, refit=True, 
                                                 warm_start=True, lmin_pct=0,
                                                 nr_ent=True, ffc=11.0,
                                                 seq_rule=True)
dev = pd.DataFrame(f_path[:, :, 0])
lam_ = lambdas[dev.mean(axis=1).idxmin()]
fig, ax = plot_elnet_cv(f_path, lambdas)
#fig.savefig("/users/lukepinkel/Downloads/ELN_BinomialGLM.png", dpi=600)
fig, ax =plt.subplots()
sns.heatmap(np.abs(bfits), cmap=plt.cm.Greys, vmin=0, ax=ax)
mng = plt.get_current_fig_manager()
mng.window.showMaximized()

bstart=bfits[dev.mean(axis=1).idxmin()-1]
beta_hat, _, fvals = binom_glmnet(X, y, lam_, alpha,  btol=1e-6, 
                                  dtol=1e-6, n_iters=10_000, 
                                  ffc=150.0, b=bstart, active=bstart!=0)

comp = pd.DataFrame(np.vstack((beta_hat, beta)).T)

fig, ax = plt.subplots()
ax.plot(bfits, color='blue', linewidth=0.8, alpha=0.8)
eta_hat = X.dot(beta_hat)
sns.jointplot(eta_hat, eta_gen, stat_func=sp.stats.pearsonr)

nnz = (betas!=0).sum(axis=2)




