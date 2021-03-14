# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 22:47:25 2020

@author: lukepinkel
"""

import numpy as np
import scipy as sp
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pystats.pyglmnet.eln_utils import plot_elnet_cv
from pystats.pyglmnet.binomial_eln import cv_binom_glmnet, inv_logit
from pystats.utilities.random_corr import multi_rand 

rng = np.random.default_rng(123)
n, p, q = 1000, 200, 20
S = 0.9**sp.linalg.toeplitz(np.arange(p))
X = multi_rand(S, np.max((n, p+1)), rng=rng)[:n]
X/= np.sqrt(np.sum(X**2, axis=0)) / np.sqrt(X.shape[0])
beta = np.zeros(p)

bvals = np.tile([-1, -0.5, 0.5, 1.0], q//4)
beta[np.arange(0, p, p//q)] = bvals 
lpred = X.dot(beta)
rsq = 0.99
eta_gen = sp.stats.norm(lpred, np.sqrt((1-rsq)/rsq * lpred.var())).rvs(random_state=rng)
mu = inv_logit(eta_gen)
y = sp.stats.binom(n=1, p=mu).rvs(random_state=rng).astype(float)
alpha = 0.99

betas, f_path, lambdas, n_its, bfits = cv_binom_glmnet(10, X, y, alpha, lambdas=400, 
                                                 btol=1e-8, dtol=1e-8, 
                                                 n_iters=5_000, refit=True, 
                                                 warm_start=True, lmin_pct=0,
                                                 nr_ent=True,  seq_rule=True,
                                                 rng=rng)
dev = pd.DataFrame(f_path[:, :, 0])
lam_ = lambdas[dev.mean(axis=1).idxmin()]
fig, ax = plot_elnet_cv(f_path, lambdas, bfits)

fig, ax = plt.subplots()
sns.heatmap(np.abs(bfits), cmap=plt.cm.Greys, vmin=0, ax=ax)

dev2 = dev.copy().mean(axis=1)
dev2.index = lambdas
dev2 = pd.DataFrame(dev2)
dev2['nonzero'] = (bfits!=0).sum(axis=1)
