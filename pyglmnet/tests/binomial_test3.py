#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 10:17:19 2020

@author: lukepinkel
"""

import numpy as np
import scipy as sp
from pystats.pyglmnet.binomial_eln import cv_glmnet, inv_logit, binom_glm_cd
from pystats.utilities.random_corr import vine_corr, multi_rand # analysis:ignore

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
Xsq = X**2
acs = np.ones(p, dtype=bool)
ix = np.arange(p)
ffc = 1.0
betas, f_path, lambdas, n_its, bfits = cv_glmnet(7, X, y, alpha, lambdas=50, 
                                                 btol=1e-5, dtol=1e-5, 
                                                 n_iters=5_000, refit=True, 
                                                 warm_start=True, lmin_pct=0,
                                                 nr_ent=True, ffc=10.0,
                                                 seq_rule=True)
b = bfits[-2]
lam_ = lambdas[-1]
pmin = 1e-9
la, dla = alpha * lam_, (1.0 - alpha) * lam_


eta = X.dot(b)
mu = inv_logit(eta)
mlb, mub = (mu<=pmin), (mu>= (1.0 - pmin))
mu[mlb] = 0.0
mu[mub] = 1.0
muconj = 1.0 - mu
w = mu * muconj
w[mlb] = pmin
w[mub] = pmin
yr = y - mu
z = yr / w
b_new, acs_new = binom_glm_cd(b.copy(), X, Xsq, z, w, la, dla, 
                              acs.copy(), ix, n, ffc=100)

xv = (w[:, None] * Xsq).sum(axis=0)
print((xv * (b_new - b)**2).max())


b = b_new










