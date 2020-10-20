#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 00:12:24 2020

@author: lukepinkel
"""



import numpy as np # analysis:ignore
import scipy as sp # analysis:ignore
import pandas as pd # analysis:ignore
import seaborn as sns # analysis:ignore
import matplotlib.pyplot as plt # analysis:ignore
from ...utilities.random_corr import vine_corr, multi_rand # analysis:ignore
from ..eln_utils import plot_elnet_cv, process_cv # analysis:ignore
from ..gaussian_eln import cv_glmnet, elnet, elnet_grad # analysis:ignore


n, p, q = 1000, 500, 50
S = vine_corr(p, eta=1.0)
#S = np.eye(p)
X = multi_rand(S, n)
X/= np.sqrt(np.sum(X**2, axis=0))
X*=np.sqrt(X.shape[0])
beta = np.zeros(p)
bvals = np.random.choice([-1, -0.5, 0.5, 1.0], q, replace=True)
beta[np.random.choice(p, q, replace=False)] = bvals 
lpred = X.dot(beta)
rsq = 0.5

y = sp.stats.norm(lpred, np.sqrt((1-rsq)/rsq * lpred.var())).rvs()


alpha = 0.99
beta_path, f_path, lambdas = cv_glmnet(7, X, y, alpha=alpha)

mse_sumstats, lambda_min_mse = process_cv(f_path[:, :, 0], lambdas)
plot_elnet_cv(f_path, lambdas)

lambda_ = lambda_min_mse
beta_hat, fvals, active = elnet(X, y, lambda_, alpha,  n_iters=5000, tol=1e-12)

comp = pd.DataFrame(np.vstack((beta, beta_hat)).T)

print(elnet_grad(beta_hat, X, y, lambda_, alpha)[active], lambda_ * 2 * alpha)
