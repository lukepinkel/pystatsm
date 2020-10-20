#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 23:51:46 2020

@author: lukepinkel
"""

import numpy as np # analysis:ignore
import scipy as sp # analysis:ignore
import pandas as pd # analysis:ignore
import seaborn as sns # analysis:ignore
import matplotlib.pyplot as plt # analysis:ignore
from ..gaussian_eln import cv_glmnet # analysis:ignore
from ..eln_utils import plot_elnet_cv # analysis:ignore
from ...utilities.random_corr import vine_corr, multi_rand # analysis:ignore




n, p, q = 1000, 500, 20
S = 0.5**sp.linalg.toeplitz(np.arange(p))
X = multi_rand(S, np.max((n, p+1)))[:n]
X/= np.sqrt(np.sum(X**2, axis=0)) / np.sqrt(X.shape[0])
beta = np.zeros(p)
#bvals = np.random.choice([-1, -0.5, 0.5, 1.0], q, replace=True)
bvals = np.tile([-1, -0.5, 0.5, 1.0], q//4)
beta[np.arange(0, p, p//q)] = bvals 
lpred = X.dot(beta)
rsq = 0.5
y = sp.stats.norm(lpred, np.sqrt((1-rsq)/rsq * lpred.var())).rvs()
alpha = 0.99

beta_path, f_path, lambdas = cv_glmnet(10, X, y, alpha, lambdas=None)
dev = pd.DataFrame(f_path[:, :, 0])
lam_ = lambdas[dev.mean(axis=1).idxmin()]
fig, ax = plot_elnet_cv(f_path, lambdas)