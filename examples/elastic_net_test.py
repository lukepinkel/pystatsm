# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 21:34:13 2020

@author: lukepinkel
"""
import numpy as np 
import scipy as sp
import scipy.stats
import pandas as pd
from pystats.pyglmnet.gaussian_eln import cv_glmnet
from pystats.pyglmnet.eln_utils import plot_elnet_cv
from pystats.utilities.random_corr import multi_rand

n, p, q = 2000, 500, 20

S = 0.95**sp.linalg.toeplitz(np.arange(p))
X = multi_rand(S, np.max((n, p+1)))[:n]
X/= np.sqrt(np.sum(X**2, axis=0)) / np.sqrt(X.shape[0])
beta = np.zeros(p)

bvals = np.tile([-1, -0.5, 0.5, 1.0], q//4)
beta[np.arange(0, p, p//q)] = bvals 
lpred = X.dot(beta)
rsq = 0.5
y = sp.stats.norm(lpred, np.sqrt((1-rsq)/rsq * lpred.var())).rvs()
alpha = 0.99


beta_path, f_path, lambdas, bfits, n_its = cv_glmnet(10, X, y, alpha, lambdas=500, 
                                                     tol=1e-6, n_iters=2000)
dev = pd.DataFrame(f_path[:, :, 0])
nnz = (bfits!=0).sum(axis=1)
lam_ = lambdas[dev.mean(axis=1).idxmin()]
fig, ax = plot_elnet_cv(f_path, lambdas, bfits)
axt = ax[0].twinx()
axt.plot(np.log(lambdas), nnz)
ax[0].set_yscale('log')

beta_diff = np.sum(((np.diff(bfits, axis=0))**2), axis=1)