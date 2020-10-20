#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 06:23:49 2020

@author: lukepinkel
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ..eln_utils import plot_elnet_cv
from ..binomial_eln import cv_binom_glmnet, binom_glmnet

alpha = 0.99
df = pd.read_csv("/users/lukepinkel/pystats/pyglmnet/tests/glmnet_test.csv", index_col=0)
X = df.iloc[:, :200].values
y = df.iloc[:, 200].values

betas, f_path, lambdas, n_its, bfits = cv_binom_glmnet(10, X, y, alpha, lambdas=140, 
                                                 btol=1e-5,  dtol=1e-5, 
                                                 n_iters=2000, 
                                                 refit=True,  warm_start=True,
                                                 lmin_pct=10.0, ffc=2.0)
dev = pd.DataFrame(f_path[:, :, 0])
lam_ = lambdas[dev.mean(axis=1).idxmin()]
plot_elnet_cv(f_path, lambdas)
fig, ax =plt.subplots()
sns.heatmap(np.abs(bfits), cmap=plt.cm.Greys, vmin=0, ax=ax)

lam_ = 0.008932892
beta_hat, _, fvals = binom_glmnet(X, y, lam_, alpha, btol=1e-9, dtol=1e-9,
                                  n_iters=10_000, pmin=1e-9,
                                  ffc=1.2)

comp = pd.read_csv("/users/lukepinkel/pystats/pyglmnet/tests/glmnet_coefs.csv", index_col=0)
comp.columns = ['b1']
comp['b2'] = beta_hat

np.allclose(comp['b1'], comp['b2'], atol=4e-4, rtol=4e-4)

