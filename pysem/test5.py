#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 03:53:39 2020

@author: lukepinkel
"""
import warnings
warnings.simplefilter("ignore", UserWarning)
import tqdm
import seaborn as sns# analysis:ignore
import numpy as np # analysis:ignore
import scipy as sp
import scipy.stats
import pandas as pd # analysis:ignore
from pystats.pysem.sem import MLSEM # analysis:ignore

from pystats.utilities.random_corr import multi_rand, vine_corr # analysis:ignore

Lambda = np.array([[ 1.0,  0.0,  0.0],
                   [ 2.0,  0.0,  0.0],
                   [ 0.5,  0.0,  0.0],
                   [ 0.0,  1.0,  0.0],
                   [ 0.0, -1.0,  0.0],
                   [ 0.0,  1.0,  0.0],
                   [ 0.0,  0.0,  1.0],
                   [ 0.0,  0.0,  1.0],
                   [ 0.0,  0.0,  1.0]])

Beta = np.array([[0.0, 0.0, 0.0],
                 [1.0, 0.0, 0.0],
                 [0.2, 0.8, 0.0]])

Phi = np.eye(3)*2.0

TH = np.eye(9)*0.5
IB = np.linalg.inv(np.eye(3) - Beta)
Sigma = Lambda.dot(IB).dot(Phi).dot(IB.T).dot(Lambda.T)+TH

X = pd.DataFrame(multi_rand(Sigma), columns=['x%i'%i for i in range(1, 10)])
#Z = X + np.random.normal(0,  X.std(axis=0)*.01, size=(X.shape))
dist = sp.stats.multivariate_normal(np.zeros(9), Sigma)
Z = pd.DataFrame(dist.rvs(1000), columns=['x%i'%i for i in range(1, 10)])
L = pd.DataFrame(Lambda!=0, index=X.columns, columns=['z%i'%i for i in range(1, 4)])
B = pd.DataFrame(Beta!=0, index=L.columns, columns=L.columns)
P = Phi!=0
T = TH!=0
res = np.zeros((1000, 21, 2))
Z_samples = dist.rvs((1000, 1000))
pbar = tqdm.tqdm(total=1000, smoothing=0.0001)
for i in range(1000):
    Z = Z_samples[i]
    Z = Z - Z.mean(axis=0)
    Z = pd.DataFrame(Z, columns=['x%i'%i for i in range(1, 10)])
    model = MLSEM(Z, L, B, T)
    model.fit(xtol=1e-9, gtol=1e-6, verbose=0)
    res[i] = model.res.values[:, :2]
    pbar.update(1)
pbar.close()
    




coef_ests = pd.DataFrame(res[:, :, 0])
coef_stnd = pd.DataFrame(res[:, :, 1])
coef_ests.agg(['mean', 'std']).T
coef_stnd.agg(['mean', 'std']).T
mc_res = pd.concat([coef_ests.agg(['mean', 'std']).T, 
           coef_stnd.agg(['mean', 'std']).T], axis=1)



