# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 05:55:40 2021

@author: lukepinkel
"""

import scipy as sp
import scipy.stats
import numpy as np
import pandas as pd
from pystats.pylmm.mlmm import MLMM, construct_model_matrices, vec, inverse_transform_theta, invech
from pystats.utilities.random_corr import vine_corr, exact_rmvnorm

rng = np.random.default_rng(123)
n_groups = 400
n_per = 20
n_obs = n_groups * n_per

S = vine_corr(5, seed=123)
X = exact_rmvnorm(S, n_obs, seed=123)
data = pd.DataFrame(X, columns=['x1', 'x2', 'x3', 'x4', 'x5'])
data['y'] = 0
data['id1'] = np.repeat(np.arange(n_groups), n_per)

formula = "y~x1+x2+x3+x4+(1+x5|id1)"
X, Z, y, dims, levels, _, _ = construct_model_matrices(formula, data=data)
B = rng.normal(0, 4, size=(5, 2))

A = np.eye(n_groups)
Sigma_r = np.array([[1.0, 0.5], [0.5, 1.0]])
Sigma_u = np.kron(Sigma_r, np.array([[2.0, 1.0], [1.0, 2.0]]))
Sigma_e = np.array([[2.0, 0.0], [0.0, 1.0]])
U1 = sp.stats.matrix_normal(mean=np.zeros((A.shape[0], Sigma_u.shape[0])),
                           rowcov=A, colcov=Sigma_u).rvs(1, random_state=rng)
U = np.concatenate([U1[:, :2].reshape(-1, 1, order='C'),
                    U1[:, 2:].reshape(-1, 1, order='C')], axis=1)

E = sp.stats.multivariate_normal(mean=np.zeros(2), cov=Sigma_e).rvs(n_obs,random_state=rng)
Y = X.dot(B) + Z.dot(U) + E
y = vec(Y.T)
data[['y1', 'y2']] = Y

formula = "(y1, y2)~x1+x2+x3+x4+(1+x5|id1)"


model = MLMM(formula, data)
model.fit(opt_kws=dict(verbose=3))






    