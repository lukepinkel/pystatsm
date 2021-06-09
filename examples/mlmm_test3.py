# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 16:12:09 2021

@author: lukepinkel
"""

import scipy as sp
import scipy.stats
import numpy as np
import pandas as pd
from pystats.pylmm.mlmm import MLMM, construct_model_matrices, vec
from pystats.utilities.random_corr import vine_corr, exact_rmvnorm

rng = np.random.default_rng(123)
ngr1, npr1 = 400, 20
ngr2, npr2 = 800, 10
n_obs = ngr1 * npr1

S = vine_corr(5, seed=123)
X = exact_rmvnorm(S, n_obs, seed=123)
data = pd.DataFrame(X, columns=['x1', 'x2', 'x3', 'x4', 'x5'])
data['y'] = 0
data['id1'] = np.repeat(np.arange(ngr1), npr1)
data['id2'] = np.repeat(np.arange(ngr2), npr2)

formula = "y~x1+x2+x3+x4+(1+x5|id1)+(1|id2)"
X, Z, y, dims, levels, _, _ = construct_model_matrices(formula, data=data)

B = rng.choice([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5], size=10).reshape((5, 2))

A = np.eye(ngr1)

G1 = np.kron(np.array([[ 1.0,  0.5], [ 0.5,  1.0]]), np.array([[ 2.0,  1.0], [ 1.0,  2.0]]))
G2 = np.kron(np.array([[ 2.0, -1.0], [-1.0,  2.0]]), np.array([[ 1.0]]))

R1 = np.array([[2.0, 0.0], [0.0, 1.0]])

U1 = rng.multivariate_normal(mean=np.zeros(4), cov=G1, size=(ngr1))
U2 = rng.multivariate_normal(mean=np.zeros(2), cov=G2, size=(ngr2))

Uy1 = np.concatenate([U1[:, :2].reshape(-1, 1, order='C'), U2[:, :1].reshape(-1, 1)], axis=0)
Uy2 = np.concatenate([U1[:, 2:].reshape(-1, 1, order='C'), U2[:, 1:].reshape(-1, 1)], axis=0)

U = np.concatenate([Uy1, Uy2], axis=1)

E = sp.stats.multivariate_normal(mean=np.zeros(2), cov=R1).rvs(n_obs, random_state=rng)
Y = X.dot(B) + Z.dot(U) + E
y = vec(Y.T)
data[['y1', 'y2']] = Y

formula = "(y1, y2)~x1+x2+x3+x4+(1+x5|id1)+(1|id2)"


model = MLMM(formula, data)
model.fit(opt_kws=dict(verbose=3))
opt_cont = ('display.max_rows', None, 'display.max_columns', None,
                  'display.float_format', '{:.4f}'.format)
with pd.option_context(*opt_cont):
    print(model.res)






    