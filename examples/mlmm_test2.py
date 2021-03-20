# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 01:58:20 2021

@author: lukepinkel
"""
import scipy as sp
import scipy.stats
import numpy as np
import pandas as pd
from pystats.pylmm.mlmm import (MLMM, construct_model_matrices, vec, inverse_transform_theta)
from pystats.utilities.random_corr import vine_corr, exact_rmvnorm

rng = np.random.default_rng(123)
n_groups = 200
n_per = 10
n_obs = n_groups * n_per

S = vine_corr(4, seed=123)
X = exact_rmvnorm(S, n_obs, seed=123)
df = pd.DataFrame(X, columns=['x1', 'x2', 'x3', 'x4'])
df['y'] = 0
df['id1'] = np.repeat(np.arange(n_groups), n_per)

formula = "y~x1+x2+x3+x4+(1|id1)"
X, Z, y, dims, levels, _, _ = construct_model_matrices(formula, data=df)
B = rng.normal(0, 4, size=(5, 2))

A = np.eye(n_groups)
Sigma_u = np.array([[2.0, 1.5], [1.5, 2.0]])
Sigma_e = np.array([[2.0, 0], [0, 1.0]])
U = sp.stats.matrix_normal(mean=np.zeros((A.shape[0], Sigma_u.shape[0])),
                           rowcov=A, colcov=Sigma_u).rvs(1, random_state=rng)

E = sp.stats.multivariate_normal(mean=np.zeros(2), cov=Sigma_e).rvs(n_obs,random_state=rng)
Y = X.dot(B) + Z.dot(U) + E
y = vec(Y.T)
df[['y1', 'y2']] = Y

formula = "(y1, y2)~x1+x2+x3+x4+(1|id1)"


model = MLMM(formula, df)


res_hess_appr = sp.optimize.minimize(model.loglike_c, model.theta, jac=model.gradient_chol, 
                                hess='3-point', bounds=model.bounds, 
                                method='trust-constr', 
                                options=dict(verbose=3))


res_hess_true = sp.optimize.minimize(model.loglike_c, model.theta, jac=model.gradient_chol, 
                                     hess=model.hessian_chol, bounds=model.bounds, 
                                     method='trust-constr', 
                                     options=dict(verbose=3, xtol=1e-21))
inverse_transform_theta(res_hess_true.x.copy(), dims, model.indices)

    
    