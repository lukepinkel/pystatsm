# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 13:28:52 2020

@author: lukepinkel
"""
import numpy as np
import pandas as pd
from pystats.utilities.random_corr import exact_rmvnorm, vine_corr
from pystats.pyglm.nb2 import NegativeBinomial
from pystats.utilities.numerical_derivs import fo_fc_cd, so_gc_cd

seed = 1234
rng = np.random.default_rng(seed)

n_obs, n_var, n_nnz, rsq, k = 5000, 20, 10, 0.5**2, 7.0
S = vine_corr(n_var, seed=seed)
X = exact_rmvnorm(S, n=n_obs, seed=seed)
beta = np.zeros(n_var)

bv = np.array([-1.0, -0.5, 0.5, 1.0])
bvals = np.tile(bv, n_nnz//len(bv))
if n_nnz%len(bv)>0:
    bvals = np.concatenate([bvals, bv[:n_nnz%len(bv)]])
    
beta[:n_nnz] = bvals
eta = X.dot(beta)
eta = eta / eta.var()
lpred = rng.normal(eta, scale=np.sqrt(eta.var()*(1.0 - rsq) / rsq))
mu = np.exp(lpred)
var = mu + k * mu**2
n = - mu**2 / (mu - var)
p = mu / var
y = rng.negative_binomial(n=n, p=p)


xcols = [f"x{i}" for i in range(1, n_var+1)]
data = pd.DataFrame(X, columns=xcols)
data['y'] = y

formula = "y~1+"+"+".join(xcols)
model = NegativeBinomial(formula=formula, data=data)
params_init = model.params.copy() + 0.01
model.fit()
params = model.params.copy()

g_num1 = fo_fc_cd(model.loglike, params_init)
g_ana1 = model.gradient(params_init)

g_num2 = fo_fc_cd(model.loglike, params)
g_ana2 = model.gradient(params)

H_num1 = so_gc_cd(model.gradient, params_init)
H_ana1 = model.hessian(params_init)

H_num2 = so_gc_cd(model.gradient, params)
H_ana2 = model.hessian(params)


np.allclose(g_num1, g_ana1)
np.allclose(g_num2, g_ana2, atol=1e-6)

np.allclose(H_num1, H_ana1)
np.allclose(H_num2, H_ana2)


