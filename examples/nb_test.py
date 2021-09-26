# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 13:28:52 2020

@author: lukepinkel
"""

import numpy as np
import pandas as pd
from pystats.utilities.random import exact_rmvnorm
from pystats.pyglm.nb2 import NegativeBinomial
from pystats.utilities.numerical_derivs import fo_fc_cd, so_gc_cd

seed = 1234
rng = np.random.default_rng(seed)

n_obs, n_var, n_nnz, rsq, k = 2000, 20, 4, 0.9**2, 4.0
X = exact_rmvnorm(np.eye(n_var), n=n_obs, seed=seed)
beta = np.zeros(n_var)

bv = np.array([-1.0, -0.5, 0.5, 1.0])
bvals = np.tile(bv, n_nnz//len(bv))
if n_nnz%len(bv)>0:
    bvals = np.concatenate([bvals, bv[:n_nnz%len(bv)]])
    
beta[:n_nnz] = bvals
eta = X.dot(beta) / np.sqrt(np.sum(beta**2))
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

theta = np.array([ 0.13049303, -0.64878454, -0.30956394,  0.2903795 ,  0.58677555,
                  -0.03022705,  0.03989469,  0.01182953, -0.00498391,  0.00788808,
                  -0.04198716, -0.00162041,  0.01523861, -0.00401566, -0.02547227,
                  -0.07309814, -0.05574522,  0.00938691, -0.0034148 , -0.01254539,
                  -0.05221309,  1.41286364])

g_num1 = fo_fc_cd(model.loglike, params_init)
g_ana1 = model.gradient(params_init)

g_num2 = fo_fc_cd(model.loglike, params)
g_ana2 = model.gradient(params)

H_num1 = so_gc_cd(model.gradient, params_init)
H_ana1 = model.hessian(params_init)

H_num2 = so_gc_cd(model.gradient, params)
H_ana2 = model.hessian(params)

assert(np.allclose(model.params, theta))
assert(np.allclose(g_num1, g_ana1))
assert(np.allclose(g_num2, g_ana2, atol=1e-4))
assert(np.allclose(H_num1, H_ana1))
assert(np.allclose(H_num2, H_ana2))
assert(model.opt_full.success)

