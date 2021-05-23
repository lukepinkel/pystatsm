# -*- coding: utf-8 -*-
"""
Created on Sun May 23 16:39:50 2021

@author: lukepinkel
"""
import numpy as np
import pandas as pd
from pystats.pyglm.clm import CLM
from pystats.utilities.random_corr import exact_rmvnorm
from pystats.utilities.numerical_derivs import fo_fc_cd, so_gc_cd
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 250)
np.set_printoptions(suppress=True)

rng = np.random.default_rng(1234)

n_obs, n_var, rsquared = 10_000, 8, 0.25
S = np.eye(n_var)
X = exact_rmvnorm(S, n=n_obs, seed=1234)
beta = np.zeros(n_var)
beta[np.arange(n_var//2)] = rng.choice([-1., 1., -0.5, 0.5], n_var//2)
var_names = [f"x{i}" for i in range(1, n_var+1)]

eta = X.dot(beta)
eta_var = eta.var()

scale = np.sqrt((1.0 - rsquared) / rsquared * eta_var)
y = rng.normal(eta, scale=scale)

df = pd.DataFrame(X, columns=var_names)
df["y"] = pd.qcut(y, 7).codes

formula = "y~-1+"+"+".join(var_names)

model = CLM(frm=formula, data=df)
model.fit()

params_init, params = model.params_init.copy(),  model.params.copy()

tol = np.finfo(float).eps**(1/3)
np.allclose(model.gradient(params_init), fo_fc_cd(model.loglike, params_init))
np.allclose(model.gradient(params), fo_fc_cd(model.loglike, params), atol=tol)


np.allclose(model.hessian(params_init), so_gc_cd(model.gradient, params_init))
np.allclose(model.hessian(params), so_gc_cd(model.gradient, params))

