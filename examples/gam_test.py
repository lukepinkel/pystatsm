# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 22:03:25 2021

@author: lukepinkel
"""

import numpy as np
import pandas as pd
from pystats.pygam.gam import GAM, Gamma
from pystats.pyglm.links import LogLink
from pystats.utilities.linalg_operations import dummy
from pystats.utilities.numerical_derivs import fo_fc_cd, so_gc_cd
rng = np.random.default_rng(123)


n_obs = 10_000
X = pd.DataFrame(np.zeros((n_obs, 4)), columns=['x0', 'x1', 'x2', 'y'])
 
X['x0'] = rng.choice(np.arange(5), size=n_obs, p=np.ones(5)/5)
X['x1'] = rng.uniform(-1, 1, size=n_obs)
X['x2'] = rng.uniform(-2, 2, size=n_obs)

u0 =  dummy(X['x0']).dot(np.array([-0.2, 0.2, -0.2, 0.2, 0.0]))
f1 = (3.0 * X['x1']**3 - 2.43 * X['x1'])
f2 = (X['x2']**3 - X['x2']) / 10
eta =  u0 + f1 + f2
mu = np.exp(eta) # mu = shape * scale
shape = mu * 2.0
X['y'] = rng.gamma(shape=shape, scale=1.0)
 


 
model = GAM("y~C(x0)+s(x1, kind='cr')+s(x2, kind='cr')", X, family=Gamma(link=LogLink))
model.fit()#approx_hess=False, opt_kws=dict(options=dict(gtol=1e-9, xtol=1e-500, maxiter=5000)))
theta = model.theta.copy()

np.allclose(np.array([49.617907465401451, 407.234892157726449]), np.exp(model.theta)[:-1])




# model.reml(theta)
# eps = np.finfo(float).eps**(1/4)
# np.allclose(model.gradient(theta), fo_fc_cd(model.reml, theta), atol=eps, rtol=eps)
# np.allclose(model.hessian(theta), so_gc_cd(model.gradient, theta), atol=eps, rtol=eps)

# beta, eta, mu, devp, success, i = model.pirls(np.exp(theta[:-1]))

# np.allclose(np.array([ 0.4773407, 0.4145791, 0.0275050, 0.4148305, 0.1751491, 0.7022753,  
#                        0.9934963, 0.8796078, 0.4450703,-0.0853405,-0.5365374,-0.6510326,
#                       -0.2447111, 0.5716490,-0.0080083, 0.1398632, 0.2014833, 0.2274299, 
#                        0.1810886, 0.1034674, 0.2092541, 0.3963561, 0.6344967]), beta)




