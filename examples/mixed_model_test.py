# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 19:49:46 2020

@author: lukepinkel
"""

import numpy as np
from pystats.pylmm.lmm import LMM
from pystats.pylmm.test_data import generate_data
from pystats.utilities.linalg_operations import invech
rng = np.random.default_rng(1234)

n_grp=200
n_per=20
formula = "y~x1+x2+(1+x3|id1)"
model_dict = {}
model_dict['gcov'] = {'id1':invech(np.array([2.0, np.sqrt(2)/2, 1.0]))}
model_dict['ginfo'] = {'id1':dict(n_grp=n_grp, n_per=n_per)} 
model_dict['mu'] = np.zeros(3)
model_dict['vcov'] = np.eye(3)
model_dict['beta'] = np.array([3.0, 2, -2])
model_dict['n_obs'] = n_grp*n_per

df, formula = generate_data(formula, model_dict, r=0.6**0.5, rng=rng)
model = LMM(formula, df)
model.fit(opt_kws=dict(verbose=3))
g = model.gradient(model.theta.copy())


assert(np.allclose(model.theta, np.array([2.11153409, 0.68077356, 0.91666827, 7.56632386])))
assert(np.allclose(np.zeros_like(g), g, atol=1e-5))
assert(model.optimizer.success==True)
