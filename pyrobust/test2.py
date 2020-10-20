#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 23:24:55 2020

@author: lukepinkel
"""


import numpy as np # analysis:ignore
import scipy as sp # analysis:ignore
import scipy.stats # analysis:ignore
import pandas as pd # analysis:ignore
from .rls import RLS # analysis:ignore

np.random.seed(200)

X = np.random.normal(size=(1000, 4))
b = np.array([-2, 2, 1, 1])
eta = X.dot(b)
eta_var = eta.var()
sd = np.sqrt((1 - 0.7) / 0.7 * eta_var)
y = sp.stats.norm(loc=eta, scale=sd).rvs()
k = int(0.05*len(y))

ix = np.random.choice(len(y), k, replace=False)
contaminant = sp.stats.cauchy(loc=-y[ix]*3).rvs()
y[ix] = contaminant



data = pd.DataFrame(X, columns=['x1', 'x2', 'x3', 'x4'])
data['y'] = y - 2
values = np.array([-2.07818837, -1.98437233,  1.78758703, 
                   1.0024736 ,  0.9680364 ])
model = RLS("y~x1+x2+x3+x4", data)
model.fit()

print(np.allclose(values, model.beta[:, 0]))