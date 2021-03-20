#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 23:20:27 2020

@author: lukepinkel
"""
import numpy as np # analysis:ignore
import scipy as sp # analysis:ignore
import scipy.stats # analysis:ignore
import pandas as pd # analysis:ignore
from .rls import RLS # analysis:ignore

np.random.seed(100)

X = np.random.normal(size=(1000, 4))
b = np.array([-2, 2, 1, 1])
eta = X.dot(b)
eta_var = eta.var()
sd = np.sqrt((1 - 0.7) / 0.7 * eta_var)
y = sp.stats.t(df=5, loc=eta, scale=sd).rvs()


data = pd.DataFrame(X, columns=['x1', 'x2', 'x3', 'x4'])
data['y'] = y - 2

model = RLS("y~x1+x2+x3+x4", data)
model.fit()
vals = np.array([-2.06131818, -1.88484841,  1.97482784,  1.05446007,  0.99190487])
print(np.allclose(vals, model.beta[:, 0]))