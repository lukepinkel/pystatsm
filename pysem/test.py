#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 22:00:15 2020

@author: lukepinkel
"""



import numpy as np
import scipy as sp
import scipy.stats
import pandas as pd
from .sem import SEM

data_bollen = pd.read_csv("/users/lukepinkel/Downloads/bollen.csv", index_col=0)
data_bollen = data_bollen[['x1', 'x2', 'x3', 'y1', 'y2', 'y3', 'y4', 'y5',
             'y6', 'y7', 'y8', ]]

L = np.array([[1., 0., 0.],
              [1., 0., 0.],
              [1., 0., 0.],
              [0., 1., 0.],
              [0., 1., 0.],
              [0., 1., 0.],
              [0., 1., 0.],
              [0., 0., 1.],
              [0., 0., 1.],
              [0., 0., 1.],
              [0., 0., 1.]])

B = np.array([[0., 0., 0.],
              [1., 0., 0.],
              [1., 1., 0.]])

Lambda1 = pd.DataFrame(L, index=data_bollen.columns, columns=['ind60', 'dem60', 'dem65'])
Beta1 = pd.DataFrame(B, index=Lambda1.columns, columns=Lambda1.columns)
S1 = data_bollen.cov()
Psi1 = pd.DataFrame(np.eye(Lambda1.shape[0]), index=Lambda1.index, columns=Lambda1.index)

off_diag = [['y1', 'y5'], ['y2', 'y4'], ['y3', 'y7'], ['y4', 'y8'],
            ['y6', 'y8'], ['y2', 'y6']]
for x, y in off_diag:
    Psi1.loc[x, y] = Psi1.loc[y, x] = 0.05
Phi1 = pd.DataFrame(np.eye(Lambda1.shape[1]), index=Lambda1.columns,
                   columns=Lambda1.columns)

model = SEM(Lambda1, Beta1, Phi1, Psi1, data=data_bollen)
model.fit()




################################################################


x = sp.stats.norm(0, 1).rvs(1000)
x = (x - x.mean()) / x.std()

u = sp.stats.norm(0, 1).rvs(1000)
u = (u - u.mean()) / u.std()
v = sp.stats.norm(0, 1).rvs(1000)
v = (v - v.mean()) / v.std()

m = 0.5*x + u
y = 0.7*m + v

data_path = pd.DataFrame(np.vstack((x, m, y)).T, columns=['x', 'm', 'y'])
data_path = data_path -data_path.mean(axis=0)
Lambda2 = pd.DataFrame(np.eye(3), index=data_path.columns, columns=data_path.columns)
Beta2 = np.array([[0.0, 0.0, 0.0],
                  [1.0, 0.0, 0.0],
                  [1.0, 1.0, 0.0]])
Beta2 = pd.DataFrame(Beta2, index=data_path.columns, columns=data_path.columns)

Phi2 = np.array([[1.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0]])
Phi2 = pd.DataFrame(Phi2, index=data_path.columns, columns=data_path.columns)

Psi2 = np.array([[0.0, 0.0, 0.0],
                 [0.0, 1.0, 0.0],
                 [0.0, 0.0, 1.0]])
Psi2 = pd.DataFrame(Psi2, index=data_path.columns, columns=data_path.columns)


model2 = SEM(Lambda2, Beta2, Phi2, Psi2, data=data_path)
model2.fit(opt_kws=dict(options=dict(gtol=1e-20, xtol=1e-100)))







