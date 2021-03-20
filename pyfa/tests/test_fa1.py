#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 03:44:38 2020

@author: lukepinkel
"""

from jax.config import config
config.update("jax_enable_x64", True)
import jax
import numpy as np
import pandas as pd

from pystats.pyfa.factor_analysis2 import FactorAnalysis, multi_rand




Lambda = np.zeros((12, 3))
Lambda[0:4, 0] = 1
Lambda[4:8, 1] = 1
Lambda[8: , 2] = 1
Phi = np.eye(3)
Psi = np.diag(np.random.uniform(low=0.2, high=1.0, size=12))

S = Lambda.dot(Phi).dot(Lambda.T) + Psi
S = pd.DataFrame(S, index=['x%i'%i for i in range(1, 13)])
S.columns = S.index
X = multi_rand(S)
X = pd.DataFrame(X, columns=['x%i'%i for i in range(1, 13)])


model = FactorAnalysis(X, 3, loadings_free=Lambda)
model.fit()
print(model.res)
print(model.sumstats)

data = pd.read_csv("/users/lukepinkel/Downloads/HolzingerSwineford1939.csv",
                   index_col=0)

data = data.iloc[:, -9:]

Lambda = np.zeros((9, 3))
Lambda[0:3, 0] = 1
Lambda[3:6, 1] = 1
Lambda[6: , 2] = 1

Lambda[0, 0] = 0
Lambda[3, 1] = 0
Lambda[6, 2] = 0

Phi = np.zeros((3, 3))
Phi[np.tril_indices(3)] = 1
model = FactorAnalysis(data, 3, loadings_free=Lambda, lcov_free=Phi)
model.LA = jax.ops.index_update(model.LA, [[0, 3, 6], [0, 1, 2]], 1)
model.fit()
print(model.res)
print(model.sumstats)




