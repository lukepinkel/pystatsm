#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 20:41:17 2020

@author: lukepinkel
"""
import numpy as np
import scipy as sp
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ..utilities import random_corr
from .factor_analysis import FactorAnalysis
from .rotation import rotate
        
        
           

A = np.array([[2, 0, 0, 0],
              [1, 0, 0, 0],
              [1, 0, 0, 0],
              [0, 1, 0, 0],
              [0, 1, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 1, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 1],
              [0, 0, 0, 1],
              [0, 0, 0, 1]])
    
Phi = np.array([[ 1.00,  0.09,  0.00,  0.00],
                [ 0.09,  1.00,  0.09,  0.00],
                [ 0.00,  0.09,  1.00,  0.09],
                [ 0.00,  0.00,  0.09,  1.00]])
    

Psi = np.diag((np.tile([0.3, 0.6, 0.9], 4)))**2

Sigma_gen = A.dot(Phi).dot(A.T)+Psi
X = random_corr.multi_rand(Sigma_gen)
FLC = np.zeros((4, 4))
FLC[np.tril_indices(4, -1)] = 1
model = FactorAnalysis(X, 4)
model.fit()
print(np.allclose(np.diag(Psi)**0.5, np.diag(model.Psi)))

res = model.res
L = model.Lambda
LR, R = rotate(L, "varimax")
Lambda = pd.DataFrame(LR, columns=[f"Factor {i}" for i in range(4)],
                                    index=[f"x {i}" for i in range(12)])

ax = sns.heatmap(Lambda, cmap=plt.cm.bwr, vmin=-1, vmax=2, center=0)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
ax.set_xticks(np.arange(4), minor=True)
ax.set_yticks(np.arange(12), minor=True)

ax.grid(which='minor')

S_hat = pd.DataFrame(model.Sigma.copy())

S_hat - Sigma_gen

Xr = random_corr.multi_rand(Sigma_gen)


model = FactorAnalysis(Xr, 4, free_loadings=A.astype(bool), free_lcov=FLC)
model.fit()
res2 = model.res



        
           

A = np.array([[2, 0, 0, 0],
              [1, 0, 0, 0],
              [1, 0, 0, 0],
              [0, 1, 0, 0],
              [0, 1, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 1, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 1],
              [0, 0, 0, 1],
              [0, 0, 0, 1]])
    
Phi = np.array([[ 1.00,  0.20,  0.00,  0.00],
                [ 0.20,  1.00,  0.20,  0.00],
                [ 0.00,  0.20,  1.00,  0.20],
                [ 0.00,  0.00,  0.20,  1.00]])
    

Psi = np.diag((np.tile([0.3, 0.6, 0.9], 4)))**2

Sigma_gen = A.dot(Phi).dot(A.T)+Psi
X = random_corr.multi_rand(Sigma_gen)
FLC = np.zeros((4, 4))
FLC[np.tril_indices(4, -1)] = 1
model = FactorAnalysis(X, 4)
model.fit()
print(np.allclose(np.diag(Psi)**0.5, np.diag(model.Psi)))

res = model.res
L = model.Lambda
LR, R = rotate(L, "varimax")
Lambda = pd.DataFrame(LR, columns=[f"Factor {i}" for i in range(4)],
                                    index=[f"x {i}" for i in range(12)])
fig, ax = plt.subplots()
ax = sns.heatmap(Lambda, cmap=plt.cm.bwr, vmin=-1, vmax=2, center=0, ax=ax)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
ax.set_xticks(np.arange(4), minor=True)
ax.set_yticks(np.arange(12), minor=True)

ax.grid(which='minor')

S_hat = pd.DataFrame(model.Sigma.copy())

S_hat - Sigma_gen

Xr = random_corr.multi_rand(Sigma_gen)


model = FactorAnalysis(Xr, 4, free_loadings=A.astype(bool), free_lcov=FLC)
model.fit()
res2 = model.res


Sigma_gen = pd.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/psych/Thurstone.csv",
                   index_col=0)
X = pd.DataFrame(random_corr.multi_rand(Sigma_gen, 213), columns=Sigma_gen.index)
model1 = FactorAnalysis(X, 2)
model1.fit()
model2 = FactorAnalysis(X, 3)
model2.fit()

print(model1.sumstats)
print(model2.sumstats)


