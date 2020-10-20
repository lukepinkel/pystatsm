#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 22:38:42 2020

@author: lukepinkel
"""


import numpy as np # analysis:ignore
import scipy as sp # analysis:ignore
import pandas as pd # analysis:ignore
import scipy.sparse as sps # analysis:ignore
from ..pylmm.glmm import GLMM # analysis:ignore
from .test_data import generate_data # analysis:ignore
from ..utilities.random_corr import vine_corr # analysis:ignore
from ..utilities.linalg_operations import invech, vech, _check_shape # analysis:ignore
from ..pylmm.families import Binomial, Poisson, Gaussian # analysis:ignore
from ..pylmm.links import LogitLink, LogLink# analysis:ignore
from sksparse.cholmod import cholesky# analysis:ignore
def prediction_table(y, yhat):
    y = _check_shape(y, 1)
    yhat = _check_shape(yhat, 1)
    
    TPR = np.sum((y==1)*(yhat==1))
    FPR = np.sum((y==0)*(yhat==1))
    TNR = np.sum((y==0)*(yhat==0))
    FNR = np.sum((y==1)*(yhat==0))
    table = pd.DataFrame([[TNR, FPR],
                          [FNR, TPR]],
                          index=['Negative', 'Positive'],
                          columns=['Predicted Negative', 'Predicted Positive'])
    return table
    

formula = "y~x1+x2+(1|id1)"
model_dict = {}
model_dict['gcov'] = {'id1':invech(np.array([2.0]))}
model_dict['ginfo'] = {'id1':dict(n_grp=100, n_per=10)} 
model_dict['mu'] = np.zeros(2)
model_dict['vcov'] = vine_corr(2)
model_dict['beta'] = np.array([0.5, 0.5, -1.0])
model_dict['n_obs'] = 1000


df, _ = generate_data(formula, model_dict, r=0.7**0.5)


df = df.rename(columns=dict(y='eta'))
df['mu'] = LogitLink().inv_link(df['eta'])
df['y'] = sp.stats.binom(n=1, p=df['mu']).rvs()

model = GLMM(formula, df, fam=Binomial())
model.fit(tol=1e-6, optimizer_kwargs=dict(options=dict(gtol=1e-16, 
                                                       xtol=1e-30,
                                                       verbose=3))) 

df2, _ = generate_data(formula, model_dict, r=0.7**0.5)

df2 = df2.rename(columns=dict(y='eta'))
df2['eta'] /= 2.5

df2['mu'] = LogLink().inv_link(df2['eta'])
df2['y'] = sp.stats.poisson(mu=df2['mu']).rvs()

model2 = GLMM(formula, df2, fam=Poisson())
model2.fit(tol=1e-2) 


df3, _ = generate_data(formula, model_dict, r=0.7**0.5)


model3 = GLMM(formula, df3, fam=Gaussian())
model3.fit(tol=1e-2) 

np.random.seed(422)

n_grp = 150
n_per = 30

j = np.ones((n_per, ))
u = sp.stats.norm(0, 2).rvs(n_grp)
z = np.kron(u, j)
r = 0.8
r2 = r**2
x = sp.stats.norm(-z, np.sqrt((1-r2) / r2 * u.var())).rvs()

df4 = pd.DataFrame(np.zeros((n_grp*n_per, 4)), columns=['x', 'y', 'z', 'id'])
df4['x'] = x
df4['z'] = z
df4['id'] = np.kron(np.arange(n_grp), j)

X = df4[['x']]
X['const'] = 1

X = X[['const', 'x']]

beta = np.array([-0.5, 0.6])

eta = X.dot(beta) + df4['z']
r = 0.8
r2 = r**2
v = np.sqrt((1-r2) / r2 * eta.var())
df4['eta'] = sp.stats.norm(eta, v).rvs()
df4['mu'] = LogitLink().inv_link(df4['eta'])
df4['y'] = sp.stats.binom(n=1, p=df4['mu']).rvs()

model4 = GLMM("y~x+1+(1|id)", data=df4, fam=Binomial())
model4.fit()


