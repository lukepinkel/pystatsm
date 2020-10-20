#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 17 02:25:24 2020

@author: lukepinkel
"""

import numpy as np  # analysis:ignore
import scipy as sp  # analysis:ignore
import scipy.stats  # analysis:ignore
import pandas as pd  # analysis:ignore
from ..utilities.random_corr import vine_corr, onion_corr # analysis:ignore
from ..pylmm.lmm import LME # analysis:ignore
import seaborn as sns
import matplotlib.pyplot as plt
n_grp = 300
n_per = 30

j = np.ones((n_per, ))
u = sp.stats.norm(0, 2).rvs(n_grp)
z = np.kron(u, j)
r = 0.9
r2 = r**2
x = sp.stats.norm(-z, np.sqrt((1-r2) / r2 * u.var())).rvs()

df = pd.DataFrame(np.zeros((n_grp*n_per, 4)), columns=['x', 'y', 'z', 'id'])
df['x'] = x
df['z'] = z
df['id'] = np.kron(np.arange(n_grp), j)

X = df[['x']]
X['const'] = 1

X = X[['const', 'x']]

beta = np.array([-4.0, 0.5])

eta = X.dot(beta) + df['z']

r = 0.7
r2 = r**2
v = np.sqrt((1-r2) / r2 * eta.var())
df['y'] = sp.stats.norm(eta, v).rvs()

mod = LME("y~x+1+(1|id)", data=df)
mod._fit(optimizer_kwargs=dict(options=dict(gtol=1e-16, xtol=1e-16, verbose=3)))
print(mod.params)

scatter_kws = dict(s=1.0, alpha=0.5, c=df['z'], color=None)
fig, ax = plt.subplots(nrows=2, ncols=2)
sns.regplot('x', 'y', data=df, ax=ax[0, 0], scatter_kws=scatter_kws)
sns.regplot('x', 'y', data=df, ax=ax[0, 1], x_partial='z', scatter_kws=scatter_kws)
sns.regplot('x', 'y', data=df, ax=ax[1, 0], y_partial='z', scatter_kws=scatter_kws)
sns.regplot('x', 'y', data=df, ax=ax[1, 1], y_partial='z', x_partial='z', 
            scatter_kws=scatter_kws)
R = df[['x', 'y', 'z']].corr()
S = df[['x', 'y', 'z']].cov()
v = np.diag(S)
sd = np.sqrt(v)
Rinv = pd.DataFrame(np.linalg.inv(R), index=['x', 'y', 'z'], columns=['x', 'y', 'z'])
Sinv = pd.DataFrame(np.linalg.inv(S), index=['x', 'y', 'z'], columns=['x', 'y', 'z'])







