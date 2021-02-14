# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 13:28:52 2020

@author: lukepinkel
"""
      
import numpy as np
import scipy as sp
import scipy.stats
import pandas as pd
from pystats.pyglm.nb2 import NegativeBinomial
from pystats.pyglm.glm import GLM, Poisson
import seaborn as sns
from pystats.utilities.random_corr import vine_corr


pd.set_option('display.max_columns', None)
np.set_printoptions(suppress=True)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
n_obs = 10_000
n_vars = 4
n_true = 2
S = vine_corr(n_vars)/5
X = np.random.multivariate_normal(np.zeros(n_vars), S, size=(n_obs))
X -= X.mean(axis=0)
beta = np.zeros(n_vars)
beta[np.random.choice(n_vars, n_true, replace=False)] = np.random.choice([-0.5, 0.5], n_true)
eta = X.dot(beta)
mu = np.exp(eta)
a = 0.1
r = 1.0 / a
p = mu / (mu + r)

y = sp.stats.nbinom(n=r, p=p).rvs()

df = pd.DataFrame(np.hstack((X, y[:, None])), 
                  columns=[f"x{i}" for i in range(n_vars)]+['y'])
formula = "y~"+"+".join([f"x{i}" for i in range(n_vars)])
model = NegativeBinomial(formula, data=df)
model.fit()
model.bootstrap()
beta_samples = pd.DataFrame(model.beta_samples, columns=model.res.index)
g = sns.PairGrid(beta_samples)
g.map_upper(sns.scatterplot)
g.map_lower(sns.kdeplot)
g.map_diag(sns.kdeplot)
vcov = beta_samples.cov()
vcor = beta_samples.corr()

glm_poisson = GLM(formula, df, fam=Poisson())
glm_poisson.fit()