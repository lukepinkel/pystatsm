#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 01:22:59 2020

@author: lukepinkel
"""

import numpy as np              # analysis:ignore
import scipy as sp              # analysis:ignore
import pandas as pd             # analysis:ignore
import mvpy.api as mv           # analysis:ignore
import seaborn as sns           # analysis:ignore
import matplotlib as mpl        # analysis:ignore
import matplotlib.pyplot as plt # analysis:ignore
from pyglm.glm import GLM, Binomial, LogitLink# analysis:ignore

link = LogitLink()
X = pd.DataFrame(np.zeros((1000, 6)), 
                 columns=['const']+['x%i'%i for i in range(1, 6)])

X['const'] = 1
X['x1'] = sp.stats.binom(1, 0.7).rvs(1000)
X['x2'] = sp.stats.binom(1, 0.5).rvs(1000)
X[['x3','x4','x5']] = sp.stats.multivariate_normal(np.zeros(3), 
                                                   mv.vine_corr(3)).rvs(1000)
X = X.sort_values(['x1', 'x2'])

beta = np.array([0.5, -0.5, 1.0, -1.0, 2.0, -2.0])

eta1 = X.dot(beta)
eta2 = sp.stats.norm(eta1, np.sqrt((1-0.3**0.5)/(0.3**0.5)*eta1.var())).rvs()
mu1 = link.inv_link(eta1)
mu2 = link.inv_link(eta2)
data = X.copy()
data['y1'] = sp.stats.binom(1, mu1).rvs()
data['y2'] = sp.stats.binom(1, mu2).rvs()
model1 = GLM("y1~1+x1+x2+x3+x4+x5", data, fam=Binomial)
model1.fit()

model2 = GLM("y2~1+x1+x2+x3+x4+x5", data, fam=Binomial)
model2.fit()

model1.res
model2.res

model1.sumstats
model2.sumstats