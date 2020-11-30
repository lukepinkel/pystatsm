#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 21:34:27 2020

@author: lukepinkel
"""


import numpy as np # analysis:ignore
import pandas as pd # analysis:ignore
from .lvcorr import Polychoric, Polyserial
from ..utilities.random_corr import multi_rand

R = np.array([[1.0, 0.5],
              [0.5, 1.0]])
X = pd.DataFrame(multi_rand(R))
X[0] = pd.qcut(X[0], 3).cat.codes.astype(float)
X[1] = pd.qcut(X[1], 5).cat.codes.astype(float)

model = Polychoric(X[0], X[1])
model.fit()



R = np.array([[1.0, 0.5],
              [0.5, 1.0]])
X = pd.DataFrame(multi_rand(R))
X[1] = pd.qcut(X[1], 3).cat.codes.astype(float)

model = Polyserial(X[0], X[1])
model.fit()
