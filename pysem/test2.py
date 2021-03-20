#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 17 20:06:31 2020

@author: lukepinkel
"""

import numpy as np
import pandas as pd
from .sem import MLSEM

data = pd.read_csv("/users/lukepinkel/Downloads/bollen.csv", index_col=0)
data = data[['x1', 'x2', 'x3', 'y1', 'y2', 'y3', 'y4', 'y5',
             'y6', 'y7', 'y8', ]]
L = np.array([[1, 0, 0],
              [1, 0, 0],
              [1, 0, 0],
              [0, 1, 0],
              [0, 1, 0],
              [0, 1, 0],
              [0, 1, 0],
              [0, 0, 1],
              [0, 0, 1],
              [0, 0, 1],
              [0, 0, 1]])
B = np.array([[False, False, False],
              [True,  False, False],
              [True,  True, False]])
LA = pd.DataFrame(L, index=data.columns, columns=['ind60', 'dem60', 'dem65'])
BE = pd.DataFrame(B, index=LA.columns, columns=LA.columns)
S = data.cov()
Zg = ZR = data
Lambda=LA!=0
Beta=BE!=0 
Lambda, Beta = pd.DataFrame(Lambda), pd.DataFrame(Beta)
Lambda.columns = ['ind60', 'dem60', 'dem65']
Lambda.index = Zg.columns
Beta.columns = Lambda.columns
Beta.index = Lambda.columns
Theta = pd.DataFrame(np.eye(Lambda.shape[0]),
                     index=Lambda.index, columns=Lambda.index)
Theta.loc['y1', 'y5'] = 0.05
Theta.loc['y2', 'y4'] = 0.05
Theta.loc['y2', 'y6'] = 0.05
Theta.loc['y3', 'y7'] = 0.05
Theta.loc['y4', 'y8'] = 0.05
Theta.loc['y6', 'y8'] = 0.05
Theta.loc['y5', 'y1'] = 0.05
Theta.loc['y4', 'y2'] = 0.05
Theta.loc['y6', 'y2'] = 0.05
Theta.loc['y7', 'y3'] = 0.05
Theta.loc['y8', 'y4'] = 0.05
Theta.loc['y8', 'y6'] = 0.05
  
mod = MLSEM(Zg, Lambda,  Beta, Theta.values)
mod.fit()



