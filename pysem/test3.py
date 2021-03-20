#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 17 20:13:08 2020

@author: lukepinkel
"""
import numpy as np
import pandas as pd
from .sem import MLSEM

data = pd.read_csv("/users/lukepinkel/Downloads/HolzingerSwineford1939.csv")
df = data.iloc[:, 7:]
LA = np.zeros((9, 3))
LA[0:3, 0] = 1.0
LA[3:6, 1] = 1.0
LA[6:9, 2] = 1.0

LA = pd.DataFrame(LA, index=df.columns, columns=['visual', 'textual', 'speed'])
BE = pd.DataFrame(np.zeros((3, 3)), index=LA.columns, columns=LA.columns)
PH=BE+np.eye(3)+0.05
TH = pd.DataFrame(np.eye(9), index=df.columns, columns=df.columns)
mod = MLSEM(df, LA, BE, PH=PH, TH=TH)
mod.fit()