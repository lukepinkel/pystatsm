#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 17 19:35:55 2020

@author: lukepinkel
"""


import numpy as np
import pandas as pd


def check_type(X):
    if type(X) is pd.DataFrame:
        X,columns,index,is_pd=X.values,X.columns.values,X.index.values,True 
    elif type(X) is pd.Series:
        X, columns, index, is_pd = X.values, X.name, X.index, True
        X = X.reshape(X.shape[0], 1)
    elif type(X) is np.ndarray:
        X, columns, index, is_pd = X, None, None, False 
        if X.ndim == 1:
            X = X.reshape(X.shape[0], 1)
    return X, columns, index, is_pd 

