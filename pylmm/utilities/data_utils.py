#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 21:28:03 2020

@author: lukepinkel
"""
import numpy as np
import pandas as pd


def _check_type(arr):
    if type(arr) is pd.DataFrame:
        X = arr.values
        columns, index = arr.columns, arr.index
        is_pd = True
    elif type(arr) is pd.Series:
        X = arr.values
        columns, index = arr.name, arr.index
        is_pd = True
        X = X.reshape(X.shape[0], 1)
    elif type(arr) is np.ndarray:
        X, columns, index, is_pd = arr, None, None, False 
        if X.ndim == 1:
            X = X.reshape(X.shape[0], 1)
    return X, columns, index, is_pd 
    