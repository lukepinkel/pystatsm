#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 23 03:54:38 2020

@author: lukepinkel
"""
import numpy as np
import scipy as sp
import scipy.stats
import pandas as pd
from .linalg_operations import _check_shape

def get_param_table(params, se_params, degfree=None, index=None,
                    parameter_label=None, pdist=None, p_const=2.0):
    if parameter_label is None:
        parameter_label = 'parameter'
    arr = np.vstack((_check_shape(params, 1), _check_shape(se_params, 1))).T
    df = pd.DataFrame(arr, index=index, columns=[parameter_label, 'SE'])
    df['t'] = df[parameter_label] / df['SE']
    if pdist is None:
        pdist = sp.stats.t(degfree)
    df['p'] = pdist.sf(np.abs(df['t'])) * p_const
    return df
    
    