#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 17:29:55 2020

@author: lukepinkel
"""

import timeit # analysis:ignore
import numpy as np # analysis:ignore
import scipy as sp # analysis:ignore
import seaborn as sns # analysis:ignore
import matplotlib as mpl# analysis:ignore
from ..pylmm.lmm import LME # analysis:ignore
import scipy.sparse as sps # analysis:ignore
import matplotlib.pyplot as plt# analysis:ignore
from sksparse.cholmod import cholesky # analysis:ignore
from .test_data import generate_data # analysis:ignore
from ..utilities.random_corr import vine_corr # analysis:ignore
from ..utilities.linalg_operations import invech, vech, scholesky # analysis:ignore
from ..utilities.special_mats import (kronvec_mat, dmat)# analysis:ignore



formula = "y~x1+x5-1+(1+x2|id1)+(1|id2)+(1+x3+x4|id3)"
model_dict = {}
model_dict['gcov'] = {'id1':invech(np.array([1., 0.2, 1.])),
                      'id2':np.array([[1.0]]),
                      'id3':invech(np.array([1., -0.2, -0.2 , 1., 0.3, 1.]))}

model_dict['ginfo'] = {'id1':dict(n_grp=800, n_per=20),
                       'id2':dict(n_grp=200 , n_per=80),
                       'id3':dict(n_grp=400, n_per=40)}
 
model_dict['mu'] = np.zeros(5)
model_dict['vcov'] = vine_corr(5, 20)
model_dict['beta'] = np.array([2, -2])
model_dict['n_obs'] = 16000
df1, formula1 = generate_data(formula, model_dict, r=0.6**0.5)


model1 = LME(formula1, df1)
model1.loglike(model1.theta)
model1.gradient(model1.theta)

model1._fit()

g_theta = model1.gradient(model1.theta)

def z_sparsity(dims):
    res = {}
    num = 0
    den = 0
    for key, value in dims.items():
        if key!='error':
            nl, nv = value['n_groups'], value['n_vars']
            res[key] = 1 - nv * (nl - 1) / (nv*nl)
            num += nv * (nl - 1)
            den += nv*nl
    res['total'] = 1 - num/den
    return res
        
        
def func(k, r, frac_k, n0):
    x = frac_k*k
    return np.log((x*(k-n0)) / (n0*(k-x))) / r
    
    
        
        
        
    