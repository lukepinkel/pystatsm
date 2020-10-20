#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 16 23:05:03 2020

@author: lukepinkel
"""
import timeit # analysis:ignore
import numpy as np # analysis:ignore
import scipy as sp # analysis:ignore
from ..pylmm.lmm import LME # analysis:ignore
import scipy.sparse as sps # analysis:ignore
from .test_data import generate_data # analysis:ignore
from ..utilities.random_corr import vine_corr # analysis:ignore
from ..utilities.linalg_operations import invech, vech, scholesky # analysis:ignore
from sksparse.cholmod import cholesky # analysis:ignore



formula = "y~x1+x5-1+(1+x2|id1)+(1|id2)+(1+x3+x4|id3)"
model_dict = {}
model_dict['gcov'] = {'id1':invech(np.array([1., 0.2, 1.])),
                      'id2':np.array([[1.0]]),
                      'id3':invech(np.array([1., -0.2, -0.2 , 1., 0.3, 1.]))}

model_dict['ginfo'] = {'id1':dict(n_grp=200, n_per=20),
                       'id2':dict(n_grp=50 , n_per=80),
                       'id3':dict(n_grp=100, n_per=40)}
 
model_dict['mu'] = np.zeros(5)
model_dict['vcov'] = vine_corr(5, 20)
model_dict['beta'] = np.array([2, -2])
model_dict['n_obs'] = 4000
df1, formula1 = generate_data(formula, model_dict, r=0.6**0.5)


model1 = LME(formula1, df1)
model1._fit()

g = model1.gradient(model1.theta)


formula = "y~x1+x2+(1+x3+x4+x5|id1)"
model_dict = {}
model_dict['gcov'] = {'id1':invech(np.array([1., -0.2, 0.4, 0.33, 1.0, 
                                              -0.2, -0.4, 1.0, 0.3, 1.0]))}


model_dict['ginfo'] = {'id1':dict(n_grp=200, n_per=20)} 
model_dict['mu'] = np.zeros(5)
model_dict['vcov'] = vine_corr(5)
model_dict['beta'] = np.array([3.0, 2, -2])
model_dict['n_obs'] = 4000

df2, formula2 = generate_data(formula, model_dict, r=0.6**0.5)
model2 = LME(formula2, df2)

model2._fit()




formula = "y~x1+x2+(1|id1)"
model_dict = {}
model_dict['gcov'] = {'id1':invech(np.array([4.0]))}
model_dict['ginfo'] = {'id1':dict(n_grp=1000, n_per=10)} 
model_dict['mu'] = np.zeros(2)
model_dict['vcov'] = vine_corr(2)
model_dict['beta'] = np.array([3.0, 2, -2])
model_dict['n_obs'] = 10_000

df3, formula3 = generate_data(formula, model_dict, r=0.6**0.5)

model3 = LME(formula3, df3)
model3._fit()



