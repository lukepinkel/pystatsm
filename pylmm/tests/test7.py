#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 10:34:31 2020

@author: lukepinkel
"""

import timeit # analysis:ignore
import numpy as np # analysis:ignore
import scipy as sp # analysis:ignore
import seaborn as sns # analysis:ignore
import matplotlib as mpl# analysis:ignore
from ..pylmm.lmm import LME # analysis:ignore
from ..pylmm.model_matrices import vech2vec, get_jacmats2, jac2deriv # analysis:ignore
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




timeit.timeit("model1.gradient(model1.theta)", globals=globals(), number=1)

timeit.timeit("model1.hessian(model1.theta)", globals=globals(), number=1)
]\

