#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 31 01:15:39 2020

@author: lukepinkel
"""


import numpy as np # analysis:ignore
import scipy as sp # analysis:ignore
import pandas as pd # analysis:ignore
import scipy.sparse as sps # analysis:ignore
from .test_data import generate_data # analysis:ignore
from ..utilities.random_corr import vine_corr # analysis:ignore
from ..utilities.linalg_operations import invech, vech, _check_shape # analysis:ignore
from ..pylmm.families import Binomial # analysis:ignore
from ..pylmm.links import LogitLink # analysis:ignore

formula = "y~x1+x2+(1|id1)"
model_dict = {}
model_dict['gcov'] = {'id1':invech(np.array([2.0]))}
model_dict['ginfo'] = {'id1':dict(n_grp=100, n_per=10)} 
model_dict['mu'] = np.zeros(2)
model_dict['vcov'] = vine_corr(2)
model_dict['beta'] = np.array([0.5, 0.5, -1.0])
model_dict['n_obs'] = 1000


link_func = LogitLink()
df, _ = generate_data(formula, model_dict, r=0.7**0.5)
df = df.rename(columns=dict(y='eta'))
df['mu'] = LogitLink().inv_link(df['eta'])
df['y'] = sp.stats.binom(n=1, p=df['mu']).rvs()
