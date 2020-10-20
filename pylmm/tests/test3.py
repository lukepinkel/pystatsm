#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 31 10:56:19 2020

@author: lukepinkel
"""

import agq_data  
from ..pylmm.glmm import GLMM, GLMM_AGQ 
from ..pylmm.families import Binomial


formula, data = agq_data.formula, agq_data.df

model = GLMM_AGQ(formula, data, Binomial)
model.fit()


model2 = GLMM(formula, data, fam=Binomial())
model2.fit()
