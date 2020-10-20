#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 22:56:30 2020

@author: lukepinkel
"""
from .glm import GLM, Binomial
import statsmodels.api as sm
spector_data = sm.datasets.spector.load_pandas()
spector_data.exog = sm.add_constant(spector_data.exog)

df = spector_data.data

model = GLM("GRADE~1+GPA+TUCE+PSI", data=df, fam=Binomial)
model.fit()