#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 22:31:37 2020

@author: lukepinkel
"""


from .clm import CLM
import pandas as pd



data = pd.read_csv("/users/lukepinkel/Downloads/wine.csv", index_col=0)
data['temp'] = data['temp'].replace({'cold':0, 'warm':1})
data['contact'] = data['contact'].replace({'no':0, 'yes':1})

model = CLM("rating ~ temp+contact-1", data=data)
model.fit()
model.res



