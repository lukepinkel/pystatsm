#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 13:34:49 2020

@author: lukepinkel
"""
import numpy as np
import scipy as sp
import scipy.special

def poisson_logp(x, mu, logp=True):
     p = sp.special.xlogy(x, mu) - sp.special.gammaln(x + 1) - mu
     if logp==False:
         p = np.exp(p)
     return p
 