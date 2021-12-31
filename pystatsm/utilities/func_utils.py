#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 13:34:49 2020

@author: lukepinkel
"""
import numpy as np
import scipy as sp
import scipy.special
SQRT2 = np.sqrt(2)
ROOT2PI = np.sqrt(2.0 * np.pi)

def poisson_logp(x, mu, logp=True):
     p = sp.special.xlogy(x, mu) - sp.special.gammaln(x + 1) - mu
     if logp==False:
         p = np.exp(p)
     return p
 
    
def log1p(x):
    return np.log(1+x)


def norm_cdf(x, mean=0.0, sd=1.0):
    z = (x - mean) / sd
    p = (sp.special.erf(z/SQRT2) + 1.0) / 2.0
    return p

def norm_pdf(x, mean=0.0, sd=1.0):
    z = (x - mean) / sd
    p = np.exp(-z**2 / 2.0) / (ROOT2PI * sd)
    return p