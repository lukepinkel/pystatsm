#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 22:53:58 2020

@author: lukepinkel
"""

import numba # analysis:ignore
import numpy as np # analysis:ignore
import scipy as sp # analysis:ignore
import scipy.stats # analysis:ignore
from math import erf

SQRT2 = np.sqrt(2.0)

@numba.jit(nopython=True)
def norm_cdf(x):
    y = (1.0 + erf(x / SQRT2)) / 2.0
    return y
    

#Adapted from MCMCglmm
@numba.jit(nopython=True)
def scalar_truncnorm(mu, sd, lb, ub):
    sample = 1
    if (lb < -1e16) or (ub>1e16):
        if (lb < -1e16) and (ub>1e16):
            z = np.random.normal(mu, sd)
        else:
            if ub > 1e16:
                tr = (lb - mu) / sd
            else:
                tr = (mu - ub) / sd
            if tr < 0:
                while sample==1:
                    z = np.random.normal(0.0, 1.0)
                    if z>tr:
                        sample = 0
            else:
                alpha = (tr + np.sqrt((tr * tr) + 4.0)) / 2.0
                while sample==1:
                    z = np.random.exponential(scale=1/alpha) + tr
                    pz = -((alpha - z) * (alpha - z) / 2.0)
                    u = -np.random.exponential(scale=1.0)
                    if (u<=pz):
                        sample = 0
    else:
        sl = (lb - mu) / sd
        su = (ub - mu) / sd
    
        tr = norm_cdf(su) - norm_cdf(sl)
        
        if tr>0.5:
            while sample==1:
                z = np.random.normal(0.0, 1.0)
                if (z>sl) and (z<su):
                    sample = 0
        else:
            while sample==1:
                z = np.random.uniform(sl, su)
                if (sl<=0.0) and (0.0<=su):
                    pz = -z*z / 2.0
                else:
                    if su<0.0:
                        pz = (su * su - z * z) / 2.0
                    else:
                        pz = (sl * sl - z * z) / 2.0
                u = -np.random.exponential(scale=1.0)
                if u<pz:
                    sample = 0
    if lb<-1e16:
        return mu-z*sd
    else:
        return z*sd+mu
            
@numba.jit(nopython=True)      
def trnorm(mu, sd, lb, ub):
    n = len(mu)
    z = np.zeros((n, ), dtype=numba.float32)
    for i in range(n):
        z[i] = scalar_truncnorm(mu[i], sd[i], lb[i], ub[i])
    return z

    
    
       
        
            
    
    