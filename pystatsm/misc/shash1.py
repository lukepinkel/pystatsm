#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 23:59:49 2023

@author: lukepinkel
"""
import numpy as np
import scipy as sp
import scipy.special
from ..utilities.indexing_utils import multiset_permutations, ascending_indices_forward, ascending_indices_reversed
from ..utilities.ordered_indices import ascending_indices, descending_indices
from .shash_lpdf4 import _logpdf4
from .shash_lpdf3 import _logpdf3
from .shash_lpdf2 import _logpdf2
from .shash_lpdf1 import _logpdf1

CONST = -1/2*np.log(np.pi) - 1/2*np.log(2)

def P(q):
    return (np.exp(0.25) / np.sqrt(8 * np.pi)) * (sp.special.kv((q + 1) / 2, 0.25) + sp.special.kv((q - 1) / 2, 0.25))

def mean(m, s, v, t):
    return m + s * np.sinh(v / t) * P(1 / t)

def median(m, s, v, t):
    return m + (s / 2) * (np.exp(v / t) - np.exp(-v / t))

def variance(m, s, v, t):
    E_Z = np.sinh(v / t) * P(1 / t)
    P_2_t = P(2 / t)
    return (s**2 / 2) * (np.cosh(2 * v / t) * P_2_t - 1) - s**2 * E_Z**2

def skewness(m, s, v, t):
    E_Z = np.sinh(v / t) * P(1 / t)
    Var_Z = (np.cosh(2 * v / t) * P(2 / t) - 1) / 2
    E_Z_3 = (1 / 4) * (np.sinh(3 * v / t) * P(3 / t) - 3 * np.sinh(v / t) * P(1 / t))
    mu_3_Y = s**3 * (E_Z_3 - 3 * Var_Z * E_Z - E_Z**3)
    return mu_3_Y / variance(m, s, v, t)**1.5

def excess_kurtosis(m, s, v, t):
    E_Z = np.sinh(v / t) * P(1 / t)
    Var_Z = (np.cosh(2 * v / t) * P(2 / t) - 1) / 2
    E_Z_4 = (1 / 8) * (np.cosh(4 * v / t) * P(4 / t) - 4 * np.cosh(2 * v / t) * P(2 / t) + 3)
    mu_4_Y = s**4 * (E_Z_4 - 4 * E_Z**3 + 6 * Var_Z * E_Z**2 + 3 * E_Z**4)
    return mu_4_Y / variance(m, s, v, t)**2 - 3

def pdf(y, m, s, v, t):
    z = (y - m) / s
    r = np.sinh(t * np.arcsinh(z) - v)
    c = np.cosh(t * np.arcsinh(z) - v)
    return (t * c) / (s * np.sqrt(2 * np.pi * (1 + z**2))) * np.exp(-r**2 / 2)

def cdf(y, m, s, v, t):
    z = (y - m) / s
    r = np.sinh(t * np.arcsinh(z) - v)
    return sp.special.ndtr(r)

def inverse_cdf(p, m, s, v, t):
    return m + s * np.sinh(v / t + np.arcsinh(sp.special.ndtri(p)) / t)

def rvs(m, s, v, t, size=1, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    p = rng.uniform(low=0, high=1, size=size)
    y = inverse_cdf(p, m, s, v, t)
    return y

def moments(m, s, v, t):
    m1 = np.sinh(v/t) * P(1 / t)
    m2 = (np.cosh(2 * v / t) * P(2 / t) - 1) / 2
    m3 = (np.sinh(3 * v / t) * P(3 / t) - 3 * np.sinh(v / t) * P(1 / t))/ 4
    m4 = (np.cosh(4 * v / t) * P(4 / t) - 4 * np.cosh(2 * v / t) * P(2 / t) + 3) / 8
    
    my1 = m + m1 * s
    my2 = m2 * s**2 + 2 * m1 * m * s + m**2
    my3 = m3 * s**3 + 3 * m2 * m * s**2 + 3 * m1 * m**2 * s + m**3
    my4 = m4 * s**4 + 4 * m3 * m * s**3 + 6 * m2 * m**2 * s**2 + 4 * m1 * m**3 * s + m**4
    return my1, my2, my3, my4

def skew(m, s, v, t):
    m1 = np.sinh(v/t) * P(1 / t)
    m2 = (np.cosh(2 * v / t) * P(2 / t) - 1) / 2
    m3 = (np.sinh(3 * v / t) * P(3 / t) - 3 * np.sinh(v / t) * P(1 / t))/ 4
    
    my1 = m + m1 * s
    my2 = m2 * s**2 + 2 * m1 * m * s + m**2
    my3 = m3 * s**3 + 3 * m2 * m * s**2 + 3 * m1 * m**2 * s + m**3
    
    sigma = np.sqrt(my2-my1**2)
    
    return (my3-3*my1*sigma**2-my1**3)/sigma**3

def kurt(m, s, v, t):
    m1 = np.sinh(v/t) * P(1 / t)
    m2 = (np.cosh(2 * v / t) * P(2 / t) - 1) / 2
    m3 = (np.sinh(3 * v / t) * P(3 / t) - 3 * np.sinh(v / t) * P(1 / t))/ 4
    m4 = (np.cosh(4 * v / t) * P(4 / t) - 4 * np.cosh(2 * v / t) * P(2 / t) + 3) / 8
    
    my1 = m + m1 * s
    my2 = m2 * s**2 + 2 * m1 * m * s + m**2
    my3 = m3 * s**3 + 3 * m2 * m * s**2 + 3 * m1 * m**2 * s + m**3
    my4 = m4 * s**4 + 4 * m3 * m * s**3 + 6 * m2 * m**2 * s**2 + 4 * m1 * m**3 * s + m**4
    sigma = np.sqrt(my2-my1**2)
    return (my4-4*my3*my1+6*my2*my1**2-3*my1**4) / sigma**4
    

def logpdf(y, params=None, m=None, s=None, t=None, v=None, derivs=0):
    y = np.asarray(y)
    if params is not None:
        if isinstance(params, (list, tuple, np.ndarray)) and len(params) == 4:
            m, s, t, v = params
        else:
            raise ValueError("Invalid 'params'. It should be a list, tuple, or numpy array of length 4.")
    elif None in (m, s, t, v):
        raise ValueError("Either pass all individual parameters (m, s, t, v) or 'params' containing all 4 values.")
    
    if derivs==0:
        x0 = t*np.arcsinh((m - y)/s) + v
        lnp = np.log(t) - 1/2*np.log(m**2 - 2*m*y + s**2 + y**2) +\
              np.log(np.cosh(x0)) - 1/2*np.sinh(x0)**2 + CONST# -1/2*np.log(np.pi) - 1/2*np.log(2)
        return lnp
    elif derivs==1:
        return _logpdf1(y, m, s, t, v)
    elif derivs==2:
        return _logpdf2(y, m, s, t, v)
    elif derivs==3:
        return _logpdf3(y, m, s, t, v)
    elif derivs==4:
        return _logpdf4(y, m, s, t, v)
