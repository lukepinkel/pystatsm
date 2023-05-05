#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 23:01:52 2023

@author: lukepinkel
"""


import numpy as np
CONST = -1/2*np.log(np.pi) - 1/2*np.log(2)

def _logpdf1(y, m, s, t, v):  
    x0 = t*np.arcsinh((m - y)/s) + v
    lnp = np.log(t) - 1/2*np.log(m**2 - 2*m*y + s**2 + y**2) +\
          np.log(np.cosh(x0)) - 1/2*np.sinh(x0)**2 + CONST# -1/2*np.log(np.pi) - 1/2*np.log(2)
    
    x1 = 2*y
    x2 = s**2
    x3 = 1/(m**2 - m*x1 + x2 + y**2)
    x4 = 1/s
    x5 = m - y
    x6 = np.arcsinh(x4*x5)
    x7 = t*x6 + v
    x8 = np.sinh(x7)
    x9 = np.cosh(x7)
    x10 = x8*x9
    x11 = 1/x2
    x12 = 1/np.sqrt(x11*x5**2 + 1)
    x13 = t*x12
    x14 = 1/x9
    x15 = x14*x8
    d1L = np.zeros(y.shape+(4,))
    d1L[...,0] = -m*x3 + t*x12*x14*x4*x8 + (1/2)*x1*x3 - x10*x13*x4
    d1L[...,1] = -s*x3 + t*x11*x12*x5*x8*x9 - x11*x13*x15*x5
    d1L[...,2] = (-t*x6*(x10 - x15) + 1)/t
    d1L[...,3] = -x10 + x14*x8
    return lnp, d1L
