#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 23:46:06 2023

@author: lukepinkel
"""

import numpy as np
CONST = -1/2*np.log(np.pi) - 1/2*np.log(2)

def _logpdf1(y, m, s, t, v):  
    x0 = v + np.exp(t)*np.arcsinh((m - y)*np.exp(-s))
    lnp = t - 1/2*np.log(m**2 - 2*m*y + y**2 + np.exp(2*s)) +\
           np.log(np.cosh(x0)) - 1/2*np.sinh(x0)**2 + CONST# -1/2*np.log(np.pi) - 1/2*np.log(2)
                  
    x1 = 2*y
    x2 = 2*s
    x3 = np.exp(x2)
    x4 = 1/(m**2 - m*x1 + x3 + y**2)
    x5 = np.exp(t)
    x6 = np.exp(-s)
    x7 = m - y
    x8 = x6*x7
    x9 = x5*np.arcsinh(x8)
    x10 = v + x9
    x11 = np.sinh(x10)
    x12 = np.cosh(x10)
    x13 = x11*x12
    x14 = 1/np.sqrt(x7**2*np.exp(-x2) + 1)
    x15 = x14*x5
    x16 = 1/x12
    x17 = x11*x16
    d1L = np.zeros(y.shape+(4,))
    d1L[...,0] = -m*x4 + (1/2)*x1*x4 + x11*x14*x16*x5*x6 - x13*x15*x6
    d1L[...,1] = x11*x12*x14*x5*x6*x7 - x15*x17*x8 - x3*x4
    d1L[...,2] = -x13*x9 + x17*x9 + 1
    d1L[...,3] = x11*x16 - x13
    return lnp, d1L