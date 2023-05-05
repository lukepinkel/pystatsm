#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 23:44:41 2023

@author: lukepinkel
"""

import numpy as np
CONST = -1/2*np.log(np.pi) - 1/2*np.log(2)

def _logpdf2(y, m, s, t, v):  
    x0 = v + np.exp(t)*np.arcsinh((m - y)*np.exp(-s))
    lnp = t - 1/2*np.log(m**2 - 2*m*y + y**2 + np.exp(2*s)) +\
           np.log(np.cosh(x0)) - 1/2*np.sinh(x0)**2 + CONST# -1/2*np.log(np.pi) - 1/2*np.log(2)
                   
    x1 = 2*y
    x2 = 2*s
    x3 = np.exp(x2)
    x4 = m**2 - m*x1 + x3 + y**2
    x5 = 1/x4
    x6 = 2*m - x1
    x7 = (1/2)*x6
    x8 = np.exp(t)
    x9 = np.exp(-s)
    x10 = m - y
    x11 = x10*x9
    x12 = np.arcsinh(x11)
    x13 = x12*x8
    x14 = v + x13
    x15 = np.sinh(x14)
    x16 = np.cosh(x14)
    x17 = x15*x16
    x18 = np.exp(-x2)
    x19 = x10**2
    x20 = x18*x19
    x21 = x20 + 1
    x22 = 1/np.sqrt(x21)
    x23 = x22*x9
    x24 = x23*x8
    x25 = 1/x16
    x26 = x15*x25
    x27 = x17*x24 - x24*x26
    x28 = x3*x5
    x29 = x11*x22
    x30 = x29*x8
    x31 = x17*x30
    x32 = x26*x30
    x33 = x13*x17
    x34 = np.exp(2*t)
    x35 = 1/x21
    x36 = x4**(-2)
    x37 = x15**2
    x38 = x34*x35
    x39 = x18*x38
    x40 = x37*x39
    x41 = x16**2
    x42 = x39*x41
    x43 = x37/x41
    x44 = np.exp(-3*s)
    x45 = x21**(-3/2)
    x46 = x44*x45*x8
    x47 = x10*x39
    x48 = x19*x46
    x49 = x12*x34
    x50 = x23*x49
    x51 = x20*x38
    x52 = x10**3
    x53 = x31 - x32
    x54 = x29*x49
    x55 = x12**2
    x56 = x34*x55
    x57 = x26*x46
    x58 = x37 + x41 + x43 - 1
    d1L = np.zeros(y.shape+(4,))
    d1L[...,0] = -x27 - x5*x7
    d1L[...,1] = -x28 + x31 - x32
    d1L[...,2] = x13*x26 - x33 + 1
    d1L[...,3] = x15*x25 - x17
    
    
    
    
    
    d2L = np.zeros(y.shape+(10,))
    d2L[...,0] = (1/2)*x15*x16*x44*x45*x6*x8 + x18*x34*x35 + x36*x6*x7 - x39*x43 - x40 - x42 - x5 - x57*x7
    d2L[...,1] = x10*x40 + x10*x42 - x17*x48 + x26*x48 + x27 + x3*x36*x6 + x43*x47 - x47
    d2L[...,2] = x12*x22*x34*x9 - x27 - x37*x50 - x41*x50 - x43*x50
    d2L[...,3] = x22*x8*x9 - x24*x37 - x24*x41 - x24*x43
    d2L[...,4] = x15*x16*x44*x45*x52*x8 + x18*x19*x34*x35 - 2*x28 + 2*x36*np.exp(4*s) - x37*x51 - x41*x51 - x43*x51 - x52*x57 - x53
    d2L[...,5] = x37*x54 + x41*x54 + x43*x54 + x53 - x54
    d2L[...,6] = x30*x58
    d2L[...,7] = x12*x15*x25*x8 - x33 + x34*x55 - x37*x56 - x41*x56 - x43*x56
    d2L[...,8] = x12*x8 - x13*x37 - x13*x41 - x13*x43
    d2L[...,9] = -x58
    
    return lnp, d1L, d2L
    
    
    
    
    
    
    