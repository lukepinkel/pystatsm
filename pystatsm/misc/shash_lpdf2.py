#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 22:59:38 2023

@author: lukepinkel
"""
import numpy as np
CONST = -1/2*np.log(np.pi) - 1/2*np.log(2)

def _logpdf2(y, m, s, t, v):  
    x0 = t*np.arcsinh((m - y)/s) + v
    lnp = np.log(t) - 1/2*np.log(m**2 - 2*m*y + s**2 + y**2) +\
          np.log(np.cosh(x0)) - 1/2*np.sinh(x0)**2 + CONST# -1/2*np.log(np.pi) - 1/2*np.log(2)
    
    x1 = s**2
    x2 = 2*y
    x3 = m**2 - m*x2 + x1 + y**2
    x4 = 1/x3
    x5 = 2*m - x2
    x6 = (1/2)*x5
    x7 = 1/s
    x8 = m - y
    x9 = np.arcsinh(x7*x8)
    x10 = t*x9
    x11 = v + x10
    x12 = np.sinh(x11)
    x13 = np.cosh(x11)
    x14 = x12*x13
    x15 = 1/x1
    x16 = x8**2
    x17 = x15*x16 + 1
    x18 = 1/np.sqrt(x17)
    x19 = x18*x7
    x20 = t*x19
    x21 = 1/x13
    x22 = x12*x21
    x23 = t*x18
    x24 = x15*x23
    x25 = x24*x8
    x26 = t**2
    x27 = 1/x17
    x28 = x3**(-2)
    x29 = x12**2
    x30 = x26*x27
    x31 = x15*x30
    x32 = x13**2
    x33 = 1/x32
    x34 = x29*x33
    x35 = s**(-3)
    x36 = x17**(-3/2)
    x37 = t*x36
    x38 = x22*x37
    x39 = x35*x8
    x40 = x30*x39
    x41 = s**(-4)
    x42 = x16*x41
    x43 = x37*x42
    x44 = x10*x19
    x45 = x30*x42
    x46 = s**(-5)
    x47 = x8**3
    x48 = x15*x18*x8
    x49 = x10*x48
    x50 = x9**2
    x51 = x29*x50
    x52 = x29*x9
    
    
    x53 = x29 + x32 + x34 - 1
    
    

      
    d1L = np.zeros(y.shape+(4,))
    d1L[...,0] = t*x12*x18*x21*x7 - x14*x20 - x4*x6
    d1L[...,1] = -s*x4 + t*x12*x13*x15*x18*x8 - x22*x25
    d1L[...,2] = (-t*x9*(x14 - x22) + 1)/t
    d1L[...,3] = x12*x21 - x14
  
  
  
  
  
    d2L = np.zeros(y.shape+(10,))
    d2L[...,0] = (1/2)*t*x12*x13*x35*x36*x5 + x15*x26*x27 + x28*x5*x6 - x29*x31 - x31*x32 - x31*x34 - x35*x38*x6 - x4
    d2L[...,1] = s*x28*x5 + x14*x24 - x14*x43 - x22*x24 + x22*x43 + x29*x40 + x32*x40 + x34*x40 - x40
    d2L[...,2] = t*x18*x7*x9 + x12*x18*x21*x7 - x14*x19 - x29*x44 - x32*x44 - x34*x44
    d2L[...,3] = t*x18*x7 - x20*x29 - x20*x32 - x20*x34
    d2L[...,4] = t*x12*x13*x36*x46*x47 + 2*t*x12*x18*x21*x35*x8 + 2*x1*x28 - 2*x14*x23*x39 + x16*x26*x27*x41 - x29*x45 - x32*x45 - x34*x45 - x38*x46*x47 - x4
    d2L[...,5] = x14*x48 - x22*x48 + x29*x49 + x32*x49 + x34*x49 - x49
    d2L[...,6] = x25*x53
    d2L[...,7] = -x32*x50 - x33*x51 + x50 - x51 - 1/x26
    d2L[...,8] = -x32*x9 - x33*x52 - x52 + x9
    d2L[...,9] = -x53
    return lnp, d1L, d2L
  
    
  
    
