# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 20:06:58 2021

@author: lukepinkel
"""

import numpy as np
import scipy as sp
from scipy.special import erf

RT2 = np.sqrt(2.0)

class SkewNorm:
    
    @staticmethod
    def loglike(y, mu, sigma, nu, tau):
        return _logpdf(y, mu, sigma, nu, tau)
    
    @staticmethod
    def d1loglike(y, mu, sigma, nu, tau):
        return _d1logpdf(y, mu, sigma, nu, tau)
    
    @staticmethod
    def d2loglike(y, mu, sigma, nu, tau):
        return _d2logpdf(y, mu, sigma, nu, tau)

    @staticmethod
    def d3loglike(y, mu, sigma, nu, tau):
        return _d3logpdf(y, mu, sigma, nu, tau)

    @staticmethod
    def d4loglike(y, mu, sigma, nu, tau):
        return _d4logpdf(y, mu, sigma, nu, tau)

    
def _expand_arrs(*arrs):
    y = np.concatenate([np.expand_dims(np.asarray(arr), -1) for arr in arrs], axis=-1)
    return y


def _d1logpdf(x, m, s, v):
    x0 = s**(-2)
    x1 = (1/2)*x0
    x2 = s**(-1)
    x3 = np.sqrt(2)
    x4 = v*x2*x3
    x5 = -m + x
    x6 = x5**2
    x7 = np.exp(-v**2*x1*x6)/(np.sqrt(np.pi)*(erf((1/2)*x4*x5) + 1))
    x8 = x3*x5*x7
    m1 = -x1*(2*m - 2*x) - x4*x7
    s1 = -v*x0*x8 - x2 + x6/s**3
    v1 = x2*x8
    g = _expand_arrs(m1, s1, v1)
    return g

def _d2logpdf(x, m, s, v):
    x0 = s**(-2)
    x1 = v**2
    x2 = np.sqrt(2)
    x3 = s**(-1)
    x4 = m - x
    x5 = x3*x4
    x6 = v*x5
    x7 = erf((1/2)*x2*x6) - 1
    x8 = x0*x4**2
    x9 = x1*x8
    x10 = np.exp(-x9)/np.pi
    x11 = 2.0*x10/x7**2
    x12 = x1*x11
    x13 = v**3
    x14 = x7**(-1)
    x15 = x2*np.exp(-1/2*x9)/np.sqrt(np.pi)
    x16 = 1.0*x15
    x17 = x14*x16
    x18 = x13*x17
    x19 = x14*x8
    x20 = x16*x9
    x21 = 2.0*x14
    x22 = x10*x21
    x23 = x22*x6
    mm = -x0*(x12 + x18*x5 + 1)
    ms = x0*(-v*x17 + x12*x5 + x13*x16*x19 + 2*x5)
    mv = x14*x3*(x16 - x20 - x23)
    ss = x0*(-x11*x9 + x15*x21*x6 - 3*x8 + 2 - x18*x4**3/s**3) - x0
    sv = x0*x14*x4*(-x16 + x20 + x23)
    vv = -x19*(x16*x6 + x22)
    g = _expand_arrs(mm, ms, mv, ss, sv ,vv)
    return g

def _d3logpdf(x, m, s, v):
    x0 = v**3
    x1 = np.sqrt(2)
    x2 = m - x
    x3 = x2/s
    x4 = v*x3
    x5 = x1*x4
    x6 = erf((1/2)*x5) - 1
    x7 = x6**(-2)
    x8 = np.pi**(-3/2)
    x9 = v**2
    x10 = s**(-2)
    x11 = x2**2
    x12 = x10*x11
    x13 = x12*x9
    x14 = np.exp(-3/2*x13)
    x15 = 4*x1*x14*x8
    x16 = x15*x7
    x17 = x1*np.exp(-1/2*x13)/np.sqrt(np.pi)
    x18 = x13*x17
    x19 = np.exp(-x13)/np.pi
    x20 = 6*x3
    x21 = x6**(-1)
    x22 = v*x21
    x23 = x20*x22
    x24 = -x17 + x19*x23
    x25 = s**(-3)
    x26 = x21*x25
    x27 = x26*(x16 + x18 + x24)
    x28 = 4*x19
    x29 = x7*x9
    x30 = v**4
    x31 = 6*x19
    x32 = x30*x31*x7
    x33 = x0*x3
    x34 = x15/x6**3
    x35 = 3*x17
    x36 = v**5
    x37 = x2**3
    x38 = x17*x37
    x39 = x26*x38
    x40 = x35*x4
    x41 = x0*x25
    x42 = x38*x41
    x43 = x21*x28
    x44 = 4*x14*x5*x7*x8
    x45 = x13*x21*x31
    x46 = x10*x21
    x47 = x46*(-x40 + x42 - x43 + x44 + x45)
    x48 = 2*x17
    x49 = 8*x19
    x50 = s**(-4)
    x51 = x2**4
    x52 = x17*x50*x51
    x53 = x21*x36
    x54 = x0*x12
    x55 = x30*x52
    x56 = x0*x26*x31*x37
    x57 = x13*x16
    mmm = x0*x27
    mms = x25*(-x12*x32 + x21*x33*x35 + x28*x29 - x33*x34 - x36*x39 + 2)
    mmv = v*x47
    mss = x25*(-5*x17*x21*x54 - x20 + x22*x48 + x25*x32*x37 - x29*x3*x49 + x34*x54 + x52*x53)
    msv = x46*(4*x18 + x24 - x55 - x56 - x57)
    mvv = x2*x47
    sss = x25*(7*x0*x39 + 12*x12 + 12*x13*x19*x7 - x17*x23 - x32*x50*x51 - x34*x37*x41 - 4 - x17*x2**5*x53/s**5)+2/s**3
    ssv = x2*x26*(-5*x18 - x21*x4*x49 + x48 + x55 + x56 + x57)
    svv = x11*x26*(x40 - x42 + x43 - x44 - x45)
    vvv = x27*x37
    g = _expand_arrs(mmm, mms, mmv, mss, msv , mvv, sss, ssv, svv, vvv)
    return g
    
def _d4logpdf(x, m, s, v):
    x0 = v**4
    x1 = np.sqrt(2)
    x2 = v**2
    x3 = s**(-2)
    x4 = m - x
    x5 = x4**2
    x6 = x3*x5
    x7 = x2*x6
    x8 = x1*np.exp(-1/2*x7)/np.sqrt(np.pi)
    x9 = 3*x8
    x10 = x4/s
    x11 = v*x10
    x12 = s**(-3)
    x13 = v**3
    x14 = x4**3
    x15 = x13*x14
    x16 = x12*x15
    x17 = x16*x8
    x18 = x1*x11
    x19 = erf((1/2)*x18) - 1
    x20 = x19**(-3)
    x21 = 24*x20
    x22 = np.exp(-2*x7)/np.pi**2
    x23 = x21*x22
    x24 = x19**(-1)
    x25 = np.exp(-x7)/np.pi
    x26 = x24*x25
    x27 = 8*x26
    x28 = 24*x10
    x29 = v*x28
    x30 = x19**(-2)
    x31 = np.pi**(-3/2)
    x32 = np.exp(-3/2*x7)
    x33 = x1*x31*x32
    x34 = x30*x33
    x35 = x29*x34
    x36 = x26*x7
    x37 = s**(-4)
    x38 = x24*x37
    x39 = x38*(x11*x9 - x17 - x23 + x27 - x35 - 14*x36)
    x40 = x4**4
    x41 = x37*x40
    x42 = x0*x41
    x43 = x42*x8
    x44 = 6*x8
    x45 = x44*x7
    x46 = 12*x30
    x47 = x33*x46
    x48 = x20*x22*x29
    x49 = x11*x26
    x50 = 26*x49
    x51 = 14*x25
    x52 = x12*x24
    x53 = x15*x52
    x54 = x51*x53
    x55 = 24*x34
    x56 = x55*x7
    x57 = x38*(x43 - x45 - x47 + x48 - x50 + x54 + x56 + x9)
    x58 = x52*(-x43 + x45 + x47 - x48 + x50 - x54 - x56 - x9)
    x59 = x2*x25
    x60 = x0*x6
    x61 = 24*x22/x19**4
    x62 = v**6
    x63 = x30*x51*x62
    x64 = x25*x30
    x65 = x20*x33
    x66 = x13*x65
    x67 = x13*x24*x8
    x68 = x12*x14
    x69 = v**5
    x70 = x21*x33
    x71 = x69*x70
    x72 = x4**5/s**5
    x73 = v**7*x24
    x74 = x73*x8
    x75 = x69*x8
    x76 = x11*x8
    x77 = x72*x75
    x78 = x18*x30*x31*x32
    x79 = x23*x7
    x80 = 14*x26
    x81 = x42*x80
    x82 = x16*x55
    x83 = x52*(-8*x17 + x27 - 38*x36 + 9*x76 + x77 - 20*x78 + x79 + x81 + x82)
    x84 = -x77 - x79 - x81 - x82
    x85 = v*x24
    x86 = 36*x30
    x87 = x0*x68
    x88 = s**(-6)
    x89 = x4**6
    x90 = x8*x88*x89
    x91 = x62*x90
    x92 = x7*x8
    x93 = x16*x23
    x94 = x69*x72
    x95 = x80*x94
    x96 = x25*x53
    x97 = x42*x55
    mmmm = x0*x39
    mmms = x13*x57
    mmmv = x2*x58
    mmss = x37*(-12*x10*x67 + 9*x14*x52*x75 + x28*x66 - x41*x63 - x46*x59 - x60*x61 + 44*x60*x64 - x68*x71 - x72*x74 - 6)
    mmsv = v*x83
    mmvv = x24*x3*(-x11*x44 + 7*x17 - 4*x26 + 32*x36 + 16*x78 + x84)
    msss = x37*(x10*x59*x86 + x28 - 12*x38*x40*x75 + x41*x71 - x44*x85 - 36*x6*x66 + 27*x6*x67 + x61*x87 + x63*x72 - 62*x64*x87 + x73*x90)
    mssv = x52*(28*x34*x7 + 10*x43 - 20*x49 + 2*x8 - x91 - 17*x92 - x93 - x95 + 50*x96 - x97)
    msvv = x4*x83
    mvvv = x5*x58
    ssss = x37*(48*x16*x65 + 15*x24*x77 + x28*x8*x85 - x42*x61 + 80*x42*x64 - 48*x53*x8 - 60*x6 - x63*x88*x89 - 72*x64*x7 - x70*x94 + 12 - x4**7*x74/s**7)-6/s**4
    sssv = x38*x4*(-x33*x7*x86 - 12*x43 - x44 + 36*x49 + x91 + 27*x92 + x93 + x95 - 62*x96 + x97)
    ssvv = x38*x5*(9*x17 - 12*x26 + x35 + 44*x36 - 12*x76 + x84)
    svvv = x14*x57
    vvvv = x39*x40
    g = _expand_arrs(mmmm, mmms, mmmv, mmss, mmsv, mmvv, msss, mssv, msvv, 
                     mvvv, ssss, sssv, ssvv, svvv, vvvv)
    return g


def _logpdf(x, m, s, v):
    z = (x - m) / s
    u = sp.special.log_ndtr(v * z) 
    l = u - np.log(s) + (np.log(2.0) - np.log(np.pi) - z**2) / 2
    return l

def _rvs(size=None, m=0, s=1, v=0, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    size = 1 if size is None else size
    z1 = rng.normal(0.0, 1.0, size)
    z2 = rng.normal(0.0, 1.0, size)
    t = v / np.sqrt(1.0 + v**2)
    x = t * z1 + z2 * np.sqrt(1.0 - t**2)
    x = np.where(z1>=0, x, -x)
    y = x * s + m
    return y

def _mean(m, s, v):
    t = np.sqrt(2.0 / ((1+v**2) * np.pi))
    mean = m + s * v * t
    return mean

def _variance(m, s, v):
    t = 2.0 * v**2 / ((1.0 + v**2) * np.pi)
    var = s**2 * (1.0 - t)
    return var

def _skewness(m, s, v):
    u = np.pi / 2.0 * (1 + 1/v**2) - 1.0
    skew = 0.5 * (4.0 - np.pi) * u**(-3.0/2.0) * np.sign(v)
    return skew

    