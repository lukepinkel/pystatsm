#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 19:54:36 2020

@author: lukepinkel
"""

import numpy as np # analysis:ignore
import scipy as sp # analysis:ignore
import scipy.stats # analysis:ignore
import pandas as pd # analysis:ignore
import matplotlib as mpl # analysis:ignore
import matplotlib.pyplot as plt# analysis:ignore


def xcorr_brute(x, y):
    n = x.shape[0]
    rho = np.zeros(n)
    rho[0] = x.dot(y)
    for i in range(1, n):
        rho[i] = np.dot(x[i:], y[:-i])
    rho = np.concatenate([rho[1:][::-1], rho])
    return rho

def xcorr_fft(x, y):
    rho = sp.signal.fftconvolve(x, y[::-1], mode='full')
    return rho

def xcorr_fftc(x, y, normalization="coef", retlags=True):
    n = x.shape[0]
    rho = np.zeros(2*n-1)
    r = sp.signal.fftconvolve(x, y[::-1], mode='full')[n-1:]
    rho[n-1:] = r
    rho[:n-1] = r[1:][::-1]
    lags = np.arange(1-n, n)
    if normalization=='unbiased':
        c = 1.0 / (n - np.abs(lags))
    elif normalization=='biased':
        c = 1.0 / n
    elif normalization=='coef':
        c = 1.0 / np.sqrt(np.dot(x, x)*np.dot(y, y))
    rho*=c
    if retlags:
        return lags, rho
    else:
        return rho    

def xcorr_mat(x, y):
    n = x.shape[0]
    r = np.zeros(2*n-1)
    lags = list(range(1-n, n))
    for i in range(2*n-1):
        m = lags[i]
        if m>=0:
            for j in range(n-m):
                r[i]+=x[j+m]*y[j]
    r[:n-1] = r[n:][::-1]
    return r
        
     