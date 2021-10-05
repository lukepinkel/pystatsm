#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 17 21:59:56 2020

@author: lukepinkel
"""

import numba # analysis:ignore
import numpy as np # analysis:ignore
import scipy as sp # analysis:ignore
import scipy.stats # analysis:ignore

QTF75 = sp.stats.norm(0, 1).ppf(0.75)
PHI75 = 1.1926 #np.sqrt(sp.stats.ncx2(df=1, nc=QTF75).median())
PHI58 = 1.0 / (sp.stats.norm(0, 1).ppf(5/8)*np.sqrt(2))


@numba.jit(nopython=True)
def vech(X):
    p = X.shape[0]
    tmp =  1 - np.tri(p, p, k=-1)
    tmp2 = tmp.flatten()
    ix = tmp2==1
    Y = X.T.flatten()[ix]
    return Y

@numba.jit(nopython=True)
def invech(v):
    rows = int(np.round(.5 * (-1 + np.sqrt(1 + 8 * len(v)))))
    res = np.zeros((rows, rows))
    tmp =  1 - np.tri(rows, rows, k=-1)
    tmp2 = tmp.flatten()
    ix = tmp2==1
    Y = res.T.flatten()
    Y[ix] = v
    Y = Y.reshape(rows, rows)
    Y = Y + Y.T
    Y = Y - (np.eye(rows) * Y) / 2
    return Y


def mad(x, corrected=True, *args):
    scale = np.median(np.abs(x - np.median(x, axis=0)))
    if corrected:
        scale = scale / QTF75
    return scale
    

def gmd(x, corrected=True, *args):
    n = len(x)
    a, b = np.tril_indices(n, -1)
    m = np.sum(np.fabs(x[a] - x[b]))
    d = len(a)
    m = m / d
    if corrected:
        m = m * np.sqrt(np.pi) / 2
    return m
   


def gmd2(x, corrected=True, *args):
    n = len(x)
    m = np.fabs(np.subtract.outer(x, x))
    m = np.sum(vech(m))
    d = (n * (n - 1)) / 2
    m = m / d
    if corrected:
        m = m * np.sqrt(np.pi) / 2
    return m
   

@numba.jit(nopython=True)
def _gmd3(x, n, m, d):
    for i in range(1, n):
        for j in range(i):
            m = m + np.fabs(x[i] - x[j])
    m = m / d
    return m
    
def gmd3(x, corrected=True):
    if x.ndim==1:
        x = x.reshape((len(x), -1))
    n = len(x)
    m = np.zeros((x.shape[1],))
    d = (n * (n - 1)) / 2
    m = _gmd3(x, n, m, d)
    if corrected:
        m = m * np.sqrt(np.pi) / 2
    return m



def sn_estimator(x, corrected=True, *args):
    diffs = np.fabs(np.subtract.outer(x, x))
    m = np.median(np.median(diffs, axis=1))
    if corrected:
        m = m*PHI75
    return m
    
    
def sn_estimator2(x, corrected=True):
    n = len(x)
    a, b = np.triu_indices(n)
    diffs = np.fabs(x[a] - x[b])
    m = np.median(np.median(invech(diffs), axis=1))
    if corrected:
        m = m*PHI75
    return m

def qn_estimator(x, corrected=True, *args):
    n = len(x)
    a, b = np.tril_indices(n, -1)
    m = np.fabs(x[a] - x[b])
    h = np.ceil(n / 2) + 1
    k = int((h * (h - 1)) / 2)
    m = m[np.argsort(m)[k]]
    if corrected:
        m = m*PHI58
    return m


def qn_estimator2(x, corrected=True):
    n = len(x)
    m = vech(np.fabs(np.subtract.outer(x, x)))
    h = np.ceil(n / 2) + 1
    k = int((h * (h - 1)) / 2)
    m = m[np.argsort(m)[k]]
    if corrected:
        m = m*PHI58
    return m



                  










