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



                  
@numba.jit(nopython=True)
def _median_1d(arr, kind="low"):
    n = arr.size
    
    if kind == "low":
        k = (n+1)//2-1
    elif kind == "high":
        k = n//2
    
    m = np.partition(arr, k)[k]
    return m

    
def _median_nd(a, kind="low", axis=None, out=None, check_arr=True):
    a = np.asarray(a) if check_arr else a
    n = a.size if axis is None else a.shape[axis]
    
    if kind == "low":
        k = (n+1)//2-1
    elif kind == "high":
        k = n//2
    
    axis = 0 if axis is None else axis
    part = np.partition(a, [k], axis=axis)
    ix = [slice(None)] * part.ndim
    ix[axis] = slice(k, k+1)
    m = part[tuple(ix)]
    return m


def sn_naive(x, corrected=True):
    n = x.size
    a1 = np.zeros_like(x)
    a2 = np.zeros_like(x)
    
    for i in range(n):
        for j in range(n):
            a1[j] = np.abs(x[i] - x[j])
        a2[i] = _median_1d(a1, kind="high")
    s = 1.1926 * _median_1d(a2, kind="low")
    if corrected:
        if n>9:
            if n%2==1:
                cn = n / (n - 0.9)
            else:
                cn = 1.0
        else:
            cn = [0.743, 1.851, 0.954, 1.351, 0.993, 1.198, 1.005, 1.131][n-2]
        s = s * cn
    return s, a2
    

    
def mad(x, cn=None, axis=None):
    m = np.median(x, axis=axis)
    if axis is not None:
        m = np.expand_dims(m, axis)
    res = np.median(np.abs(x - m), axis=axis)
    cn = 1.0 / sp.special.ndtri(0.75) if cn is None else cn
    res = res * cn
    return res
    



@numba.jit(nopython=True)
def _sn_a(x, a, n, k1, k2):
    for i in range(2, k1+1):
        nA, nB = i-1, n-i
        ndiff = nB - nA
        lA = lB = 1
        rA = rB = nB
        Amin, Amax = ndiff // 2 + 1, ndiff // 2 + nA
        
        while lA<rA:
            t = rA - lA + 1
            p = 1 - t % 2
            h = (t - 1) // 2
            tA, tB = lA+h, lB+h
            if tA<Amin:
                rB = tB
                lA = tA + p
            else:
                if tA > Amax:
                    rA = tA
                    lB = tB + p
                else:
                    mA = x[i-1] - x[i - tA + Amin - 2]
                    mB = x[tB + i - 1] - x[i - 1]
                    if mA>=mB:
                        rA, lB = tA, tB + p
                    else:
                        rB, lA = tB, tA + p
        if lA > Amax:
            a[i-1] = x[lB + i - 1] - x[i - 1]
        else:
            mA = x[i - 1] - x[i - lA + Amin - 2]
            mB = x[lB + i - 1] - x[i - 1]
            a[i-1] = np.minimum(mA, mB)
    return a

@numba.jit(nopython=True)
def _sn_b(x, a, n, k1, k2):
    for i in range(k1+1, n):
        nA, nB = n - i, i - 1
        ndiff = nB - nA
        lA = lB = 1
        rA = rB = nB
        Amin, Amax = ndiff // 2 + 1, ndiff // 2 + nA
        
        while lA < rA:
            t = rA - lA + 1
            p = 1 - t % 2
            h = (t - 1) // 2
            tA, tB = lA + h, lB + h
            
            if tA < Amin:
                rB, lA = tB, tA + p
            else:
                if tA > Amax:
                    rA, lB = tA, tB + p
                else:
                    mA = x[i + tA - Amin] - x[i - 1]
                    mB = x[i - 1] - x[i - tB - 1]
                    if mA >= mB:
                        rA, lB = tA, tB + p
                    else:
                        rB, lA = tB, tA + p
        if lA > Amax:
            a[i-1] = x[i - 1] - x[i - lB - 1]
        else:
            mA = x[i + lA - Amin] - x[i - 1]
            mB = x[i - 1] - x[i - lB - 1]
            a[i-1] = np.minimum(mA, mB)
        
    return a


def sn(x, corrected=True):
    x = np.sort(x)
    a = np.zeros_like(x)
    
    n = x.size
    k1 = (n + 1) // 2
    k2 = n // 2
    
    
    a[0] = x[n//2] - x[0]
    
    a = _sn_a(x, a, n, k1, k2)
    a = _sn_b(x, a, n, k1, k2)
    
    a[n - 1] = x[n - 1] - x[k1 - 1]
    s = 1.1926 * np.partition(a, k1-1)[k1-1]
    if corrected:
        if n>9:
            if n%2==1:
                cn = n / (n - 0.9)
            else:
                cn = 1.0
        else:
            cn = [0.743, 1.851, 0.954, 1.351, 0.993, 1.198, 1.005, 1.131][n-2]
        s = s * cn
    return s, a
    











