#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  5 06:13:28 2022

@author: lukepinkel
"""

import numba
import numpy as np



@numba.jit(nopython=True)
def d1_pivot(L, dL, k, x):
    dL[k, k, x] = 0.5 * dL[k, k, x] / L[k, k]
    return dL

@numba.jit(nopython=True)
def d2_pivot(L, dL, d2L, k, x, y):
    d2L[k, k, x, y] = (0.5 * d2L[k, k, x, y] - dL[k, k, x] * dL[k, k, y]) / L[k, k]
    return d2L

@numba.jit(nopython=True)
def d1_col(L, dL, j, k, x):
    dL[j, k, x] = (dL[j, k, x] - L[j, k] * dL[k, k, x]) / L[k, k]
    return dL


@numba.jit(nopython=True)
def d2_col(L, dL, d2L, j, k, x, y):
    t1 = d2L[j, k, x, y]
    t2 = L[j, k] * d2L[k, k, x, y]
    t3 = dL[j, k, x] * dL[k, k, y]
    t4 = dL[j, k, y] * dL[k, k, x]
    d2L[j, k, x, y] = (t1 - t2 - t3 - t4) / L[k, k]
    return d2L


@numba.jit(nopython=True)
def d1_row(L, dL, i, j, k, x):
    dL[i, j, x] = dL[i, j, x] - dL[i, k, x] * L[j, k] - L[i, k] * dL[j, k, x]
    return dL


@numba.jit(nopython=True)
def d2_row(L, dL, d2L, i, j, k, x, y):
    t1 = d2L[i, j, x, y]
    t2 = d2L[i, k, x, y] * L[j, k]
    t3 = L[i, k] * d2L[j, k, x, y]
    t4 = dL[i, k, x] * dL[j, k, y]
    t5 = dL[i, k, y] * dL[j, k, x]
    d2L[i, j, x, y] = t1 - t2 - t3 - t4 - t5
    return d2L


@numba.jit(nopython=True)
def dchol(M, dM, d2M, order=2):
    L   = np.tril(M)
    dMt = np.swapaxes(np.swapaxes(dM, 0, 2), 1, 2)
    dLt  = np.tril(dMt)
    dL = np.swapaxes(np.swapaxes(dLt, 1, 2), 0, 2)
    d2L = np.tril(d2M)
    n = L.shape[-1]
    m = dM.shape[-1]
    
    for k in range(n):
        #(a) Define Pivot
        L[k, k] = np.sqrt(L[k, k])
        if order > 0: 
            for x in range(m):
                dL = d1_pivot(L, dL, k, x)
                if order > 1:
                    for y in range(m):
                        d2L = d2_pivot(L, dL, d2L, k, x, y)
        #(b) Adjust Lead Column
        for j in range(k+1, n):
            L[j, k] = L[j, k] / L[k, k]
            if order > 0:
                for x in range(m):
                    dL = d1_col(L, dL, j, k, x)
                    if order > 1:
                        for y in range(m):
                            d2L = d2_col(L, dL, d2L, j, k, x, y)
        #(c) Row Operations
        for j in range(k+1, n):
            for i in range(j, n):
                L[i, j] = L[i, j] - L[i, k] * L[j, k]
                if order > 0:
                    for x in range(m):
                        dL = d1_row(L, dL, i, j, k, x)
                        if order > 1:
                            for y in range(m):
                                d2L = d2_row(L, dL, d2L, i, j, k, x, y)
    #TODO: Figure out why this correction is needed to equal numerical approx
    for i in range(m):
        for j in range(m):
            if i<j:
                d2L[:, :, i, j] = d2L[:, :, j, i]
    return L, dL, d2L


