# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 18:51:20 2021

@author: lukepinkel
"""
import numba
import numpy as np
@numba.jit(nopython=True)
def _cr_mats(n, h):
    D = np.zeros((n-2, n))
    ldb = np.zeros(n-2)
    for i in range(n-2):
        D[i, i] = 1.0 / h[i]
        D[i, i+1] = -1.0 / h[i] - 1.0 / h[i+1]
        D[i, i+2] = 1.0 / h[i+1]
        ldb[i] = (h[i]+h[i+1]) / 3.0
    return D, ldb

@numba.jit(nopython=True)
def _cc_mats(h, n):
    B = np.zeros((n, n))
    D = np.zeros((n, n))
    B[0, 0] = (h[-1]+h[0]) / 3
    B[0, 1] = h[0] / 6
    B[0,-1] = h[-1] / 6
    D[0, 0] = -(1/h[0]+1/h[-1])
    D[0, 1] = 1 / h[0]
    D[0,-1] = 1 / h[-1]
    for i in range(1, n-1):
        B[i, i-1] = h[i-1] / 6
        B[i, i] = (h[i-1] + h[i]) / 3
        B[i, i+1] = h[i] / 6
        D[i, i-1] = 1 / h[i-1]
        D[i, i] = -(1 / h[i-1] + 1 / h[i])
        D[i, i+1] = 1 / h[i]
    B[-1, -2] = h[-2] / 6
    B[-1, -1] = (h[-2] + h[-1]) / 3
    B[-1, 0] = h[-1] / 6
    D[-1, -2] = 1 / h[-2]
    D[-1, -1] = -(1 / h[-2] + 1 / h[-1])
    D[-1, 0] = 1 / h[-1]
    return D, B


