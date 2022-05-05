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
                    for y in range(x+1):
                        d2L = d2_pivot(L, dL, d2L, k, x, y)
                        d2L[:, :, y, x] = d2L[:, :, x, y]
        #(b) Adjust Lead Column
        for j in range(k+1, n):
            L[j, k] = L[j, k] / L[k, k]
            if order > 0:
                for x in range(m):
                    dL = d1_col(L, dL, j, k, x)
                    if order > 1:
                        for y in range(x+1):
                            d2L = d2_col(L, dL, d2L, j, k, x, y)
                            d2L[:, :, y, x] = d2L[:, :, x, y]
        #(c) Row Operations
        for j in range(k+1, n):
            for i in range(j, n):
                L[i, j] = L[i, j] - L[i, k] * L[j, k]
                if order > 0:
                    for x in range(m):
                        dL = d1_row(L, dL, i, j, k, x)
                        if order > 1:
                            for y in range(x+1):
                                d2L = d2_row(L, dL, d2L, i, j, k, x, y)
                                d2L[:, :, y, x] = d2L[:, :, x, y]
    #TODO: Figure out why this correction is needed to equal numerical approx
    #for i in range(m):
    #    for j in range(m):
    #        if i<j:
    #            d2L[:, :, i, j] = d2L[:, :, j, i]
    return L, dL, d2L

def unit_matrices(n):
    m = int(n * (n + 1) / 2)
    dM  = np.zeros((n, n, m))
    d2M = np.zeros((n, n, m, m))
    j, i = np.triu_indices(n)
    k = np.arange(m)
    dM[i, j, k] = 1.0
    return dM, d2M



# from pystatsm.pystatsm.utilities.random import r_lkj
# from pystatsm.pystatsm.utilities.linalg_operations import (_vech as vech,
#                                                            _invech as invech)
# from pystatsm.pystatsm.utilities.numerical_derivs import jac_approx


# def chol_vech(x):
#     A = invech(x)
#     L = np.linalg.cholesky(A)
#     return L

# def dchol_vech(x):
#     M = invech(x)
#     n = M.shape[-1]
#     m = int(n * (n + 1) / 2)
#     dM  = np.zeros((n, n, m))
#     d2M = np.zeros((n, n, m, m))
#     j, i = np.triu_indices(n)
#     k = np.arange(m)
#     dM[i, j, k] = 1.0
#     _, dL, _ = dchol(M, dM, d2M, order=1)
#     return dL


# n = 5
# m = int(n * (n + 1) / 2)

# rng = np.random.default_rng(1234)
# R = r_lkj(eta=1.0, n=1, dim=n, rng=rng)[0, 0]
# V = np.diag(rng.uniform(low=0.5, high=2, size=n))
# M = V.dot(R).dot(V)
# dM  = np.zeros((n, n, m))

# d2M = np.zeros((n, n, m, m))
# j, i = np.triu_indices(n)
# k = np.arange(m)

# dM[i, j, k] = 1.0


# L, dL, d2L = dchol(M, dM, d2M)

# dLn = jac_approx(chol_vech, vech(M))
# d2Ln = jac_approx(dchol_vech, vech(M), d=1e-6)

# assert(np.allclose(L, np.linalg.cholesky(M)))
# assert(np.allclose(dL, dLn, rtol=1e-2, atol=1e-2))
# assert(np.allclose(d2L, d2Ln, rtol=1e-2, atol=1e-2))

#




    
