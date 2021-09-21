#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 16 21:47:11 2020

@author: lukepinkel
"""

import numba # analysis:ignore
import numpy as np # analysis:ignore
import scipy as sp # analysis:ignore
import scipy.sparse as sps # analysis:ignore
from sksparse.cholmod import cholesky



def _check_shape(x, ndims=1):
    order = None
    
    if x.flags['C_CONTIGUOUS']:
        order = 'C'
    
    elif x.flags['F_CONTIGUOUS']:
        order = 'F'
    
    if x.ndim>ndims:
        x = x.reshape(x.shape[:-1], order=order)
    elif x.ndim<ndims:
        x = np.expand_dims(x, axis=-1)
    
    return x

@numba.jit(nopython=False)
def _check_shape_nb(x, ndims=1):
    if x.ndim>ndims:
        y = x.reshape(x.shape[:-1])
    elif x.ndim<ndims:
        y = np.expand_dims(x, axis=-1)
    elif x.ndim==ndims:
        y = x
    return y


def _check_np(x):
    if type(x) is not np.ndarray:
        x = x.values
    return x

def add_chol_row(A, L, i):
    x, r = A[i, :i], A[i, i]
    if i>0:
        b = sp.linalg.solve_triangular(L[:i, :i], x, lower=True, check_finite=False)
        L[i, :i] = b
    else:
        b = L[i, :i]
    s = np.dot(b.T, b)
    L[i, i] = np.sqrt(r-s)
    return L
    

@numba.jit(nopython=True)
def chol_downdate(L, k):
    n = L.shape[0]
    L1 = L.copy()[np.arange(n)!=k]
    for t in range(k, n-1):
        a, b = L1[t, t], L1[t, t+1]
        v = np.sqrt(a**2+b**2)
        c, s = a / v, b / v
        for i in range(t, n-1):
            Lit = L1[i, t]
            Lit1 = L1[i, t+1]
            L1[i, t] = c * Lit + s * Lit1
            L1[i, t+1] = c * Lit1 - s * Lit
        L1[t, t] = v
        L1[t, t+1] = 0.0
    L1 = L1[:, :-1]
    return L1


@numba.jit(nopython=True)
def toeplitz_cholesky_lower_nb(n, A):
    g = np.zeros((2, n), dtype=np.double)
    for j in range(0, n):
        g[0, j] = A[j, 0]
    for j in range(1, n):
        g[1, j] = A[j, 0]
    L = np.zeros((n, n), dtype=np.double)
    for j in range(0, n):
        L[j, 0] = g[0, j]
    for j in range(n-1, 0, -1):
        g[0, j] = g[0, j-1]
    g[0, 0] = 0.0
    for i in range(1, n):
        rho = -g[1, i] / g[0, i]
        gamma = np.sqrt((1.0 - rho) * (1.0 + rho))
        for j in range(i, n):
            alpha = g[0, j]
            beta = g[1, j]
            g[0, j] = (alpha + rho * beta) / gamma
            g[1, j] = (rho * alpha + beta) / gamma
        for j in range(i, n):
            L[j, i] = g[0, j]
        for j in range(n-1, i, -1):
            g[0, j] = g[0, j-1]
        g[0, i] = 0.0
    return L


def vec(X):
    return X.reshape(-1, order='F')


def invec(x, n_rows, n_cols):
    return x.reshape(int(n_rows), int(n_cols), order='F')


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
    '''
    Inverse half vectorization operator
    '''
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

@numba.jit(nopython=True)
def vecl(X):
    p = X.shape[0]
    tmp =  1 - np.tri(p, p)
    tmp2 = tmp.flatten()
    ix = tmp2==1
    Y = X.T.flatten()[ix]
    return Y

@numba.jit(nopython=True)
def invecl(v):
    rows = int(np.round(.5 * (1 + np.sqrt(1 + 8 * len(v)))))
    res = np.zeros((rows, rows))
    tmp =  1 - np.tri(rows, rows)
    tmp2 = tmp.flatten()
    ix = tmp2==1
    Y = res.T.flatten()
    Y[ix] = v
    Y = Y.reshape(rows, rows)
    Y = Y + Y.T
    Ir = np.eye(rows)
    Y = Y - (Ir * Y) + Ir
    return Y


def invech_chol(lvec):
    p = int(0.5 * ((8*len(lvec) + 1)**0.5 - 1))
    L = np.zeros((p, p))
    a, b = np.triu_indices(p)
    L[(b, a)] = lvec
    return L

@numba.jit(nopython=True)
def _dummy(x, fullrank=True, categories=None):
    if categories is None:
        categories = np.unique(x)
    p = len(categories)
    if fullrank is False:
        p = p - 1
    n = x.shape[0]
    Y = np.zeros((n, p))
    for i in range(p):
        Y[x==categories[i], i] = 1.0
    return Y

def dummy(x, fullrank=True, categories=None):
    x = _check_shape(_check_np(x))
    return _dummy(x, fullrank, categories)


def vdg(X):
    V = np.diag(vec(X))
    return V

     
def gb_diag(*arrs):
    shapes = np.array([arr.shape for arr in arrs])
    res = np.zeros(np.sum(shapes, axis=0))
    ix = np.zeros(len(res.shape), dtype=int)
    for i, k in enumerate(shapes):
        s = [slice(a, a+b) for a, b in list(zip(ix, k))]
        res[tuple(s)] = arrs[i]
        ix += k
    return res
  