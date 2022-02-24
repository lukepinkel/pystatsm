#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 16 21:47:11 2020

@author: lukepinkel
"""

import numba 
import numpy as np 
import scipy as sp 
from sksparse.cholmod import cholesky

def wcrossp(X, w):
    Y =  (X * w[:, np.newaxis]).T.dot(X)
    return Y

def wdcrossp(X, w, Y):
    Y =  (X * w[:, np.newaxis]).T.dot(Y)
    return Y


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
    ix = np.arange(n)!=k
    L1 = L.copy()[ix]
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
    #L1 = L1[:, :-1]
    #return L1
    L[:-1, :-1] = L1[:, :-1]
    return L


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


@numba.jit(nopython=True)
def ar1_chol(n, rho):
    L = np.zeros((n, n))
    L[:, 0] = rho**np.arange(n)
    L[1:, 1] = L[:-1, 0] * np.sqrt(1.0 - rho**2)
    for i in range(2, n):
        L[i:, i] = L[1:1-i, 1]
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


def vecl_inds(n):
    i, j = np.indices((n, n))
    i, j = i.flatten(), j.flatten()
    ix = j>i
    return ix


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


def wls_chol(X, y, w):
    Xwt = X.T * w
    L = np.linalg.cholesky(Xwt.dot(X))
    c = sp.linalg.solve_triangular(L, Xwt.dot(y), lower=True)
    b = sp.linalg.solve_triangular(L.T, c, lower=False)
    return b

def wls_qr(X, y, w):
    wsqr = np.sqrt(w)
    Q, R = np.linalg.qr(X * wsqr[:, None])
    c = Q.T.dot(wsqr * y)
    b = sp.linalg.solve_triangular(R, c, lower=False)
    return b

def nwls(X, y, w):
    neg = w < 0
    w_neg = np.zeros_like(w)
    w_neg[neg] = -w[neg]
    w_sqr = np.sqrt(np.abs(w))
    
    Q, R = np.linalg.qr((X * w_sqr[:, None]))
    non_pos = (w <= 0)
    U, s, Vt = np.linalg.svd(Q * non_pos[:, None], full_matrices=False)
    neg = neg * -1.0 + (~neg) * 1.0
    Qtz = Q.T.dot(w_sqr * neg * y)
    a = 1.0 / (1.0  - 2.0 * s**2)
    c = Vt.T.dot(a * Vt.dot(Qtz))
    b = sp.linalg.solve_triangular(R, c, lower=False)
    return b

def lsqr(X, y):
    Q, R = np.linalg.qr(X, mode='reduced')
    b = np.linalg.solve(R, Q.T.dot(y))
    return b

def diag_outer_prod(A, B):
    v = np.einsum("ij,ij->i", A, B, optimize=True)
    return v

def wdiag_outer_prod(X, W, Y):
    v = np.einsum("ij,jk,ik->i", X, W, Y, optimize=True)
    return v
    



@numba.jit(nopython=True)
def _solve_vander(a, x, n):
    for k in range(n-1):
        x[k+1 : n] -= a[k] * x[k : n-1]
    for k in range(n-1, 0, -1):
        x[k:n] /= a[k:n] - a[:n-k]
        x[k-1 : n-1] -= x[k:n]
    return x

def solve_vander(a, b):
    x = b.copy()
    n = a.size
    return _solve_vander(a, x, n)

@numba.jit(nopython=True)
def _inv_vander(a):
    n = len(a)
    B = np.zeros((n, n))
    for i in range(n):
        B[i, n - i - 1] = 1.0
        _solve_vander(a, B[i], n)
    return B








  