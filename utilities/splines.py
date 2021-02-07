#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  2 03:48:59 2020

@author: lukepinkel
"""
import numba
import scipy as sp           # analysis:ignore
import scipy.interpolate
import numpy as np           # analysis:ignore


def difference_mat(k, order=2):
    Dk = np.diff(np.eye(k), order, axis=0)
    return Dk
    

def equispaced_knots(x, degree, ndx):
    xl = np.min(x)
    xr = np.max(x)
    dx = (xr - xl) / ndx
    order = degree + 1
    lb = xl - order * dx
    ub = xr + order * dx
    knots = np.arange(xl - order * dx, xr + order * dx, dx)
    return knots, order, lb, ub

def _bspline(x, knots, degree, deriv=0):
    if len(knots)<=(degree+1):
        raise ValueError("Number of knots must be greater than order")
    order = degree + 1
    q = len(knots) - order
    u = np.zeros(q)
    B = np.zeros((len(x), q))
    for i in range(q):
        u[i] = 1
        tck = (knots, u, degree)
        B[:, i] = sp.interpolate.splev(x, tck, der=deriv)
        u[i] = 0
    return B[:, 1:]

def _bspline_des(x, degree=3, ndx=20, deriv=0):
    knots, order, _, _ = equispaced_knots(x, degree, ndx)
    B = _bspline(x, knots, degree, deriv)
    return B

        
def bspline_knots(x, df, degree):
    nk = df - degree + 1
    xu, xl = np.max(x), np.min(x)
    xr = xu - xl
    xl, xu = xl - xr*0.001, xu + xr * 0.001
    dx = (xu - xl) / (nk - 1)
    all_knots = np.linspace(xl-dx*degree, xu+dx*degree, nk+2*degree)
    inner_knots = all_knots[degree:-(degree)]
    return all_knots, inner_knots

def bspline_basis(x, knots, degree, deriv=0, intercept=False):
    if len(knots)<=(degree+1):
        raise ValueError("Number of knots must be greater than order")
    order = degree + 1
    q = len(knots) - order
    u = np.zeros(q)
    B = np.zeros((len(x), q))
    for i in range(q):
        u[i] = 1
        tck = (knots, u, degree)
        B[:, i] = sp.interpolate.splev(x, tck, der=deriv)
        u[i] = 0
    if intercept:
        B = B[:, 1:]
    return B

def bspline_penalty(all_knots, knots, degree, penalty=2, intercept=False):
    pord = penalty-1
    h1 = np.repeat(np.diff(knots)/pord, pord)
    k1 = np.concatenate((np.array([knots[0]]), h1)).cumsum()
    D = bspline_basis(k1, all_knots, degree, deriv=2, intercept=False)
    a = np.tile(np.linspace(-1, 1, pord+1), pord+1)
    b = np.repeat(np.arange(0, pord+1), pord+1)
    P = (a**b).reshape(pord+1, pord+1, order='F')
    P = np.linalg.inv(P)
    i1 = np.tile(np.arange(1, pord+2), pord+1) +\
         np.repeat(np.arange(1, pord+2), pord+1)
    H = ((1+(-1)**(i1-2)) / (i1-1)).reshape(pord+1, pord+1, order='F')
    W1 = P.T.dot(H).dot(P)
    h = np.diff(knots) / 2
    
    ld0 = np.tile(np.diag(W1), len(h))*np.repeat(h, pord+1)
    i1 = np.tile(np.arange(1, pord+1), len(h))+\
         np.repeat(np.arange(0, len(h))*(pord+1), pord)
    i1 = np.concatenate((i1, np.array([len(ld0)])))
    ld = ld0[i1.astype(int)-1]
    
    i0 = np.arange(1, len(h)) * pord + 1
    i2 = np.arange(1, len(h)) *(pord + 1)
    
    ld[i0.astype(int)-1] = ld[i0.astype(int)-1]+ld0[i2.astype(int) - 1]
    B = np.zeros((pord+1, len(ld)))
    B[0] = ld
    for i in range(1, pord+1):
        diwk = np.diag(W1, -i)
        ind = np.arange(1, len(ld)-i+1)
        tmp = np.concatenate((diwk, np.zeros(i-1)))
        bi = (np.repeat(h, pord)*np.tile(tmp, len(h)))[ind-1]
        B[i, ind.astype(int)-1] = bi
    B = sp.linalg.cholesky_banded(B, lower=True) 
    D1 = B[[0]].T*D
    for i in range(1, pord+1):
        ind = np.arange(1, D.shape[0]-i+1)
        ind = ind.astype(int) - 1
        D1[ind] = D1[ind] + B[i, ind].reshape(-1, 1) * D[ind+i]
    D = D1
    S = D.T.dot(D)
    return D, S



def crspline_knots(x, df=None, degree=3):
    xu = np.unique(x)
    knots = np.quantile(xu, np.linspace(0, 1, df))
    return knots


@numba.jit(nopython=True)
def get_DldB(n, h):
    D = np.zeros((n-2, n))
    ldb = np.zeros(n-2)
    for i in range(n-2):
        D[i, i] = 1.0 / h[i]
        D[i, i+1] = -1.0 / h[i] - 1.0 / h[i+1]
        D[i, i+2] = 1.0 / h[i+1]
        ldb[i] = (h[i]+h[i+1]) / 3.0
    return D, ldb


def crspline_penalty(xk):
    n = len(xk)
    h = np.diff(xk)
    D, ldb = get_DldB(n, h)
    sdb = h[1:-1]/6
    dptsv = sp.linalg.get_lapack_funcs("ptsv")
    ldb, sdb, BinvD, info = dptsv(ldb, sdb, D)
    F = np.concatenate((np.zeros((1, n)), BinvD,
                        np.zeros((1, n))), axis=0)
    S = D.T.dot(BinvD)
    return D, F, S

def get_crsplines(x, df=10, degree=3):
    xk = crspline_knots(x, df=df, degree=degree)
    D, F, S = crspline_penalty(xk)
    j = np.searchsorted(xk, x) - 1
    j[j==-1] = 0
    j[j==len(xk)-1] = len(xk) - 2
    h = np.diff(xk)[j]
    
    dp, dn = x - xk[j], xk[j+1] - x
    
    a_pos = dp / h
    a_neg = dn / h
    
    c_pos = (dp**3 / h - h * dp) / 6.0
    c_neg = (dn**3 / h - h * dn) / 6.0
    I = np.eye(len(xk))
    
    B = a_pos * I[j+1, :].T + a_neg * I[j, :].T+\
        c_pos * F[j+1, :].T + c_neg * F[j, :].T
    return D, F, B.T, S, xk
    

def get_bsplines(x, df=10, degree=3, penalty=2, intercept=False):
    all_knots, knots = bspline_knots(x, df, degree)
    D, S = bspline_penalty(all_knots, knots, degree=degree, penalty=penalty, intercept=intercept)
    B = bspline_basis(x, all_knots, degree=degree, deriv=0, intercept=intercept)
    return D, B, S, all_knots


def transform_spline_modelmat(X, S):
    q, r = np.linalg.qr(X.mean(axis=0).reshape(-1, 1), mode='complete')
    ZSZ = np.dot(q.T, S)[1:]
    S = np.dot(q.T, ZSZ.T)[1:].T
    X = np.dot(q.T, X.T)[1:].T
    return X, S
    
def get_penalty_scale(X, S):
    sc = np.linalg.norm(S, ord=1)/np.abs(X).sum(axis=1).max()**2
    return sc

def ccspline_knots(x, nk):
    xu = np.sort(np.unique(x))
    n = len(xu)
    dx = (n - 1.0) / (nk - 1.0)
    lb = (np.floor(dx * np.arange(1, nk - 1))+ 1.0).astype(int)
    rm = dx * np.arange(1, nk - 1) + 1.0 - lb
    knots = np.zeros(nk)
    knots[-1] = xu[n-1]
    knots[0] = xu[0]
    knots[1:-1] = xu[lb-1]*(1-rm)+xu[1:][lb-1]*rm
    return knots


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
    
def ccspline_penalty(x, xk):
    h = np.diff(xk)
    n = len(xk) - 1
    D, B = _cc_mats(h, n)
    BinvD = np.linalg.inv(B).dot(D)
    S = D.T.dot(BinvD)
    S = (S + S.T) / 2.0
    return BinvD, S

def ccspline_basis(x, xk, H):
    n = len(xk)
    h = np.diff(xk)
    j = x.copy()
    for i in range(n, 1, -1):
        j[x<=xk[i-1]] = i-1
    j1 = hj = j - 1
    j[j==(n-1)] = 0
    I = np.eye(n - 1)
    a = xk[j1+1]
    b = xk[j1]
    c = h[hj]
    
    amx = a - x
    xmb = x - b
    
    d1 = (amx**3 / (6 * c)).reshape(-1, 1)
    d2 = ((xmb**3) / (6 * c)).reshape(-1, 1)
    d3 = ((c * amx / 6)).reshape(-1, 1)
    d4 = ((c * xmb / 6)).reshape(-1, 1)
    d5 = (amx / c ).reshape(-1, 1)
    d6 = (xmb / c).reshape(-1, 1)
    A, B, C, D = H[j1], H[j], I[j1], I[j]
    X = A * d1 + B * d2 - A * d3 -\
        B * d4 + C * d5 + D * d6
    return X
    
def get_ccsplines(x, df=10):
    knots = ccspline_knots(x, df)
    BinvD, S = ccspline_penalty(x, knots)
    X = ccspline_basis(x, knots, BinvD)
    return X, BinvD, S, knots
    