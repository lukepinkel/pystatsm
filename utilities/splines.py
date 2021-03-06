#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  2 03:48:59 2020

@author: lukepinkel
"""
import numpy as np  
import scipy as sp
import scipy.interpolate
from .spline_utils import _cc_mats, _cr_mats

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

        
def absorb_constraints(q, X=None, S=None):
    if X is not None:
        X = np.dot(q.T, X.T)[1:].T
    if S is not None:
        S = np.dot(q.T, np.dot(q.T, S)[1:].T)[1:].T
    return X, S

def get_penalty_scale(X, S):
    sc = np.linalg.norm(S, ord=1)/np.abs(X).sum(axis=1).max()**2
    return sc

def bspline_knots(x, df, degree=3):
    nk = df - degree + 1
    xu, xl = np.max(x), np.min(x)
    xr = xu - xl
    xl, xu = xl - xr*0.001, xu + xr * 0.001
    dx = (xu - xl) / (nk - 1)
    all_knots = np.linspace(xl-dx*degree, xu+dx*degree, nk+2*degree)
    inner_knots = all_knots[degree:-(degree)]
    return all_knots, inner_knots

def crspline_knots(x, df=None, degree=3):
    xu = np.unique(x)
    knots = np.quantile(xu, np.linspace(0, 1, df))
    return knots

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

def bspline_basis(x, knots, deriv=0, degree=3):
    order = degree + 1
    q = len(knots) - order
    u = np.zeros(q)
    B = np.zeros((len(x), q))
    for i in range(q):
        u[i] = 1
        tck = (knots, u, degree)
        B[:, i] = sp.interpolate.splev(x, tck, der=deriv)
        u[i] = 0
    return B
    
def crspline_basis(x, knots, F):
    j = np.searchsorted(knots, x) - 1
    j[j==-1] = 0
    j[j==len(knots)-1] = len(knots) - 2
    h = np.diff(knots)[j]
    dp, dn = x - knots[j], knots[j+1] - x
    ap, an = dp / h, dn / h    
    cp, cn = (dp**3 / h - h * dp) / 6.0, (dn**3 / h - h * dn) / 6.0
    I = np.eye(len(knots))
    Xt = ap * I[j+1, :].T + an * I[j, :].T + cp * F[j+1, :].T + cn * F[j, :].T
    X = Xt.T
    return X
    
def ccspline_basis(x, knots, F):
    n, h, j = len(knots), np.diff(knots), x.copy()
    for i in range(n, 1, -1):
        j[x<=knots[i-1]] = i-1
    j1 = hj = j - 1
    j[j==(n-1)] = 0
    I = np.eye(n - 1)
    a, b, c = knots[j1+1], knots[j1], h[hj]    
    amx, xmb = a - x, x - b
    d1 = (amx**3 / (6 * c)).reshape(-1, 1)
    d2 = (xmb**3 / (6 * c)).reshape(-1, 1)
    d3 = ((c * amx / 6)).reshape(-1, 1)
    d4 = ((c * xmb / 6)).reshape(-1, 1)
    d5 = (amx / c ).reshape(-1, 1)
    d6 = (xmb / c).reshape(-1, 1)
    A, B, C, D = F[j1], F[j], I[j1], I[j]
    X = A * d1 + B * d2 - A * d3 - B * d4 + C * d5 + D * d6
    return X

def bspline_penalty(all_knots, knots, degree, penalty=2, intercept=False):
    pord = penalty-1
    h1 = np.repeat(np.diff(knots)/pord, pord)
    k1 = np.concatenate((np.array([knots[0]]), h1)).cumsum()
    G = bspline_basis(k1, all_knots, degree, deriv=2, intercept=False)
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
    R = sp.linalg.cholesky_banded(B, lower=True) 
    D = R[[0]].T*G
    for i in range(1, pord+1):
        ind = np.arange(1, G.shape[0]-i+1)
        ind = ind.astype(int) - 1
        D[ind] = D[ind] + R[i, ind].reshape(-1, 1) * G[ind+i]
    S = D.T.dot(D)
    return D, S

def crspline_penalty(knots, ret_s=True):
    n, h = len(knots), np.diff(knots)
    D, ldb = _cr_mats(n, h)
    sdb = h[1:-1]/6
    dptsv = sp.linalg.get_lapack_funcs("ptsv")
    ldb, sdb, BinvD, info = dptsv(ldb, sdb, D)
    F = np.concatenate((np.zeros((1, n)), BinvD, np.zeros((1, n))), axis=0)
    S = D.T.dot(BinvD) if ret_s else None
    return S, F

def ccspline_penalty(knots, ret_s=True):
    n, h = len(knots) - 1, np.diff(knots)
    D, B = _cc_mats(h, n)
    F = np.linalg.inv(B).dot(D)
    if ret_s:
        S = D.T.dot(F)
        S = (S + S.T) / 2.0
    else:
        S = None
    return S, F

def _get_crsplines(x, df=10):
    knots = crspline_knots(x, df=df)
    S, F = crspline_penalty(knots)
    X = crspline_basis(x, knots, F)
    fkws = {"F":F}
    return X, S, knots, fkws
    
def _get_bsplines(x, df=10):
    knots, inner_knots = bspline_knots(x, df)
    F, S = bspline_penalty(knots, inner_knots)
    X = bspline_basis(x, knots, deriv=0)
    fkws = {"deriv":0}
    return X, S, knots, fkws

def _get_ccsplines(x, df=10):
    knots = ccspline_knots(x, df)
    S, F = ccspline_penalty(x, knots)
    X = ccspline_basis(x, knots, F)
    fkws = {"F":F}
    return X, S, knots, fkws

    
def gcv(a, X, y, S):
    A = X.dot(np.linalg.inv(X.T.dot(X)+S*a)).dot(X.T)
    r = y - A.dot(y)
    n = y.shape[0]
    v = n * r.T.dot(r) / (n - np.trace(A))**2
    return v

def double_gcv(a, X, y, S, gamma=1.5):
    A = X.dot(np.linalg.inv(X.T.dot(X)+S*a)).dot(X.T)
    r = y - A.dot(y)
    n = y.shape[0]
    v = n * r.T.dot(r) / (n - gamma * np.trace(A))**2
    return v

def grad_gcv(a, X, y, St):
    V = np.linalg.inv(X.T.dot(X)+St*a)
    A = X.dot(V).dot(X.T)
    r = y - A.dot(y)
    M = X.dot(V).dot(St).dot(V).dot(X.T)
    n = y.shape[0]
    u = n - np.trace(A)
    u2 = u**2
    u3 = u2 * u
    g = 2 * n * r.T.dot(M.dot(y) / u2 - np.trace(M) * r / u3)
    return np.atleast_1d(g)


def grad_double_gcv(a, X, y, St, gamma=1.5):
    V = np.linalg.inv(X.T.dot(X)+St*a)
    A = X.dot(V).dot(X.T)
    r = y - A.dot(y)
    M = X.dot(V).dot(St).dot(V).dot(X.T)
    n = y.shape[0]
    u = n - gamma * np.trace(A)
    u2 = u**2
    u3 = u2 * u
    g = 2 * n * r.T.dot(M.dot(y) / u2 - gamma * np.trace(M) * r / u3)
    return np.atleast_1d(g)
