#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 16 21:48:19 2020

@author: lukepinkel
"""

import numba # analysis:ignore
import numpy as np # analysis:ignore
from .data_utils import corr as _corr, cov as _cov, csd as _csd


def project_psd(A, tol_e=1e-9, use_transpose=True):
    u, V = np.linalg.eigh(A)
    VU = (V * np.maximum(u, tol_e))
    if use_transpose:
        A = VU.dot(V.T)
    else:
        A = VU.dot(np.linalg.inv(V)) # probably should just use .H
    return A
    

def near_psd(A, n_iters=200, tol=1e-7, tol_e=1e-7, dykstra=True,
             symmetrize=False, unit_diag=True, keep_diag=False, 
             return_info=False):
    n = A.shape[1]
    D = np.zeros_like(A)
    X = A.copy()
    dg_indices = (np.arange(n), np.arange(n))
    if unit_diag:
        dgs = np.ones(n)
    elif keep_diag:
        dgs = np.diag(A)
        
    for i in range(n_iters):
        Y = X.copy()
        if dykstra:
            R = Y - D
            d, Q = np.linalg.eigh(R)
        else:
            d, Q = np.linalg.eigh(Y)
        positive_mask = (d > tol_e * d.max())
        if  not positive_mask.any():
            break
        Q = Q[:, positive_mask]
        X = (Q * d[positive_mask]).dot(Q.T)
        
        if dykstra:
            D = X - R
        
        if symmetrize:
            X = (X + X.T) / 2.0
        if unit_diag or keep_diag:
            X[dg_indices] = dgs
        ndiff = np.linalg.norm(Y - X) / np.linalg.norm(Y)
        if ndiff<tol:
            break
    X = project_psd(X, tol_e=tol_e)
    if unit_diag or keep_diag:
        X[dg_indices] = dgs
    
    if return_info:
        return X, dict(n_iters=i, normalized_diff=ndiff)
    else:  
        return X
            

@numba.jit(nopython=True)
def vine_corr(d, eta=1, beta=None, seed=None):
    if beta is None:
        beta = eta + (d - 1) / 2.0
    if seed is not None:
        np.random.seed(seed)
    P = np.zeros((d, d))
    S = np.eye(d)
    for k in range(d-1):
        beta -= 0.5
        for i in range(k+1, d):
            P[k, i] = np.random.beta(beta, beta)
            P[k, i] = (P[k, i] - 0.5)*2.0
            p = P[k, i]
            for l in range(k-1, 1, -1):
                p = p * np.sqrt((1 - P[l, i]**2)*(1 - P[l, k]**2)) + P[l, i]*P[l, k]
            S[k, i] = p
            S[i, k] = p
    u, V = np.linalg.eigh(S)
    umin = np.min(u[u>0])
    u[u<0] = [umin*0.5**(float(i+1)/len(u[u<0])) for i in range(len(u[u<0]))]
    V = np.ascontiguousarray(V)
    S = V.dot(np.diag(u)).dot(np.ascontiguousarray(V.T))
    v = np.diag(S)
    v = np.diag(1/np.sqrt(v))
    S = v.dot(S).dot(v)
    return S

@numba.jit(nopython=True)
def onion_corr(d, eta=1, beta=None):
    if beta is None:
        beta = eta + (d - 2) / 2.0
    u = np.random.beta(beta, beta)
    r12 = 2 * u  - 1
    S = np.array([[1, r12], [r12, 1]])
    I = np.array([[1.0]])
    for i in range(3, d+1):
        beta -= 0.5
        r = np.sqrt(np.random.beta((i - 1) / 2, beta))
        theta = np.random.normal(0, 1, size=(i-1, 1))
        theta/= np.linalg.norm(theta)
        w = r * theta
        c, V = np.linalg.eig(S)
        R = (V * np.sqrt(c)).dot(V.T)
        q = R.dot(w)
        S = np.concatenate((np.concatenate((S, q), axis=1),
                            np.concatenate((q.T, I), axis=1)), axis=0)
    return S


@numba.jit(nopython=True)
def exact_rmvnorm(S, n=1000, mu=None, seed=None):
    if seed is not None:
        np.random.seed(seed)
    p = S.shape[0]
    U, d, _ = np.linalg.svd(S)
    d = d.reshape(1, -1)
    L = U * d**0.5
    X = _csd(np.random.normal(0.0, 1.0, size=(n, p)))
    R = _corr(X)
    L = L.dot(np.linalg.inv(np.linalg.cholesky(R)))
    X = X.dot(L.T)
    if mu is not None:
        X = X + mu
    return X

def _whiten(X, keep_mean=False):
    mean = np.mean(X, axis=0) if keep_mean else np.zeros(X.shape[1])
    S = _cov(X)
    L = np.linalg.inv(np.linalg.cholesky(S))
    Y = mean + X.dot(L.T)
    return Y

def _svd_cov_transform(S):
    U, d, _ = np.linalg.svd(S, full_matrices=False)
    L = U * np.sqrt(d[:, np.newaxis])
    return L

def _exact_cov(X, mean=None, cov=None, keep_mean=False):
    p = X.shape[1]
    mean = np.zeros(p) if mean is None else mean
    cov = np.eye(p) if cov is None else cov
    Y = _whiten(X, keep_mean=keep_mean)
    L = _svd_cov_transform(cov)
    Z = mean + Y.dot(L.T)
    return Z
    

def students_t(loc, scale, nu=1, size=None, rng=None):
     rng = np.random.default_rng() if rng is None else rng
     x = loc + rng.standard_t(df=nu, size=size) * scale
     return x
    
def multivariate_t(mean, cov, nu=1, size=None, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    u = np.sqrt(nu / rng.chisquare(df=nu, size=size))[:, np.newaxis]
    Y = rng.multivariate_normal(mean=mean, cov=cov, size=size)
    X = mean + u * Y
    return X
    
    
    
    
    
        