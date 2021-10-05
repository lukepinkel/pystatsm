#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 16 21:48:19 2020

@author: lukepinkel
"""

import numpy as np # analysis:ignore
from .data_utils import cov as _cov


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
    


    
    
        