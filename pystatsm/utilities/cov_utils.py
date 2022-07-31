#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 16 21:48:19 2020

@author: lukepinkel
"""

import numpy as np # analysis:ignore
import scipy as sp
import scipy.stats
from .data_utils import cov as _cov, scale_diag, normalize_xtrx
from .func_utils import handle_default_kws
from .indexing_utils import diag_indices
from .linalg_operations import eighs
from .random import r_lkj

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
    

def get_ar1_corr(n=1, rho=0.0):
    """

    Parameters
    ----------
    n : int, optional
        Size of correlation matrix. The default is 1.
    rho : float, optional
        Autocorrelation. The default is 0.0.

    Returns
    -------
    R : array
        AR1(rho) correlation matrix.

    """
    R = rho**sp.linalg.toeplitz(np.arange(n))
    return R

def get_exchangeable_corr(n=1, c=0.0):
    """

    Parameters
    ----------
    n : int
        Size of correlation matrix.  The default is 1
    c : float
        Off diagonal constant. The default is 0.0.

    Returns
    -------
    R : array
        Exchangeable/Compound Symmetry correlation matrix.

    """
    lb = -1.0 / (n - 1)
    if c < lb or c>1:
        raise ValueError("c must be between -1/(n-1) and 1")
    R = np.eye(n)
    R[np.tril_indices(n, -1)] = R[np.triu_indices(n, 1)] = c
    return R

def get_mdependent_corr(off_diags):
    """

    Parameters
    ----------
    off_diags : list
        List of arrays with off_diags[i] corresponding to the ith diagonal.

    Returns
    -------
    R : array
        m-dependent correlation matrix.

    """
    n = np.max([len(r) for r in off_diags])+1
    m = len(off_diags)
    R = np.eye(n)
    for i in range(1, m+1):
        R[diag_indices(n, -i)] = R[diag_indices(n, i)] = off_diags[i-1]
    return R

def get_antedependence1_corr(rho):
    """

    Parameters
    ----------
    rho : array

    Returns
    -------
    R : array
        Antedependence 1 Correlation matrix.
    
    rho with n elements produces an n+1 sized correlation matrix 

    """
    n = len(rho) + 1
    r = []
    for i in range(n-1):
        r.append(np.cumprod(rho[i:]))
    r = np.concatenate(r)
    R = np.eye(n)
    R[np.triu_indices(n, 1)[::-1]] = R[np.tril_indices(n, -1)[::-1]] = r
    return R

def get_factor_cov(A, Psi):
    """

    Parameters
    ----------
    A : array
        (n x k) array of factor loadings.
    Psi : array
        vector or (n x n) diagonal array of n residual covariances.

    Returns
    -------
    S : array
        Factor structure covariance matrix.

    """
    if Psi.ndim==2:
        Psi = np.diag(Psi)
    S = np.dot(A, A.T)
    S[np.diag_indices_from(S)] += Psi
    return S

def get_spatialpower_corr(rho, distances):
    """

    Parameters
    ----------
    rho : float
        Correlation parameter.
    distances : array
        Either n*(n-1)/2 vector of distances or n*n distance matrix

    Returns
    -------
    R : array
        Spatial Power Correlation.

    """
    if distances.ndim==2:
        R = rho**distances
    else:
        R = sp.spatial.distance.squareform(rho**distances)
        R[np.diag_indices_from(R)] += 1.0
    return R

def get_spatialexp_corr(theta, distances):
    """

    Parameters
    ----------
    theta : float
    distances : array
        Either n*(n-1)/2 vector of distances or n*n distance matrix

    Returns
    -------
    R : array
    """
    if distances.ndim==2:
        R = np.exp(-distances / theta)
    else:
        R = sp.spatial.distance.squareform(np.exp(-distances/theta))
        R[np.diag_indices_from(R)] += 1.0
    return R

def get_spatialgaussian_corr(s, distances):
    """

    Parameters
    ----------
    s : float
    distances : array
        Either n*(n-1)/2 vector of distances or n*n distance matrix
    Returns
    -------
    R : array
    """
    if distances.ndim==2:
        R = np.exp(-distances / s**2)
    else:
        R = sp.spatial.distance.squareform(np.exp(-distances/s**2))
        R[np.diag_indices_from(R)] += 1.0
    return R

def get_toeplitz_corr(rho):
    """

    Parameters
    ----------
    rho : array
        n vector of off diagonal components
    Returns
    -------
    R : array
         (n+1) x (n+1) correlation matrix
    """
    R = sp.linalg.toeplitz(np.r_[1., rho])
    return R

def get_lkj_corr(n_vars=1, eta=1.0, r_kws=None):
    r_kws = handle_default_kws(r_kws, dict(n=1, seed=None, rng=None))
    r_kws["dim"], r_kws["eta"] = n_vars, eta
    R = r_lkj(**r_kws)
    if r_kws["n"]==1:
        R = R[0, 0]
    return R

def _get_joint_corr_eig(Sxx, Syy, Vx, Vy, r):
    Wx, Wy = normalize_xtrx(Vx, Sxx), normalize_xtrx(Vy, Syy)
    C = np.einsum("ij,j,kj->ik", Wx, r, Wy) 
    Sxy = Sxx.dot(C).dot(Syy)
    S = np.block([[Sxx, Sxy], [Sxy.T, Syy]])
    return S

def _get_joint_corr_r(Sxx, Syy, r):
    x_eig, A0 = eighs(Sxx)
    y_eig, B0 = eighs(Syy)
    
    A = A0 * (np.sqrt(1/x_eig))
    B = B0 * (np.sqrt(1/y_eig))
    
    B_inv = np.linalg.inv(B)
    
    B1 = B_inv[:Sxx.shape[0]]
                    
    Sxy = np.linalg.inv(A.T).dot(np.diag(r)).dot(B1)
    S = np.block([[Sxx, Sxy], [Sxy.T, Syy]])
    return S

def get_canvar_corr(Sxx, Syy, r, Vx=None, Vy=None):
    if Vx is not None and Vy is not None:
        S = _get_joint_corr_eig(Sxx, Syy, Vx, Vy, r)
    else:
        S = _get_joint_corr_r(Sxx, Syy, r)
    return S

def get_eig_corr(u, rng=None):
    u = u / np.sum(u) * len(u)
    R = sp.stats.random_correlation.rvs(u, random_state=rng)
    return R

        