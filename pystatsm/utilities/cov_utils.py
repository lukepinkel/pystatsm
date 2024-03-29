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
from .special_mats import dmat, dpmat, nmat, pattern_mat_d, pattern_mat_s, pattern_mat_k
from .linalg_operations import eighs, vech, vec
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
    

def get_ar1_corr(n_var=1, rho=0.0):
    """

    Parameters
    ----------
    n_var : int, optional
        Size of correlation matrix. The default is 1.
    rho : float, optional
        Autocorrelation. The default is 0.0.

    Returns
    -------
    R : array
        AR1(rho) correlation matrix.

    """
    R = rho**sp.linalg.toeplitz(np.arange(n_var))
    return R

def get_exchangeable_corr(n_var=1, c=0.0):
    """

    Parameters
    ----------
    n_var: int
        Size of correlation matrix.  The default is 1
    c : float
        Off diagonal constant. The default is 0.0.

    Returns
    -------
    R : array
        Exchangeable/Compound Symmetry correlation matrix.

    """
    lb = -1.0 / (n_var - 1)
    if c < lb or c>1:
        raise ValueError("c must be between -1/(n_var-1) and 1")
    R = np.eye(n_var)
    R[np.tril_indices(n_var, -1)] = R[np.triu_indices(n_var, 1)] = c
    return R

def get_mdependent_corr(off_diags, n_var=None):
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
    n_var = np.max([len(r) for r in off_diags])+1
    m = len(off_diags)
    R = np.eye(n_var)
    for i in range(1, m+1):
        R[diag_indices(n_var, -i)] = R[diag_indices(n_var, i)] = off_diags[i-1]
    return R

def get_antedependence1_corr(rho, n_var=None):
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
    n_var = len(rho) + 1
    r = []
    for i in range(n_var-1):
        r.append(np.cumprod(rho[i:]))
    r = np.concatenate(r)
    R = np.eye(n_var)
    R[np.triu_indices(n_var, 1)[::-1]] = R[np.tril_indices(n_var, -1)[::-1]] = r
    return R

def get_factor_cov(A, Psi, Phi=None, n_var=None):
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
    Phi = np.eye(A.shape[1]) if Phi is None else Phi
    if Psi.ndim==2:
        Psi = np.diag(Psi)
    S = np.dot(A.dot(Phi), A.T)
    S[np.diag_indices_from(S)] += Psi
    return S

def get_spatialpower_corr(rho, distances, n_var=None):
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

def get_spatialexp_corr(theta, distances, n_var=None):
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

def get_spatialgaussian_corr(s, distances, n_var=None):
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

def get_toeplitz_corr(rho, n_var=None):
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

def get_lkj_corr(n_var=1, eta=1.0, r_kws=None):
    r_kws = handle_default_kws(r_kws, dict(n=1, seed=None, rng=None))
    r_kws["dim"], r_kws["eta"] = n_var, eta
    R = r_lkj(**r_kws)
    if r_kws["n"]==1:
        R = R[0, 0]
    return R

def _get_joint_corr_eig(Sxx, Syy, Vx, Vy, r, n_var=None):
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

def get_canvar_corr(Sxx, Syy, r, Vx=None, Vy=None, n_var=None):
    if Vx is not None and Vy is not None:
        S = _get_joint_corr_eig(Sxx, Syy, Vx, Vy, r)
    else:
        S = _get_joint_corr_r(Sxx, Syy, r)
    return S

def get_eig_corr(u, rng=None, n_var=None):
    u = u / np.sum(u) * len(u)
    R = sp.stats.random_correlation.rvs(u, random_state=rng)
    return R

def get_eigvals(n_var=10, p_eff=0.5, a=1.0, b=0.1, c=0.5):
    x = np.arange(n_var) / (n_var * p_eff)
    u = ((1 - c) * np.exp(-a * x**2) + c * np.exp(-b * x))**2
    return u

def multivar_marginal_kurtosis(X):
    k = sp.stats.moment(X, 4)/(sp.stats.moment(X, 2)**2)/3.0
    return k



def cov_sample_cov(X=None, S=None, excess_kurt=None, kurt=None):
    if S is None:
        m = np.mean(X, axis=0)
        N = X.shape[0]
        n = N - 1.0
        Z = X - m
        S = Z.T.dot(Z) / n
    if kurt is None:
        if excess_kurt is None:
            if X is None:
                kurt = np.ones(S.shape[1])
            else:
                kurt = multivar_marginal_kurtosis(X)
        else:
            kurt = (excess_kurt + 3.0) / 3.0
    D = dpmat(S.shape[0])
    u = np.atleast_2d(np.sqrt(kurt))
    A = 0.5 * (u + u.T)
    C = A*S
    v = np.vstack([vech(C), vech(S)]).T
    M = np.eye(2)
    M[1, 1] = -1
    CoC = np.kron(C, C)
    tmp = D.dot(CoC)
    (D.dot(tmp.T)).T
    V = 2*(D.dot(tmp.T)).T + v.dot(M).dot(v.T)
    return V      

    

def corr_acov_neudecker(S):
    p = S.shape[0]
    s = np.sqrt(np.diag(S))
    vS = vec(S)
    w = 1 / s
    ww = np.kron(w, w)
    R = scale_diag(S, w)
    Ms = nmat(p) / 2.0
    Md = pattern_mat_d(p)
    IoR = np.kron(np.eye(p), R)
    MsIoR = Ms.dot(IoR)
    A = (np.eye(p**2) -  (Md.T.dot(MsIoR.T)).T) *  ww
    V = 2 * Ms.dot(np.kron(S, S)) + np.outer(vS, vS)
    Acov = A.dot(V).dot(A.T)
    return Acov

def corr_acov_browne(S, kurt=1.0):
    p = S.shape[0]
    s = np.sqrt(np.diag(S))
    w = 1 / s
    #vS = vec(S)
    R = scale_diag(S, w)
    vS = vec(R)
    Ms = pattern_mat_s(p)
    Kd = pattern_mat_k(p)
    A = Ms.dot(np.kron(R, np.eye(p)))
    A = (Kd.T.dot(A.T)).T
    G = 2.0 * Ms.dot(np.kron(R, R))
    G = kurt * G + (kurt - 1) * np.outer(vS, vS)
    B = (Kd.T.dot(G.T)).T
    C = Kd.T.dot(B)
    ABt = np.dot(A, B.T)
    Acov = G - ABt - ABt.T + A.dot(C).dot(A.T)
    return Acov
