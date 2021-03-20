# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 13:35:12 2020

@author: lukepinkel
"""
import numba
import numpy as np
from ..utilities.data_utils import _row_sum, _col_sum
 
@numba.jit(nopython=True)
def _u_center_biased(A, unbiased=False):
    n, m = A.shape
    rsum = _row_sum(A).reshape(-1, 1)
    csum = _col_sum(A).reshape(1, -1)

    rden, cden, tden = m, n, n * m
    
    col_mean = (csum / cden)
    row_mean = (rsum / rden)
    mean = csum.sum() / tden
    A = A - row_mean - col_mean + mean
    return A


@numba.jit(nopython=True)
def _u_center_unbiased(A, unbiased=False):
    n, m = A.shape
    rsum = _row_sum(A).reshape(-1, 1)
    csum = _col_sum(A).reshape(1, -1)
    rden, cden, tden = m - 2, n - 2, (n - 1) * (m - 2)
    col_mean = (csum / cden)
    row_mean = (rsum / rden)
    mean = csum.sum() / tden
    A = A - row_mean - col_mean + mean
    np.fill_diagonal(A, 0)
    return A

@numba.jit(nopython=True, parallel=True, nogil=True)
def euclidean_distance(X):
    n = X.shape[0]
    A = np.zeros((n, n), dtype=numba.float64)
    for i in numba.prange(1, n):
        for j in numba.prange(i):
            A[i, j] = A[j, i] = ((X[i] - X[j])**2).sum()**0.5
    return A


@numba.jit(nopython=True)
def _dcorr(X, Y, unbiased=True):
    Dx = euclidean_distance(X)
    Dy = euclidean_distance(Y)
    
    if unbiased:
        Ax = _u_center_unbiased(Dx)
        Ay = _u_center_unbiased(Dy)
    else:
        Ax = _u_center_biased(Dx)
        Ay = _u_center_biased(Dy)
    Vxx = np.mean(Ax*Ax)
    Vyy = np.mean(Ay*Ay)
    Sxy = np.mean(Ax*Ay)
    Rxy = Sxy / np.sqrt(Vxx*Vyy)
    return Rxy



def dcorr(X, Y, unbiased=True):
    if X.ndim==1:
        X = X.reshape(-1, 1)
    if Y.ndim==1:
        Y = Y.reshape(-1, 1)
    return _dcorr(X, Y, unbiased)
 
@numba.jit(nopython=True, parallel=True)
def _dcorr_permutation_test(X, Y, n_perms=1000, unbiased=True):
    n = X.shape[0]
    r = _dcorr(X, Y, unbiased=unbiased)
    c = 0.0
    for i in numba.prange(n_perms):
        rp =  _dcorr(X, Y[np.random.permutation(n)], unbiased=unbiased)
        c += abs(rp)>abs(r)
    p = c / n
    return r, p
    
@numba.jit(nopython=True, parallel=True)
def _dcorr_permutation_dist(X, Y, n_perms=1000, unbiased=True):
    n = X.shape[0]
    r = _dcorr(X, Y, unbiased=unbiased)
    permutation_dist = np.zeros(n_perms)
    for i in numba.prange(n_perms):
        rp =  _dcorr(X, Y[np.random.permutation(n)], unbiased=unbiased)
        permutation_dist[i] = rp
    return r, permutation_dist
    
def dcorr_permutation_test(X, Y, n_perms=100, unbiased=True, return_samples=True):
    if X.ndim==1:
        X = X.reshape(-1, 1)
    if Y.ndim==1:
        Y = Y.reshape(-1, 1)
    if return_samples:
        r, permutation_dist = _dcorr_permutation_dist(X, Y, n_perms, unbiased)
        p_value = (np.abs(r)<np.abs(permutation_dist)).mean()
    else:
        r, p_value = _dcorr_permutation_test(X, Y, n_perms, unbiased)
        permutation_dist = None
    return r, permutation_dist, p_value

@numba.jit(nopython=True, parallel=True)
def _gauss_kernel(X, bw):
    n, m = X.shape
    K = np.zeros((n, n))
    v = 2.0 * bw**2
    for i in numba.prange(n):
        for j in numba.prange(i):
            u = ((X[i] - X[j])**2).sum()
            K[i, j] = K[j, i] = u
    K = np.exp(-K/v)
    return K

@numba.jit(nopython=True)
def _median_bandwith(X):
    n, m = X.shape
    q = n * (n + 1) // 2 - n
    w = np.zeros((q,), dtype=numba.float64)
    k = 0
    for i in numba.prange(1, n):
        for j in numba.prange(i):
            w[k] = ((X[i] - X[j])**2).sum()
            k+=1
    bw = np.sqrt(np.median(w)*0.5)
    return bw


@numba.jit(nopython=True)
def _dhsic(X, Y, bwx, bwy):
    n = X.shape[0]
    Kx, Ky = _gauss_kernel(X, bwx), _gauss_kernel(Y, bwy)
    
    a = np.sum(Kx * Ky)
    b = 1 / n**4 * Kx.sum() * Ky.sum()
    c = np.sum(2 / n**3 * _col_sum(Kx) * _col_sum(Ky))
    stat = 1 / n**2 * a + b - c
    return stat

def dhsic(X, Y, bw='median'):
    if X.ndim==1:
        X = X.reshape(-1, 1)
    if Y.ndim==1:
        Y = Y.reshape(-1, 1)
    if bw == 'median':
        bwx = _median_bandwith(X)
        bwy = _median_bandwith(Y)
    elif bw == 'nvars':
        bwx = ((1.0 / X.shape[0]) / 2.0)**0.5
        bwy = ((1.0 / Y.shape[0]) / 2.0)**0.5
    elif type(bw) is list:
        bwx, bwy = bw
    
    stat = _dhsic(X, Y, bwx, bwy)
    return stat, bwx, bwy


@numba.jit(nopython=True, parallel=True)
def dhsic_permutation_test(X, Y, bwx, bwy, n_perms):
    samples = np.zeros(n_perms, dtype=numba.float64)
    for i in numba.prange(n_perms):
        samples[i] = _dhsic(X, Y[np.random.permutation(Y.shape[0])], bwx, bwy)
    return samples

def dhsic_test(X, Y, bw='median', n_perms=200):
    if X.ndim==1:
        X = X.reshape(-1, 1)
    if Y.ndim==1:
        Y = Y.reshape(-1, 1)
    stat, bwx, bwy = dhsic(X, Y, bw=bw)
    samples = dhsic_permutation_test(X, Y, bwx, bwy, n_perms)
    return stat, bwx, bwy, samples, (np.abs(samples)>np.abs(stat)).mean()
   





