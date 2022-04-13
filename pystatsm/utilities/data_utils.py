#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 21:28:03 2020

@author: lukepinkel
"""
import numba
import numpy as np
import pandas as pd

@numba.jit(nopython=True, parallel=True)
def _col_sum(X):
    p = X.shape[1]
    res = np.zeros(p, dtype=numba.float64)
    for i in numba.prange(p):
        res[i] = X[:, i].sum()
    return res

@numba.jit(nopython=True, parallel=True)
def _row_sum(X):
    p = X.shape[0]
    res = np.zeros(p, dtype=numba.float64)
    for i in numba.prange(p):
        res[i] = X[i].sum()
    return res

@numba.jit(nopython=True)
def _col_mean(X):
    m = _col_sum(X) / X.shape[0]
    return m

@numba.jit(nopython=True, parallel=True)
def _col_std(X):
    p = X.shape[1]
    res = np.zeros(p, dtype=numba.float64)
    for i in numba.prange(p):
        res[i] = X[:, i].std()
    return res

@numba.jit(nopython=True)
def center(X):
    X = X - _col_mean(X)
    return X

@numba.jit(nopython=True)
def standardize(X):
    X = X / _col_std(X)
    return X

@numba.jit(nopython=True)
def csd(X):
    return center(standardize(X))

@numba.jit(nopython=True)
def cov(X):
    X = center(X)
    n = X.shape[0]
    S = np.dot(X.T, X) / n
    return S

@numba.jit(nopython=True)
def corr(X):
    X = csd(X)
    n = X.shape[0]
    S = np.dot(X.T, X) / n
    return S

def ssq(arr):
    s = np.einsum("ij,ij->j", arr, arr, optimize=True)
    return s

def scale_diag(A, s):
    if s.ndim==1:
        s = s.reshape(-1, 1)
    A = s.T * A * s
    return A

def norm_diag(A):
    s = np.sqrt(1.0 / np.diag(A)).reshape(-1, 1)
    A = s.T * A * s
    return A

def eighs(A):
    u, V = np.linalg.eigh(A)
    u, V = u[::-1], V[:, ::-1]
    return u, V    

def flip_signs(V):
    j = np.argmax(np.abs(V), axis=0)
    s = np.sign(V[j, np.arange(V.shape[1])])
    V = V * s
    return V
    
def _check_type(arr):
    if type(arr) is pd.DataFrame:
        X = arr.values
        columns, index = arr.columns, arr.index
        is_pd = True
    elif type(arr) is pd.Series:
        X = arr.values
        columns, index = [arr.name], arr.index
        is_pd = True
        X = X.reshape(X.shape[0], 1)
    elif type(arr) is np.ndarray:
        X, columns, index, is_pd = arr, None, None, False 
        if X.ndim == 1:
            X = X.reshape(X.shape[0], 1)
        columns = [f"X{i}" for i in range(1, X.shape[1]+1)]
        index = np.arange(X.shape[0])
    return X, columns, index, is_pd 


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



@numba.jit(nopython=True)
def _sign_change_1d(arr, b):
    n = len(arr)
    s = np.sign(arr[0]-b)
    for i in range(1, n):
        if np.sign(arr[i]-b)!=s:
            break
    return i-1

@numba.jit(nopython=True)
def _sign_change_2d(arr, b):
    m = arr.shape[1]
    ix = np.zeros(m, dtype=numba.int64)
    for i in range(m):
        ix[i] = _sign_change_1d(arr[:, i], b[i])
    return ix


@numba.jit(nopython=True)
def _kth_sign_change_1d(arr, b, k=1):
    n = len(arr)
    s = np.sign(arr[0]-b)
    j = 0
    for i in range(1, n):
        if np.sign(arr[i]-b)!=s:
            j += 1
        if j==k:
            break
    return i-1

@numba.jit(nopython=True)
def _kth_sign_change_2d(arr, b, k=1):
    m = arr.shape[1]
    ix = np.zeros(m, dtype=numba.int64)
    for i in range(m):
        ix[i] = _kth_sign_change_1d(arr[:, i], b[i], k)
    return ix


def sign_change(arr, nth_change=1, offset=None):
    ndims = arr.ndim
    if ndims==1:
        offset = 0.0 if offset is None else offset
    elif ndims==2:
        offset = np.zeros(arr.shape[1]) if offset is None else offset
        
    if nth_change==1:
        if ndims==1:
            ix = _sign_change_1d(arr, offset)
        elif ndims==2:
            ix = _sign_change_2d(arr, offset)
    else: 
        if ndims==1:
            ix = _kth_sign_change_1d(arr, offset, nth_change)
        elif ndims==2:
            ix = _kth_sign_change_2d(arr, offset, nth_change)
    return ix


@numba.jit(nopython=True)
def _welford_update(x, n, mean, sumsq):
    n += 1
    d1 = x - mean
    mean += d1 / n
    d2 = x - mean
    sumsq += d1 * d2
    return n, mean, sumsq


@numba.jit(nopython=True)
def _welford_variance(arr, n, mean, sumsq):
    i = 0
    for _ in range(n):
        i, mean, sumsq = _welford_update(arr[i], i, mean, sumsq)
    return mean, sumsq
    
    
def welford_variance(arr, axis=None, unbiased=False):
    axis = 0 if axis is None else axis
    arr = np.asarray(arr)
    shape = list(arr.shape)
    n = shape.pop(axis)
    mean = np.zeros(tuple(shape))
    sumsq = np.zeros(tuple(shape))
    arr = np.swapaxes(arr, 0, axis)
    mean, sumsq = _welford_variance(arr, n, mean, sumsq)
    variance = sumsq / (n - unbiased)
    return mean, variance
    

@numba.jit(nopython=True)
def _welford_cov_cross(X, Y, x_mean, y_mean, xy_prod, n):
    for i in range(n):
        dx = X[i] - x_mean
        x_mean += dx / (i + 1)
        y_mean += (Y[i] - y_mean) / (i + 1)
        dy = Y[i] - y_mean
        xy_prod += np.outer(dx, dy)
    return x_mean, y_mean, xy_prod

 

@numba.jit(nopython=True)
def _welford_cov(X, x_mean, x_prod, n):
    for i in range(n):
        dx = X[i] - x_mean
        x_mean += dx / (i + 1)
        x_prod += np.outer(dx, dx) * i / (i + 1)
    return x_mean, x_prod

    
def welford_cov(arr1, arr2=None, unbiased=False):
    if arr2 is None:
        n, k = arr1.shape
        mean1, prod = np.zeros(k), np.zeros((k, k))
        mean, prod = _welford_cov(arr1, mean1, prod, n)
    else:
        n, k = arr1.shape
        p = arr2.shape[1]
        mean1, mean2, prod = np.zeros(k), np.zeros(p), np.zeros((k, p))
        mean1, mean2, prod = _welford_cov_cross(arr1, arr2, mean1, mean2, prod, n)
        mean = np.r_[mean2, mean1]
    cov = prod / (n - unbiased)
    return mean, cov 





