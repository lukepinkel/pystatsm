#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 16 21:46:16 2020

@author: lukepinkel
"""

import numba # analysis:ignore
import numpy as np # analysis:ignore
import scipy as sp # analysis:ignore
import scipy.sparse as sps # analysis:ignore
from .indexing_utils import flat_mgrid

def difference_mat(k, order=2):
    Dk = np.diff(np.eye(k), order, axis=0)
    return Dk


@numba.jit(nopython=True)
def _lmat(n):
    p = int(n * (n + 1) / 2)
    template = np.arange(n)
    z = np.ones((p,), dtype=numba.int64)
    k = np.zeros((p,), dtype=numba.int64)
    a = int(0)
    for i in range(n):
        k[a:a+n-i] = template
        template = template[:-1] + n +1
        a = a + n - i
    return (z, (np.arange(p), k)), (p, n**2)  

    
def lmat(n):
    data, shape = _lmat(n)
    K = sp.sparse.csc_matrix(data, shape=shape)
    return K

@numba.jit(nopython=True)
def _kmat(p, q):
    p = int(p)
    q = int(q)
    pq = p * q
    
    template = np.arange(0, int(q)) * int(p)
    z = np.ones((pq, ), dtype=numba.int64)
    k = np.zeros((pq,), dtype=numba.int64)
    for i in range(p):
        k[i*q:(i+1)*q] = template + i
    return (z, (np.arange(pq), k)), (pq, pq)
    
def kmat(p, q):
    data, shape = _kmat(p, q)
    K = sp.sparse.csc_matrix(data, shape=shape)
    return K

@numba.jit(nopython=True)
def _dmat(n):
    p = int(n * (n + 1) / 2)
    m = int(n**2)
    r = int(0)
    a = int(0)
    
    d = np.zeros((m,), dtype=numba.int64)
    t = np.ones((m,), dtype=np.double)
    for i in range(n):
        d[r:r+i] = i - n + np.cumsum(n - np.arange(0, i)) + 1
        r = r + i
        d[r:r+n-i] = np.arange(a, a+n-i)+1
        r = r + n - i
        a = a + n - i 
    
    return (t, (np.arange(m), d-1)), (m, p)
    
def dmat(n):
    data, shape = _dmat(n)
    D = sp.sparse.csc_matrix(data, shape=shape)
    return D

def nmat(n):
    K = kmat(n, n)
    I = sp.sparse.eye(n**2)
    N = K + I
    return N

def dpmat(n):
    (data, indices), shape = _dmat(n)
    data = data * 0.5
    data[np.arange(0, n**2, n+1)] = 1.0
    Dp = sp.sparse.csc_matrix((data, indices), shape=shape).T
    return Dp

def kronvec_mat(A_dims, B_dims):
  n, p = A_dims
  q, r = B_dims

  Kv = sp.sparse.kron(sp.sparse.eye(p), kmat(r, n))
  Kv = sp.sparse.kron(Kv, sp.sparse.eye(q))
  return Kv

def pattern_mat_c(p):
    n = np.arange(p)
    i, j, g, h = flat_mgrid(n, n, n, n)
    ix = (((i == g) & (h==j)) | ( (i == h) & (j == g))) & (i!=j)  & (g!=h)
    
    rc = i[ix] + j[ix] * p, g[ix] + h[ix] * p
    vals = np.repeat(0.5, len(rc[0]))
    data = (vals, rc)
    shape = (p**2 , p**2)
    Mc = sp.sparse.csc_matrix(data, shape=shape)
    return Mc


def pattern_mat_d(p):
    n = np.arange(p)
    i, j, g, h = flat_mgrid(n, n, n, n)
    ix = (i==j) & (j==g) & (g==h)
    
    rc = i[ix] + j[ix] * p, g[ix] + h[ix] * p
    vals = np.repeat(1.0, len(rc[0]))
    data = (vals, rc)
    shape = (p**2 , p**2)
    Md = sp.sparse.csc_matrix(data, shape=shape)
    return Md

def pattern_mat_s(p):
    n = np.arange(p)
    i, j, g, h = flat_mgrid(n, n, n, n)
    ix1 = (i==j) & (j==g) & (g==h)
    ix2 = (((i == g) & (h==j)) | ( (i == h) & (j == g))) & (~ix1)
    
    r1, c1 = i[ix1] + j[ix1] * p, g[ix1] + h[ix1] * p
    r2, c2 = i[ix2] + j[ix2] * p, g[ix2] + h[ix2] * p
    rc = np.concatenate((r1, r2)), np.concatenate((c1, c2))
    vals = np.r_[np.repeat(1., len(r1)), np.repeat(0.5, len(r2))]
    data = (vals, rc)
    shape = (p**2 , p**2)
    Ms = sp.sparse.csc_matrix(data, shape=shape)
    return Ms

def pattern_mat_k(p):
    n = np.arange(p)
    r = n * (p+1)
    data = (np.ones(p), (r, n))
    shape = p**2, p
    Kd = sp.sparse.csc_matrix(data, shape=shape)
    return Kd


