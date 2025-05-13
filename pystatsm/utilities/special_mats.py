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
from .indexing_utils import (flat_mgrid, tril_indices,
                             commutation_matrix_indices,
                             elimination_matrix_indices)
from .python_wrappers import sparse_kron

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



def _apply_kmat1(arr, i):
    return arr[i]

def _apply_kmat2(arr, K):
    return K.dot(arr)

def apply_kmat1(arr, m, n, indices=None):
    if indices is None:
        ii = np.arange(m * n, dtype=int)
        indices = (ii % n) * m + ii // n
    out = _apply_kmat1(arr, indices)
    return out

def apply_kmat2(arr, m, n, K=None):
    if K is None:
        K =  kmat(m, n)
    out = _apply_kmat2(arr, K)
    return out


def _apply_lmat1(arr, i):
    return arr[i]

def _apply_lmat2(arr, L):
    return L.dot(arr)

def _apply_lmat_post1(arr, i):
    return arr[:, i]

def _apply_lmat_post2(arr, L):
    return sp.sparse.csc_matrix.dot(arr, L)


def _apply_lmat_pre_post1(arr, i):
    return arr[i][:, i]

def apply_lmat1(arr, n=None, indices=None):
    n = int(np.sqrt(arr.shape[0])) if n is None else n
    if indices is None:
        i, j = tril_indices(n)
        indices = j * n + i
    out = _apply_lmat1(arr, indices)
    return out


def apply_lmat2(arr, n=None, L=None):
    n = int(np.sqrt(arr.shape[0])) if n is None else n
    if L is None:
        L =  lmat(n)
    out = _apply_lmat2(arr, L)
    return out

def apply_lmat_post1(arr, n=None, indices=None):
    n = int(np.sqrt(arr.shape[1])) if n is None else n
    if indices is None:
        i, j = tril_indices(n)
        indices = j * n + i
    out = _apply_lmat_post1(arr, indices)
    return out


def apply_lmat_post2(arr, n=None, L=None):
    n = int(np.sqrt(arr.shape[1])) if n is None else n
    if L is None:
        L =  lmat(n)
    out = _apply_lmat_post2(arr, L)
    return out

def apply_lmat_pre_post1(arr, n=None, indices=None):

    n = int(np.sqrt(arr.shape[0])) if n is None else n
    if indices is None:
        i, j = tril_indices(n)
        indices = j * n + i
    out = _apply_lmat_pre_post1(arr, indices)
    return out



def _apply_dmat1(arr, rl, cl, eq, pre_mult=True, post_mult=False):
    if pre_mult:
        out = arr[rl] + arr[cl]
        out[eq] = out[eq] / 2.0
    else:
        out = arr
    if post_mult:
        out = out[:,rl] + out[:,cl]
        out[:,eq] = out[:,eq] / 2.0
    return out

def apply_dmat1(arr, n=None, row_indices=None, col_indices=None, eq_indices=None, pre_mult=True, post_mult=False):
    n = int(np.sqrt(arr.shape[0])) if n is None else n
    if row_indices is None or col_indices is None or eq_indices is None:
        il, jl = tril_indices(n)
        row_indices = jl * n + il
        col_indices = il * n + jl
        eq_indices, = np.where(il==jl)
    out = _apply_dmat1(arr, row_indices, col_indices, eq_indices, pre_mult, post_mult)
    return out

def _apply_dmat2_pre(arr, D, transpose=True):
    if transpose:
        D = D.T
    arr = D.dot(arr)
    return arr


def apply_dmat2_pre(arr, n=None, D=None, transpose=True):
    n = int(np.sqrt(arr.shape[0])) if n is None else n
    D = dmat(n) if D is None else D
    out = _apply_dmat2_pre(arr, D, transpose)
    return out


def _apply_dmat2_post(arr, D, transpose=True):
    if transpose:
        D = D.T
    arr = sp.sparse.csc_matrix.dot(arr, D)
    return arr


def apply_dmat2_post(arr, n=None, D=None, transpose=True):
    n = int(np.sqrt(arr.shape[1])) if n is None else n
    D = dmat(n) if D is None else D
    arr = _apply_dmat2_post(arr, D, transpose)
    return arr

def _apply_dmat2_pre_post(arr, D1, D2, t1=True, t2=False):
    if t1:
        D1 = D1.T
    if t2:
        D2 = D2.T
    arr = D1.dot(arr)
    arr = (D2.T.dot(arr.T)).T
    return arr

def apply_dmat2_pre_post(arr, n1=None, n2=None,D1=None, D2=None, t1=True, t2=False):
    n1 = int(np.sqrt(arr.shape[0])) if n1 is None else n1
    n2 = int(np.sqrt(arr.shape[1])) if n2 is None else n2

    if n1==n2:
        if D1 is None and D2 is not None:
            D1 = D2
        elif D2 is None and D1 is not None:
            D2 = D1
        elif D1 is None and D2 is None:
            D1 = D2 = dmat(n1)
    else:
        D1 = dmat(n1) if D1 is None else D1
        D2 = dmat(n2) if D2 is None else D2
    out = _apply_dmat2_pre_post(arr, D1, D2, t1, t2)
    return out


def _apply_dmat3(arr, rl, cl, eq, pre_mult=True, post_mult=False):
    if pre_mult:
        out = arr[...,rl,:] + arr[...,cl,:]
        out[...,eq,:] = out[...,eq,:] / 2.0
    else:
        out = arr
    if post_mult:
        out = out[...,rl] + out[...,cl]
        out[...,eq] = out[...,eq] / 2.0
    return out

def apply_dmat3(arr, n=None, row_indices=None, col_indices=None, eq_indices=None, pre_mult=True, post_mult=False):
    n = int(np.sqrt(arr.shape[-1])) if n is None else n
    if row_indices is None or col_indices is None or eq_indices is None:
        il, jl = tril_indices(n)
        row_indices = jl * n + il
        col_indices = il * n + jl
        eq_indices, = np.where(il==jl)
    out = _apply_dmat3(arr, row_indices, col_indices, eq_indices, pre_mult, post_mult)
    return out



class CommutationMatrix:
    def __init__(self, m: int, n: int):

        self.m = m
        self.n = n
        self.K = kmat(m, n)
        self.indices = self._compute_indices()

    def _compute_indices(self):
        ii = np.arange(self.m * self.n, dtype=int)
        i = (ii % self.n) * self.m + ii // self.n
        return i

    def apply(self, A):
        return A[self.indices]

    def apply_alt(self, A):
        return self.K.dot(A)

class EliminationMatrix:
    def __init__(self, n: int):
        self.n = n
        self.L = lmat(n)
        self.indices = self._compute_indices()

    def _compute_indices(self):
        i, j = tril_indices(self.n)
        indices = j * self.n + i
        return indices

    def apply(self, A):
        return A[self.indices]

    def apply_post(self, A):
        return A[:, self.indices]

    def apply_alt(self, A):
        return self.L.dot(A)

    def apply_post_alt(self, A):
        return sp.sparse.csc_matrix.dot(A, self.L)

    def app(self, A, pre=True, post=False):
        if pre:
            A = A[self.indices]
        if post:
            A = A[:, self.indices]
        return A

class DuplicationMatrix:
    def __init__(self, n: int):
        self.n = n
        self.D = dmat(n)
        self.indices_rl, self.indices_cl, self.eq_indices = self._compute_indices()

    def _compute_indices(self):
        il, jl = tril_indices(self.n)
        rl = jl * self.n + il
        cl = il * self.n + jl
        eq = np.where(il == jl)[0]
        return rl, cl, eq

    def apply(self, A, pre_mult=True, post_mult=False):
        if pre_mult:
            out = A[self.indices_rl] + A[self.indices_cl]
            out[self.eq_indices] /= 2.0
        else:
            out = A

        if post_mult:
            out = out[:, self.indices_rl] + out[:, self.indices_cl]
            out[:, self.eq_indices] /= 2.0
        return out

    def apply_pre(self, A, D=None, transpose=True):
        if D is None:
            D = self.D
        if transpose:
            D = D.T
        return D.dot(A)

    def apply_post(self, A, D=None, transpose=True):
        if D is None:
            D = self.D
        if transpose:
            D = D.T
        return (D.dot(A.T)).T

    def apply_pre_post(self, A, D1=None, D2=None, t1=True, t2=False):
        if D1 is None and D2 is None:
            D1 = D2 = self.D
        elif D1 is None:
            D1 = D2
        elif D2 is None:
            D2 = D1

        if t1:
            D1 = D1.T
        if t2:
            D2 = D2.T

        A = D1.dot(A)
        A = (D2.dot(A.T)).T
        return A

class DpMatrix:
    def __init__(self, n: int):
        self.n = n
        self.D = dpmat(n)
        self.indices_rl, self.indices_cl, self.eq_indices = self._compute_indices()

    def _compute_indices(self):
        il, jl = tril_indices(self.n)
        rl = jl * self.n + il
        cl = il * self.n + jl
        eq = np.where(il == jl)[0]
        return rl, cl, eq

    def apply(self, A, pre_mult=True, post_mult=False):
        if pre_mult:
            out = (A[self.indices_rl] + A[self.indices_cl])/2.0
        else:
            out = A

        if post_mult:
            out = (out[:, self.indices_rl] + out[:, self.indices_cl])/2.0
        return out



def comm_mat(m,n, dtype=np.double):
    mn = m * n
    i, j = commutation_matrix_indices(m, n)
    x = np.ones(mn, dtype=dtype)
    i, j = i.astype(np.int32), j.astype(np.int32)
    K = sp.sparse.csc_array((x, (i, j)), shape=(mn, mn), dtype=dtype)
    return K

def elim_mat(n, dtype=np.double):
    p = int((n * (n + 1)) // 2)
    i, j = elimination_matrix_indices(n)
    i, j = i.astype(np.int32), j.astype(np.int32)
    x = np.ones(p)
    E = sp.sparse.csc_array((x, (i, j)), shape=(p, n*n), dtype=dtype)
    return E



def make_mats(n):
    N = nmat(n).astype(np.double)
    L = lmat(n).astype(np.double)
    I2 = sp.sparse.eye(n*n, format="csc")
    K = kmat(n, n).astype(np.double)
    LN = sp.sparse.csc_matrix.dot(L, N).astype(np.double)
    LN2 = sparse_kron(L, LN)
    I = sp.sparse.eye(n, format="csc")

    Kvm = sparse_kron(sparse_kron(I, K), I)
    Iv = I.reshape(-1, 1, order='F', copy=True).tocsc()
    I2Iv = sparse_kron(I2, Iv)
    D2=LN2.dot(Kvm.dot(I2Iv.dot(L.T)))
    return LN, I, L, D2

def make_mats2(n):
    K = comm_mat(n, n)
    I = sp.sparse.eye(n, format="csc", dtype=np.double)
    L = elim_mat(n)
    I2 = sp.sparse.eye(n*n, format="csc", dtype=np.double)
    N = K + I2
    LN = sp.sparse.csc_matrix.dot(L, N).astype(np.double)

    LN2 = sparse_kron(L, LN)
    Kvm = sparse_kron(sparse_kron(I, K), I)
    Iv = I.reshape(-1, 1, order='F', copy=True).tocsc()
    I2Iv=sparse_kron(I2, Iv)
    D2=LN2.dot(Kvm.dot(I2Iv.dot(L.T)))
    return LN, I, L, D2
