#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 16 21:47:11 2020

@author: lukepinkel
"""

import numba 
import numpy as np 
import scipy as sp 
from sksparse.cholmod import cholesky

def wcrossp(X, w):
    Y =  (X * w[:, np.newaxis]).T.dot(X)
    return Y

def wdcrossp(X, w, Y):
    Y =  (X * w[:, np.newaxis]).T.dot(Y)
    return Y


def add_chol_row(A, L, i):
    x, r = A[i, :i], A[i, i]
    if i>0:
        b = sp.linalg.solve_triangular(L[:i, :i], x, lower=True, check_finite=False)
        L[i, :i] = b
    else:
        b = L[i, :i]
    s = np.dot(b.T, b)
    L[i, i] = np.sqrt(r-s)
    return L
    

@numba.jit(nopython=True)
def chol_downdate(L, k):
    n = L.shape[0]
    ix = np.arange(n)!=k
    L1 = L.copy()[ix]
    for t in range(k, n-1):
        a, b = L1[t, t], L1[t, t+1]
        v = np.sqrt(a**2+b**2)
        c, s = a / v, b / v
        for i in range(t, n-1):
            Lit = L1[i, t]
            Lit1 = L1[i, t+1]
            L1[i, t] = c * Lit + s * Lit1
            L1[i, t+1] = c * Lit1 - s * Lit
        L1[t, t] = v
        L1[t, t+1] = 0.0
    #L1 = L1[:, :-1]
    #return L1
    L[:-1, :-1] = L1[:, :-1]
    return L


@numba.jit(nopython=True)
def toeplitz_cholesky_lower_nb(n, A):
    g = np.zeros((2, n), dtype=np.double)
    for j in range(0, n):
        g[0, j] = A[j, 0]
    for j in range(1, n):
        g[1, j] = A[j, 0]
    L = np.zeros((n, n), dtype=np.double)
    for j in range(0, n):
        L[j, 0] = g[0, j]
    for j in range(n-1, 0, -1):
        g[0, j] = g[0, j-1]
    g[0, 0] = 0.0
    for i in range(1, n):
        rho = -g[1, i] / g[0, i]
        gamma = np.sqrt((1.0 - rho) * (1.0 + rho))
        for j in range(i, n):
            alpha = g[0, j]
            beta = g[1, j]
            g[0, j] = (alpha + rho * beta) / gamma
            g[1, j] = (rho * alpha + beta) / gamma
        for j in range(i, n):
            L[j, i] = g[0, j]
        for j in range(n-1, i, -1):
            g[0, j] = g[0, j-1]
        g[0, i] = 0.0
    return L


@numba.jit(nopython=True)
def ar1_chol(n, rho):
    L = np.zeros((n, n))
    L[:, 0] = rho**np.arange(n)
    L[1:, 1] = L[:-1, 0] * np.sqrt(1.0 - rho**2)
    for i in range(2, n):
        L[i:, i] = L[1:1-i, 1]
    return L



def vec(X):
    return X.reshape(-1, order='F')


def invec(x, n_rows, n_cols):
    return x.reshape(int(n_rows), int(n_cols), order='F')


@numba.jit(nopython=True)
def vech(X):
    p = X.shape[0]
    tmp =  1 - np.tri(p, p, k=-1)
    tmp2 = tmp.flatten()
    ix = tmp2==1
    Y = X.T.flatten()[ix]
    return Y

@numba.jit(nopython=True)
def invech(v):
    '''
    Inverse half vectorization operator
    '''
    rows = int(np.round(.5 * (-1 + np.sqrt(1 + 8 * len(v)))))
    res = np.zeros((rows, rows))
    tmp =  1 - np.tri(rows, rows, k=-1)
    tmp2 = tmp.flatten()
    ix = tmp2==1
    Y = res.T.flatten()
    Y[ix] = v
    Y = Y.reshape(rows, rows)
    Y = Y + Y.T
    Y = Y - (np.eye(rows) * Y) / 2
    return Y

@numba.jit(nopython=True)
def vecl(X):
    p = X.shape[0]
    tmp =  1 - np.tri(p, p)
    tmp2 = tmp.flatten()
    ix = tmp2==1
    Y = X.T.flatten()[ix]
    return Y

@numba.jit(nopython=True)
def invecl(v):
    rows = int(np.round(.5 * (1 + np.sqrt(1 + 8 * len(v)))))
    res = np.zeros((rows, rows))
    tmp =  1 - np.tri(rows, rows)
    tmp2 = tmp.flatten()
    ix = tmp2==1
    Y = res.T.flatten()
    Y[ix] = v
    Y = Y.reshape(rows, rows)
    Y = Y + Y.T
    Ir = np.eye(rows)
    Y = Y - (Ir * Y) + Ir
    return Y


def invech_chol(lvec):
    p = int(0.5 * ((8*len(lvec) + 1)**0.5 - 1))
    L = np.zeros((p, p))
    a, b = np.triu_indices(p)
    L[(b, a)] = lvec
    return L



@numba.jit(nopython=True)
def _vech_nb(x):
    m = x.shape[-1]
    s, r = np.triu_indices(m, k=0)
    i = r+s*m
    res = x.T.flatten()[i]
    return res


def vdg(X):
    V = np.diag(vec(X))
    return V

     
def gb_diag(*arrs):
    shapes = np.array([arr.shape for arr in arrs])
    res = np.zeros(np.sum(shapes, axis=0))
    ix = np.zeros(len(res.shape), dtype=int)
    for i, k in enumerate(shapes):
        s = [slice(a, a+b) for a, b in list(zip(ix, k))]
        res[tuple(s)] = arrs[i]
        ix += k
    return res


def wls_chol(X, y, w):
    Xwt = X.T * w
    L = np.linalg.cholesky(Xwt.dot(X))
    c = sp.linalg.solve_triangular(L, Xwt.dot(y), lower=True)
    b = sp.linalg.solve_triangular(L.T, c, lower=False)
    return b

def wls_qr(X, y, w):
    wsqr = np.sqrt(w)
    Q, R = np.linalg.qr(X * wsqr[:, None])
    c = Q.T.dot(wsqr * y)
    b = sp.linalg.solve_triangular(R, c, lower=False)
    return b

def nwls(X, y, w):
    neg = w < 0
    w_neg = np.zeros_like(w)
    w_neg[neg] = -w[neg]
    w_sqr = np.sqrt(np.abs(w))
    
    Q, R = np.linalg.qr((X * w_sqr[:, None]))
    non_pos = (w <= 0)
    U, s, Vt = np.linalg.svd(Q * non_pos[:, None], full_matrices=False)
    neg = neg * -1.0 + (~neg) * 1.0
    Qtz = Q.T.dot(w_sqr * neg * y)
    a = 1.0 / (1.0  - 2.0 * s**2)
    c = Vt.T.dot(a * Vt.dot(Qtz))
    b = sp.linalg.solve_triangular(R, c, lower=False)
    return b

def lsqr(X, y):
    Q, R = np.linalg.qr(X, mode='reduced')
    b = np.linalg.solve(R, Q.T.dot(y))
    return b

def diag_outer_prod(A, B):
    v = np.einsum("ij,ij->i", A, B, optimize=True)
    return v

def wdiag_outer_prod(X, W, Y):
    v = np.einsum("ij,jk,ik->i", X, W, Y, optimize=True)
    return v

def cproject(A):
    "project onto column space"
    #U, _, _ = np.linalg.svd(A, full_matrices=False)
    #P = U.dot(U.T)
    P = A.dot(np.linalg.pinv(A))
    return P

def eighs(A):
    u, V = np.linalg.eigh(A)
    u, V = u[::-1], V[:, ::-1]
    return u, V    

def inv_sqrt(arr):
    u, V  = eighs(arr)
    u[u>1e-12] = 1.0 / np.sqrt(u[u>1e-12])
    arr = (V * u).dot(V.T)
    return arr

def mat_sqrt(arr):
    u, V  = eighs(arr)
    u[u>1e-12] = np.sqrt(u[u>1e-12])
    arr = (V * u).dot(V.T)
    return arr



@numba.jit(nopython=True)
def _solve_vander(a, x, n):
    for k in range(n-1):
        x[k+1 : n] -= a[k] * x[k : n-1]
    for k in range(n-1, 0, -1):
        x[k:n] /= a[k:n] - a[:n-k]
        x[k-1 : n-1] -= x[k:n]
    return x

def solve_vander(a, b):
    x = b.copy()
    n = a.size
    return _solve_vander(a, x, n)

@numba.jit(nopython=True)
def _inv_vander(a):
    n = len(a)
    B = np.zeros((n, n))
    for i in range(n):
        B[i, n - i - 1] = 1.0
        _solve_vander(a, B[i], n)
    return B


def _vec(x):
    old_shape = x.shape
    new_shape = old_shape[:-2] + (np.prod(old_shape[-2:]),)
    y = x.reshape(new_shape, order='F')
    return y

def _invec(x, n, m):
    old_shape = x.shape
    new_shape = old_shape[:-1] + (n, m)
    y = x.reshape(new_shape, order='F')
    return y

def _vech(x):
    m = x.shape[-1]
    ix, jx = np.triu_indices(m, k=0)
    res = x[...,jx, ix]
    return res

def _invech(x):
    old_shape = x.shape
    n = x.shape[-1]
    m = int((np.sqrt(8 * n + 1) - 1) // 2)
    out_shape = old_shape[:-1] + (m, m)
    res = np.zeros(out_shape, dtype=x.dtype)
    ix, jx = np.triu_indices(m, k=0)
    is_diag = ix == jx
    diag_elements = x[..., is_diag]
    off_diag_elem = x[...,~is_diag]
    ixo, jxo = ix[~is_diag], jx[~is_diag]
    ixd, jxd = ix[is_diag],  jx[is_diag]
    res[..., jxo, ixo] = off_diag_elem
    res[..., ixo, jxo] = off_diag_elem
    res[..., ixd, jxd] = diag_elements
    return res

def _vecl(x):
    m = x.shape[-1]
    ix, jx = np.triu_indices(m, k=1)
    res = x[...,jx, ix]
    return res

def _invecl(x):
    old_shape = x.shape
    n = x.shape[-1]
    m = int((np.sqrt(8 * n + 1) + 1) // 2)
    out_shape = old_shape[:-1] + (m, m)
    res = np.zeros(out_shape, dtype=x.dtype)
    ix, jx = np.triu_indices(m, k=1)
    is_diag = ix == jx
    off_diag_elem = x[...,~is_diag]
    ixo, jxo = ix[~is_diag], jx[~is_diag]
    ixd = jxd = np.arange(m)
    res[..., jxo, ixo] = off_diag_elem
    res[..., ixo, jxo] = off_diag_elem
    res[..., ixd, jxd] = 1
    return res


def lhv_size_to_mat_size(lhv_size):
    mat_size = int((np.sqrt(8 * lhv_size + 1) + 1) // 2)
    return mat_size

def mat_size_to_lhv_size(mat_size):
    lhv_size = int(mat_size * (mat_size - 1) // 2)
    return lhv_size

def hv_size_to_mat_size(lhv_size):
    mat_size = int((np.sqrt(8 * lhv_size + 1) - 1) // 2)
    return mat_size

def mat_size_to_hv_size(mat_size):
    lhv_size = int(mat_size * (mat_size + 1) // 2)
    return lhv_size

def lower_half_vec(x):
    mat_size = x.shape[-1]
    i, j = np.triu_indices(mat_size, k=1)
    y = x[...,j, i]
    return y

def inv_lower_half_vec(y):
    old_shape = y.shape
    lhv_size = y.shape[-1]
    mat_size = lhv_size_to_mat_size(lhv_size)
    out_shape = old_shape[:-1] + (mat_size, mat_size)
    x = np.zeros(out_shape, dtype=y.dtype)
    i, j = np.triu_indices(mat_size, k=1)
    x[..., j, i] = y
    return x

def lhv_indices(shape):
    arr_inds = np.indices(shape)
    lhv_inds = [lower_half_vec(x) for x in arr_inds]
    return lhv_inds

def hv_indices(shape):
    arr_inds = np.indices(shape)
    hv_inds = [_vech(x) for x in arr_inds]
    return hv_inds


def lhv_ind_parts(mat_size):
    i = np.cumsum(np.arange(mat_size))
    return list(zip(*(i[:-1], i[1:])))

def lhv_row_norms(y):
    lhv_size = y.shape[-1]
    mat_size = lhv_size_to_mat_size(lhv_size)
    r, c = lhv_indices((mat_size, mat_size))
    rj = np.argsort(r)
    ind_partitions = lhv_ind_parts(mat_size)
    row_norms = np.zeros(mat_size)
    row_norms[0] = 1.0
    for i, (a, b) in enumerate(ind_partitions):
        ii = rj[a:b]
        row_norms[i+1] = np.sqrt(np.sum(y[ii]**2)+1)
    return row_norms


class LHV(object):
    
    def __init__(self, mat_size):
        self.mat_size = mat_size
        self.lhv_size = mat_size_to_lhv_size(mat_size)
        self.row_inds, self.col_inds = lhv_indices((mat_size, mat_size))
        self.row_sort = np.argsort(self.row_inds)
        self.ind_parts = lhv_ind_parts(mat_size)
        self.row_norm_inds = [self.row_sort[a:b] for a, b in self.ind_parts]
    
    def _fwd(self, x):
        row_norms = np.zeros_like(x)
        for ii in self.row_norm_inds:
            row_norms[ii] = np.sqrt(np.sum(x[ii]**2)+1)
        y = x / row_norms
        return y
    
    def _rvs(self, y):
        diag = np.zeros_like(y)
        for ii in self.row_norm_inds:
            diag[ii] = np.sqrt(1-np.sum(y[ii]**2))
        x = y / diag
        return x
    
    def _jac_fwd(self, x):
        dy_dx = np.zeros((x.shape[0],)*2)
        for ii in self.row_norm_inds:
            xii = x[ii]
            s = np.sqrt(np.sum(xii**2)+1)
            v1 = 1.0 / s * np.eye(len(ii))
            v2 = 1.0 / (s**3) * xii[:, None] * xii[:,None].T
            dy_dx[ii, ii[:, None]] = v1 - v2
        return dy_dx
    
    def _hess_fwd(self, x):
        d2y_dx2 = np.zeros((x.shape[0],)*3)
        for ii in self.row_norm_inds:
            x_ii = x[ii]
            s = np.sqrt(1.0 + np.sum(x_ii**2))
            s3 = s**3
            s5 = s**5
            for i in ii:
                for j in ii:
                    for k in ii:
                        t1 = -1.0*(j==k) / s3 * x[i]
                        t2 = -1.0*(j==i) / s3 * x[k]
                        t3 = -1.0*(k==i) / s3 * x[j]
                        t4 = 3.0 / (s5) * x[j] * x[k] * x[i]
                        d2y_dx2[i, j, k] = t1+t2+t3+t4
        return d2y_dx2
                        
    def _jac_rvs(self, y):
        dx_dy = np.zeros((y.shape[0],)*2)
        for ii in self.row_norm_inds:
            yii = y[ii]
            s = np.sqrt(1.0 - np.sum(yii**2))
            v1 = 1.0 / s * np.eye(len(ii))
            v2 = 1.0 / (s**3) * yii[:, None] * yii[:,None].T
            dx_dy[ii, ii[:, None]] = v1 + v2
        return dx_dy
    
    def _hess_rvs(self, y):
        d2x_dy2 = np.zeros((y.shape[0],)*3)
        for ii in self.row_norm_inds:
            y_ii = y[ii]
            s = np.sqrt(1.0 - np.sum(y_ii**2))
            s3 = s**3
            s5 = s**5
            for i in ii:
                for j in ii:
                    for k in ii:
                        t1 = 1.0*(j==k) / s3 * y[i]
                        t2 = 1.0*(j==i) / s3 * y[k]
                        t3 = 1.0*(k==i) / s3 * y[j]
                        t4 = 3.0 / s5 * y[i] * y[j] * y[k]
                        d2x_dy2[i, j, k] = t1 + t2 + t3 + t4
        return d2x_dy2
    

def dmat_exp(A):
    n = A.shape[0]
    u, V = np.linalg.eig(A)
    arr = np.zeros((n * n), dtype=float)
    eu = np.exp(u)
    for i, j in np.ndindex((n, n)):
        k = n * i + j
        if i!=j:
            arr[k] = (eu[i] - eu[j]) / (u[i] - u[j])
        else:
            arr[k] = eu[i]        
    W = np.kron(V, V)
    J = (W * arr).dot(W.T)
    return J


  
def _sparse_post_mult(A, S):
    prod = S.T.dot(A.T)
    prod = prod.T
    return prod
