#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 13:04:05 2022

@author: lukepinkel
"""


import numpy as np
from .utilities.special_mats import lmat, nmat, dmat, kmat



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


class UnconstrainedCholeskyCorr(object):
    
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
    

class CovCorr(object):
    
    def __init__(self, mat_size):
        self.mat_size = mat_size
        self.hv_size = mat_size_to_hv_size(mat_size)
        self.row_inds, self.col_inds = hv_indices((mat_size, mat_size))
        self.diag_inds, = np.where(self.row_inds==self.col_inds)
        self.tril_inds, = np.where(self.row_inds!=self.col_inds)
        self.row_diag_inds = self.diag_inds[self.row_inds]
        self.col_diag_inds = self.diag_inds[self.col_inds]
        ii = self.row_diag_inds!=self.col_diag_inds
        self.dr_perm = np.vstack((self.col_diag_inds[ii], self.row_diag_inds[ii])).T.flatten()
        self.dc_perm = np.repeat(self.tril_inds, 2)
        self.ii = ii

        
    def _fwd(self, x):
        sj = np.sqrt(x[self.row_diag_inds])
        sk = np.sqrt(x[self.col_diag_inds])
        y = x / (sj * sk)
        y[self.diag_inds] = x[self.diag_inds] #np.log(x[self.diag_inds])
        return y
    
    def _rvs(self, y):
        sj = np.sqrt(y[self.row_diag_inds])
        sk = np.sqrt(y[self.col_diag_inds])
        x = y * sj * sk
        x[self.diag_inds] = y[self.diag_inds]
        return x
    
    def _jac_fwd(self, x):
        dy_dx = np.zeros((x.shape[0],)*2)
        sj = np.sqrt(x[self.row_diag_inds])
        sk = np.sqrt(x[self.col_diag_inds])
        t1 = 1 / (sj*sk)
        t2 = -1.0 / 2.0 * x / (sj**3 * sk)
        t3 = -1.0 / 2.0 * x / (sk**3 * sj)
        t = np.vstack((t3, t2)).T[self.ii].flatten()
        dy_dx[self.diag_inds, self.diag_inds] = 1.0# / x[self.diag_inds]
        dy_dx[self.tril_inds, self.tril_inds] = t1[self.tril_inds]
        dy_dx[self.dc_perm, self.dr_perm] = t
        return dy_dx
    
    def _jac_rvs(self, y):
        dx_dy = np.zeros((y.shape[0],)*2)
        sj = np.sqrt(y[self.row_diag_inds])
        sk = np.sqrt(y[self.col_diag_inds])
        t1 = (sj*sk)
        t2 = 1.0 / 2.0 * y * sk / sj
        t3 = 1.0 / 2.0 * y * sj / sk
        t = np.vstack((t3, t2)).T[self.ii].flatten()
        dx_dy[self.diag_inds, self.diag_inds] = 1.0#np.exp(y[self.diag_inds])
        dx_dy[self.tril_inds, self.tril_inds] = t1[self.tril_inds]
        dx_dy[self.dc_perm, self.dr_perm] = t
        return dx_dy
    
    
class CholeskyCorr(object):
    
    def __init__(self, mat_size):
        self.n = self.mat_size = mat_size
        self.m = self.vec_size = int((mat_size + 1) * mat_size / 2)
        self.I = np.eye(self.n)
        self.N = nmat(self.n)
        self.E = lmat(self.n)
        self.diag_inds = np.diag_indices(self.mat_size)
        self.lhv_size = mat_size_to_lhv_size(mat_size)
        self.row_inds, self.col_inds = lhv_indices((mat_size, mat_size))
        self.row_sort = np.argsort(self.row_inds)
        self.ind_parts = lhv_ind_parts(mat_size)
        self.row_norm_inds = [self.row_sort[a:b] for a, b in self.ind_parts]
        j, i = np.triu_indices(self.n)
        self.non_diag, = np.where(j!=i)
        
    def _fwd(self, x):
        R = _invecl(x)
        L = np.linalg.cholesky(R)
        y = lower_half_vec(L)
        return y
    
    def _rvs(self, y):
        L = inv_lower_half_vec(y)
        L[self.diag_inds] = np.sqrt(1-np.linalg.norm(L, axis=-1)**2)
        R = np.dot(L, L.T)
        x = lower_half_vec(R)
        return x
    
    def _jac_fwd(self, x):
        non_diag = self.non_diag
        R = _invecl(x)
        L = np.linalg.cholesky(R)
        In, Nn, Ln = self.I, self.N, self.E
        dy_dx = np.linalg.inv(Ln.dot((Ln.dot(Nn).dot(np.kron(L, In))).T)).T
        dy_dx = dy_dx[np.ix_(non_diag, non_diag)]
        return dy_dx
    
    def _jac_rvs(self, y):
        x = self._rvs(y)
        dx_dy = np.linalg.inv(self._jac_fwd(x))
        return dx_dy
        
        
    
class CholeskyCov(object):
    
    def __init__(self, mat_size):
        self.cholcorr_to_unconstrained = UnconstrainedCholeskyCorr(mat_size)
        self.corr_to_cholcorr = CholeskyCorr(mat_size)
        self.cov_to_corr = CovCorr(mat_size)
        
    def _fwd(self, x):
        ix = self.cov_to_corr.tril_inds
        y = self.cov_to_corr._fwd(x)
        y[ix] = self.corr_to_cholcorr._fwd(y[ix])
        y[ix] = self.cholcorr_to_unconstrained._fwd(y[ix])
        return y
        
    def _rvs(self, y):
        y = y.copy()
        ix = self.cov_to_corr.tril_inds
        y[ix] = self.cholcorr_to_unconstrained._rvs(y[ix])
        y[ix] = self.corr_to_cholcorr._rvs(y[ix])
        x = self.cov_to_corr._rvs(y)
        return x
    
    def _jac_fwd(self, x):
        ix = self.cov_to_corr.tril_inds
        y = self.cov_to_corr._fwd(x)
        z = self.corr_to_cholcorr._fwd(y[ix].copy())
        
        Jwz = self.cholcorr_to_unconstrained._jac_fwd(z)
        Jzy = self.corr_to_cholcorr._jac_fwd(y[ix].copy())
        Jyx = self.cov_to_corr._jac_fwd(x)
        Jwy = Jwz.dot(Jzy)
        Jwx = Jyx.copy()
        Jwx[ix] = Jwy.dot(Jwx[ix])
        return Jwx
    
    
    
        

# from pystatsm.pystatsm.utilities.numerical_derivs import jac_approx

# mat_size = 5
# lhv_size = mat_size_to_lhv_size(mat_size)

# x = np.linspace(1.0, 2.0, lhv_size)

# b = UnconstrainedCholeskyCorr(mat_size)
# y = b._fwd(x)
# L = inv_lower_half_vec(x) + np.eye(mat_size)
# y2 = lower_half_vec(L / np.linalg.norm(L, axis=-1)[:, None])
# assert(np.allclose(y, y2))


# Jf1 = jac_approx(b._fwd, x)
# Jr1 = jac_approx(b._rvs, y)
# Jf2 = b._jac_fwd(x)
# Jr2 = b._jac_rvs(y)

# assert(np.allclose(Jf1, Jf2))
# assert(np.allclose(Jr1, Jr2))

# Hf1 = jac_approx(b._jac_fwd, x)
# Hr1 = jac_approx(b._jac_rvs, y)

# Hf2 = b._hess_fwd(x)
# Hr2 = b._hess_rvs(y)

# assert(np.allclose(Hf1, Hf2))
# assert(np.allclose(Hr1, Hr2))

# mat_size = 5
# lhv_size = mat_size_to_lhv_size(mat_size)
# c = CovCorr(mat_size)

# x = np.linspace(1.0, 2.0, lhv_size)

# b = UnconstrainedCholeskyCorr(mat_size)
# y = b._fwd(x)
# L = inv_lower_half_vec(x) + np.eye(mat_size)
# L = L / np.linalg.norm(L, axis=-1)[:, None]
# R = L.dot(L.T)
# S = np.sqrt(np.diag(np.arange(2, 2+mat_size)))
# V = S.dot(R).dot(S) 
# x = _vech(V)
# c = CovCorr(mat_size)

# y = c._fwd(x)
# x = c._rvs(y)

# assert(np.allclose(c._rvs( c._fwd(x)), x))
# assert(np.allclose(c._fwd( c._rvs(y)), y))




# Jf1 = jac_approx(c._fwd, x)
# Jr1 = jac_approx(c._rvs, y)

# Jf2 = c._jac_fwd(x)
# Jr2 = c._jac_rvs(y)

# assert(np.allclose(Jf1, Jf2))
# assert(np.allclose(Jr1, Jr2))



# mat_size = 5
# lhv_size = mat_size_to_lhv_size(mat_size)
# x = np.linspace(1.0, 2.0, lhv_size)

# b = UnconstrainedCholeskyCorr(mat_size)
# y = b._fwd(x)
# L = inv_lower_half_vec(x) + np.eye(mat_size)
# L = L / np.linalg.norm(L, axis=-1)[:, None]
# R = L.dot(L.T)


# t1 = CholeskyCorr(mat_size)
# x = lower_half_vec(R)
# y = t1._fwd(x)

# assert(np.allclose(t1._rvs( t1._fwd(x)), x))
# assert(np.allclose(t1._fwd( t1._rvs(y)), y))


# Jf1 = jac_approx(t1._fwd, x)
# Jr1 = jac_approx(t1._rvs, y)
# Jf2 = t1._jac_fwd(x)
# Jr2 = t1._jac_rvs(y)
# assert(np.allclose(Jf1, Jf2, atol=1e-4))
# assert(np.allclose(Jr1, Jr2, atol=1e-4))



# mat_size = 5
# lhv_size = mat_size_to_lhv_size(mat_size)
# c = CovCorr(mat_size)

# x = np.linspace(1.0, 2.0, lhv_size)

# b = UnconstrainedCholeskyCorr(mat_size)
# y = b._fwd(x)
# L = inv_lower_half_vec(x) + np.eye(mat_size)
# L = L / np.linalg.norm(L, axis=-1)[:, None]
# R = L.dot(L.T)
# S = np.sqrt(np.diag(np.arange(2, 2+mat_size)))
# V = S.dot(R).dot(S) 
# x = _vech(V)

# t = CholeskyCov(mat_size)
# y = t._fwd(x)

# assert(np.allclose(t._rvs( t._fwd(x)), x))
# assert(np.allclose(t._fwd( t._rvs(y)), y))


