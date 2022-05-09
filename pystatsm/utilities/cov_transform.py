#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  5 10:42:50 2022

@author: lukepinkel
"""



import numpy as np
from .special_mats import lmat, nmat
from .numerical_derivs import jac_approx
from .dchol import dchol, unit_matrices


def _hess_chain_rule(d2z_dy2, dy_dx, dz_dy, d2y_dx2):
    H1 = np.einsum("ijk,kl->ijl", d2z_dy2, dy_dx, optimize=True)
    H1 = np.einsum("ijk,jl->ilk", H1, dy_dx, optimize=True)
    H2 = np.einsum("ij,jkl->ikl", dz_dy, d2y_dx2, optimize=True)
    d2z_dx2 = H1 + H2
    return d2z_dx2

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
    
    def _hess_fwd(self, x):
        rix, cix = self.row_diag_inds, self.col_diag_inds
        d2y_dx2 = np.zeros((self.hv_size,)*3)
        for i in range(self.hv_size):
            j, k = rix[i], cix[i]
            if j!=k:
                xi, xj, xk = x[i], x[j], x[k] #w, y, z = x[i], x[j], x[k]
                sjk =  np.sqrt(xj * xk)       #syz = np.sqrt(y * z)
                d2yi_dxidxj = -xk / (2.0 * sjk**3) #d2y_dwdy = -z / (2.0 * syz**3)
                d2yi_dxidxk = -xj / (2.0 * sjk**3) #d2y_dwdz = -y / (2.0 * syz**3)
                d2yi_dxjdxj = (3.0 * xi * xk**2)   / (4.0 * sjk**5)  #d2y_dydy = (3.0 * w * z**2)  / (4.0 * syz**5)
                d2yi_dxjdxk = (3.0 * xi * xj * xk) / (4.0 * sjk**5) - xi / (2.0 * sjk**3) #d2y_dydz = (3.0 * w * y * z) / (4.0 * syz**5) - (w) / (2.0 * syz**3)
                d2yi_dxkdxk = (3.0 * xi * xj**2)   / (4.0 * sjk**5) #d2y_dzdz = (3.0 * w * y**2)  / (4.0 * syz**5)
                
                d2y_dx2[i, i, j] = d2y_dx2[i, j, i] = d2yi_dxidxj
                d2y_dx2[i, i, k] = d2y_dx2[i, k, i] = d2yi_dxidxk
                d2y_dx2[i, j, j] = d2yi_dxjdxj
                d2y_dx2[i, j, k] = d2y_dx2[i, k, j] = d2yi_dxjdxk
                d2y_dx2[i, k, k] = d2yi_dxkdxk
        return d2y_dx2
    
    def _hess_rvs(self, y):
        rix, cix = self.row_diag_inds, self.col_diag_inds
        d2x_dy2 = np.zeros((self.hv_size,)*3)
        for i in range(self.hv_size):
            j, k = rix[i], cix[i]
            if j!=k:
               yi, yj, yk = y[i], y[j], y[k]  
               d2xi_dyidyj = np.sqrt(yk) / (2.0 * np.sqrt(yj))
               d2xi_dyidyk = np.sqrt(yj) / (2.0 * np.sqrt(yk))
               d2xi_dyjdyj = (-yi * np.sqrt(yk)) / (4.0 * np.sqrt(yj)**3)
               d2xi_dyjdyk = yi / (4.0 * np.sqrt(yj * yk))
               d2xi_dykdyk = (-yi * np.sqrt(yj)) / (4.0 * np.sqrt(yk)**3)
               
               d2x_dy2[i, i, j] = d2x_dy2[i, j, i] = d2xi_dyidyj
               d2x_dy2[i, i, k] = d2x_dy2[i, k, i] = d2xi_dyidyk
               d2x_dy2[i, j, j] = d2xi_dyjdyj
               d2x_dy2[i, j, k] = d2x_dy2[i, k, j] = d2xi_dyjdyk
               d2x_dy2[i, k, k] = d2xi_dykdyk
        return d2x_dy2
        
   
class LogScale(object):
    
    def __init__(self, mat_size):
        self.mat_size = mat_size
        self.hv_size = mat_size_to_hv_size(mat_size)
        self.row_inds, self.col_inds = hv_indices((mat_size, mat_size))
        self.diag_inds, = np.where(self.row_inds==self.col_inds)
        self.tril_inds, = np.where(self.row_inds!=self.col_inds)
    
    def _fwd(self, x):
        ix = self.diag_inds
        y = x.copy()
        y[ix] = np.log(x[ix])
        return y
    
    def _rvs(self, y):
        ix = self.diag_inds
        x = y.copy()
        x[ix] = np.exp(y[ix])
        return x
        
    def _jac_fwd(self, x):
        ix = self.diag_inds
        dy_dx = np.eye(x.shape[-1])
        dy_dx[ix, ix] = 1 / x[ix]
        return dy_dx
    
    def _jac_rvs(self, y):
        ix = self.diag_inds
        dx_dy = np.eye(y.shape[-1])
        dx_dy[ix, ix] = np.exp(y[ix])
        return dx_dy
    
    def _hess_fwd(self, x):
        ix = self.diag_inds
        d2y_dx2 = np.zeros((x.shape[-1],)*3)
        d2y_dx2[ix, ix, ix] = -1 / x[ix]**2
        return d2y_dx2
        
    def _hess_rvs(self, y):
        ix = self.diag_inds
        d2x_dy2 = np.zeros((y.shape[-1],)*3)
        d2x_dy2[ix, ix, ix] = np.exp(y[ix])
        return d2x_dy2
    
    
class CorrCholesky(object):
    
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
        self.dM, self.d2M = unit_matrices(mat_size)
        row_inds, col_inds = hv_indices((self.mat_size, self.mat_size))
        self.tril_inds, = np.where(row_inds!=col_inds)
        
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
        M = _invecl(x)
        _, dL, _ = dchol(M, self.dM, self.d2M, order=1)
        dL = dL.reshape(np.product(dL.shape[:2]), dL.shape[2], order='F')
        dy_dx = self.E.dot(dL)[np.ix_(self.tril_inds, self.tril_inds)]
        return dy_dx
    
    def _jac_rvs(self, y):
        x = self._rvs(y)
        dx_dy = np.linalg.inv(self._jac_fwd(x))
        return dx_dy
    
    def _hess_fwd(self, x):
        M = _invecl(x)
        _, _, d2L = dchol(M, self.dM, self.d2M, order=2)
        k = (np.product(d2L.shape[:2]),)
        d2L = d2L.reshape(k+d2L.shape[2:], order='F')
        d2L = d2L[self.E.tocoo().col]
        d2y_dx2 = d2L[np.ix_(self.tril_inds, self.tril_inds, self.tril_inds)]
        return d2y_dx2
    
    def _hess_rvs(self, y):
        #x = self._rvs(y)
        #M = _invecl(x)
        #L, dL, d2L = dchol(M, self.dM, self.d2M, order=2)
        #k = (np.product(d2L.shape[:2]),)
        #d2L = d2L.reshape(k+d2L.shape[2:], order='F')
        #d2L = d2L[self.E.tocoo().col]
        #d2y_dx2 = d2L[np.ix_(self.tril_inds, self.tril_inds, self.tril_inds)]
        #dL = dL.reshape(np.product(dL.shape[:2]), dL.shape[2], order='F')
        #dL = self.E.dot(dL)[np.ix_(self.tril_inds, self.tril_inds)]
        #dx_dy = np.linalg.inv(dL)
        #d2x_dy2 = -np.einsum("ijk,kl->ijl", d2y_dx2, dx_dy)
        return jac_approx(self._jac_rvs, y.copy())
        

class OffDiagMask(object):
    
    def __init__(self, transform):
        self.mat_size = transform.mat_size
        self.hv_size = mat_size_to_hv_size(self.mat_size)
        self.row_inds, self.col_inds = hv_indices((self.mat_size, self.mat_size))
        self.diag_inds, = np.where(self.row_inds==self.col_inds)
        self.tril_inds, = np.where(self.row_inds!=self.col_inds)
        self.transform = transform
        
    def _fwd(self, x):
        y = x.copy()
        y[self.tril_inds] = self.transform._fwd(y[self.tril_inds])
        return y
    
    def _rvs(self, y):
        x = y.copy()
        x[self.tril_inds] = self.transform._rvs(x[self.tril_inds])
        return x
    
    def _jac_fwd(self, x):
        dy_dx = np.zeros((self.hv_size, self.hv_size))
        ii, ij = self.diag_inds, self.tril_inds
        dy_dx[np.ix_(ij, ij)] = self.transform._jac_fwd(x[ij].copy())
        dy_dx[np.ix_(ii, ii)] = np.eye(len(ii))
        return dy_dx
    
    def _jac_rvs(self, y):
        dx_dy = np.zeros((self.hv_size, self.hv_size))
        ii, ij = self.diag_inds, self.tril_inds
        dx_dy[np.ix_(ij, ij)] = self.transform._jac_rvs(y[ij].copy())
        dx_dy[np.ix_(ii, ii)] = np.eye(len(ii))
        return dx_dy
    
    def _hess_fwd(self, x):
        d2y_dx2 = np.zeros((self.hv_size, self.hv_size, self.hv_size))
        ij = self.tril_inds
        d2y_dx2[np.ix_(ij, ij, ij)] = self.transform._hess_fwd(x[ij].copy())
        return d2y_dx2
    
    def _hess_rvs(self, y):
        d2x_dy2 = np.zeros((self.hv_size, self.hv_size, self.hv_size))
        ij = self.tril_inds
        d2x_dy2[np.ix_(ij, ij, ij)] = self.transform._hess_rvs(y[ij].copy())
        return d2x_dy2

  
    


class CholeskyCov(object):
    
    def __init__(self, mat_size):
        self.covn_to_corr = CovCorr(mat_size)
        self.corr_to_chol = OffDiagMask(CorrCholesky(mat_size))
        self.chol_to_real = OffDiagMask(UnconstrainedCholeskyCorr(mat_size))
        self.vars_to_logs = LogScale(mat_size)
        
    def _fwd(self, x):
        y = self.covn_to_corr._fwd(x)
        z = self.corr_to_chol._fwd(y)
        w = self.chol_to_real._fwd(z)
        u = self.vars_to_logs._fwd(w)
        return u
    
    def _rvs(self, u):
        w = self.vars_to_logs._rvs(u)
        z = self.chol_to_real._rvs(w)
        y = self.corr_to_chol._rvs(z)
        x = self.covn_to_corr._rvs(y)
        return x
    
    def _jac_fwd(self, x):
        y = self.covn_to_corr._fwd(x)
        z = self.corr_to_chol._fwd(y)
        w = self.chol_to_real._fwd(z)
        #u = self.vars_to_logs._fwd(w)
        
        dy_dx = self.covn_to_corr._jac_fwd(x)
        dz_dy = self.corr_to_chol._jac_fwd(y)
        dw_dz = self.chol_to_real._jac_fwd(z)
        du_dw = self.vars_to_logs._jac_fwd(w)
        dw_dx = dw_dz.dot(dz_dy).dot(dy_dx)
        du_dx = du_dw.dot(dw_dx)
        return du_dx
    
    def _jac_rvs(self, u):
        w = self.vars_to_logs._rvs(u)
        z = self.chol_to_real._rvs(w)
        y = self.corr_to_chol._rvs(z)
        
        
        dx_dy = self.covn_to_corr._jac_rvs(y)
        dy_dz = self.corr_to_chol._jac_rvs(z)
        dz_dw = self.chol_to_real._jac_rvs(w)
        dw_du = self.vars_to_logs._jac_rvs(u)
        dx_dw = dx_dy.dot(dy_dz).dot(dz_dw)
        dx_du = dx_dw.dot(dw_du)
        return dx_du
    
    
    def _hess_fwd(self, x):
        y = self.covn_to_corr._fwd(x)
        z = self.corr_to_chol._fwd(y)
        w = self.chol_to_real._fwd(z)
        dy_dx = self.covn_to_corr._jac_fwd(x)
        dz_dy = self.corr_to_chol._jac_fwd(y)
        dw_dz = self.chol_to_real._jac_fwd(z)
        du_dw = self.vars_to_logs._jac_fwd(w)
        
        dz_dx = dz_dy.dot(dy_dx)
        dw_dx = dw_dz.dot(dz_dy.dot(dy_dx))
        d2y_dx2 = self.covn_to_corr._hess_fwd(x)
        d2z_dy2 = self.corr_to_chol._hess_fwd(y)
        d2w_dz2 = self.chol_to_real._hess_fwd(z)
        d2u_dw2 = self.vars_to_logs._hess_fwd(w)
        
        d2z_dx2 = _hess_chain_rule(d2z_dy2, dy_dx, dz_dy, d2y_dx2)
        d2w_dx2 = _hess_chain_rule(d2w_dz2, dz_dx, dw_dz, d2z_dx2)
        d2u_dx2 = _hess_chain_rule(d2u_dw2, dw_dx, du_dw, d2w_dx2)
        
        # m = x.shape[-1]
        # d2z_dx2 = np.zeros((m,)*3)
        # for i in range(m):
        #     d2z_dx2[i] += dy_dx.T.dot(d2z_dy2[i]).dot(dy_dx)
        #     for j in range(m):
        #         d2z_dx2[i] += dz_dy[i, j] * d2y_dx2[j]
        # d2w_dx2 = np.zeros((m,)*3)
        # for i in range(m):
        #     d2w_dx2[i] += dz_dx.T.dot(d2w_dz2[i]).dot(dz_dx)
        #     for j in range(m):
        #         d2w_dx2[i] += dw_dz[i, j] * d2z_dx2[j]
        # d2u_dx2 = np.zeros((m,)*3)
        # for i in range(m):
        #     d2u_dx2[i] += dw_dx.T.dot(d2u_dw2[i]).dot(dw_dx)
        #     for j in range(m):
        #         d2u_dx2[i] += du_dw[i, j] * d2w_dx2[j]
        return d2u_dx2

