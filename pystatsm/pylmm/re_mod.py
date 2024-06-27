#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 14:11:35 2024

@author: lukepinkel
"""
import patsy
import pandas as pd
import numpy as np
import formulaic
import scipy as sp
from ..utilities.cs_kron import (coo_to_csc, sparse_kron, sparse_dense_kron, 
                                 tile_1d, cs_add_inplace, csc_matmul)
from ..utilities.linalg_operations import (invech_chol, 
                                           _invech, invech,
                                           _vech, vech,
                                           _vec)
from ..utilities.formula import parse_random_effects


# def make_zinds(n_ob, n_rv, cols):
#     z_rows = np.repeat(np.arange(n_ob), n_rv)
#     z_cols = np.repeat(cols * n_rv, n_rv)  + np.tile(np.arange(n_rv), n_ob)
#     return z_rows, z_cols

# def make_ginds_bdiag(n_rv, n_lv):
#     gdim = n_rv * n_lv
#     g_cols = np.repeat(np.arange(gdim), n_rv)
#     g_rows = np.repeat(np.arange(n_lv)*n_rv, n_rv * n_rv) + np.tile(np.arange(n_rv), gdim)
#     return g_rows, g_cols

# def make_remod_mat(arr1, arr2, return_array=False):
#     cols, u = pd.factorize(arr1, sort=False)
#     n_lv = len(u)
#     n_ob, n_rv = arr2.shape
#     z_rows, z_cols = make_zinds(n_ob, n_rv, cols)
#     z_data = arr2.reshape(-1, order='C')
#     z_size = (n_ob, n_lv * n_rv)
#     Z = coo_to_csc(z_rows, z_cols, z_data, z_size, return_array=True)
#     return Z

# def make_recov_mat(n_rv, n_lv, a_cov=None):
#     if a_cov is None:
#         a_cov = sp.sparse.eye(n_lv, format="csc") 
#         bdiag = True
#     else:
#         bdiag = False
#     G0 = np.eye(n_rv)
#     g_data = np.tile(G0.reshape(-1, order='F'), n_lv)
#     if bdiag:
#         g_rows, g_cols = make_ginds_bdiag(n_rv, n_lv)
#         G = coo_to_csc(g_rows, g_cols, g_data, (n_rv*n_lv, n_rv*n_lv), return_array=True)
#     else:
#         G = sparse_dense_kron(a_cov, G0)
#     return G, bdiag
    
from sksparse.cholmod import cholesky
import sksparse


def logpdet(arr):
    u = np.linalg.eigvalsh(arr)
    pld = np.sum(np.log(u[u>0]))
    return pld

def sparsity(a):
    return a.nnz / np.prod(a.shape)

def make_zinds(n_ob, n_rv, cols):
    z_rows = np.repeat(np.arange(n_ob), n_rv)
    z_cols = np.repeat(cols * n_rv, n_rv)  + np.tile(np.arange(n_rv), n_ob)
    return z_rows, z_cols

def make_ginds_bdiag(n_rv, n_lv):
    gdim = n_rv * n_lv
    g_cols = np.repeat(np.arange(gdim), n_rv)
    g_rows = np.repeat(np.arange(n_lv)*n_rv, n_rv * n_rv) + np.tile(np.arange(n_rv), gdim)
    return g_rows, g_cols

def make_remod_mat(arr1, arr2, return_array=False):
    cols, u = pd.factorize(arr1, sort=False)
    n_lv = len(u)
    n_ob, n_rv = arr2.shape
    z_rows, z_cols = make_zinds(n_ob, n_rv, cols)
    z_data = arr2.reshape(-1, order='C')
    z_size = (n_ob, n_lv * n_rv)
    Z = coo_to_csc(z_rows, z_cols, z_data, z_size, return_array=True)
    return Z, n_rv, n_lv

def d_recov_icov(n_rv, n_lv, g_rows, g_cols):
    m = int(n_rv * (n_rv + 1) // 2)
    d_theta = np.zeros(m)
    G_deriv = []

    for i in range(m):
        d_theta[i] = 1
        dG_theta = _vec(_invech(d_theta))
        dGi = sp.sparse.csc_matrix((np.tile(dG_theta, n_lv), (g_rows, g_cols)))
        dGi.eliminate_zeros()
        G_deriv.append(dGi)
        d_theta[i] = 0
    
    return G_deriv

def make_recov_icov(n_rv, n_lv, make_derivs=True):
    G0 = np.eye(n_rv)
    theta_init = _vech(G0)
    g_data = np.tile(G0.reshape(-1, order='F'), n_lv)
    
    g_rows, g_cols = make_ginds_bdiag(n_rv, n_lv)
    G = coo_to_csc(g_rows, g_cols, g_data, (n_rv*n_lv, n_rv*n_lv), return_array=True)
    
    G_deriv = d_recov_icov(n_rv, n_lv, g_rows, g_cols) if make_derivs else []
    
    return G, True, G_deriv, theta_init

def d_recov_acov(n_rv, a_cov):
    m = int(n_rv * (n_rv + 1) // 2)
    d_theta = np.zeros(m)
    G_deriv = []

    for i in range(m):
        d_theta[i] = 1
        dG0 = _invech(d_theta)
        dGi = sparse_dense_kron(a_cov, dG0)
        dGi.eliminate_zeros()
        G_deriv.append(dGi)
        d_theta[i] = 0
    
    return G_deriv

def make_recov_acov(n_rv, n_lv, a_cov, make_derivs=False):
    G0 = np.eye(n_rv)
    theta_init = _vech(G0)
    
    G = sparse_dense_kron(a_cov, G0)
    
    G_deriv = d_recov_acov(n_rv, a_cov) if make_derivs else []

    return G, False, G_deriv, theta_init


def make_recov_mat(n_rv, n_lv, a_cov=None, make_derivs=False):
    if a_cov is None:
        return make_recov_icov(n_rv, n_lv, make_derivs)
    else:
        return make_recov_acov(n_rv, n_lv, a_cov, make_derivs)


class RandomEffectTerm:
    def __init__(self, re_arr, gr_arr, a_cov=None):
        self.level_indices, self.levels = pd.factorize(gr_arr, sort=False)
        self.n_levels = len(self.levels)
        self.n_obs, self.n_revars = re_arr.shape
        self.n_ranefs = self.n_revars * self.n_levels 
        
        self.G, self.iid_level_cov, self.dG, self.theta_init = make_recov_mat(
            self.n_revars, self.n_levels, a_cov, True)
        
        self.Z = self._make_z_matrix(re_arr)
        self.n_gcovnz = self.G.nnz
        
        self.a_cov = a_cov
        if a_cov is not None:
            self.a_inv = np.linalg.pinv(a_cov) 
            self.lda_const = logpdet(self.a_cov) * self.n_revars
        else:
            self.a_inv = None
            self.lda_const = 0.0

        # Pre-allocate arrays for G data
        self.g_data = self.G.data
        self.g_indices = self.G.indices
        self.g_indptr = self.G.indptr


    @classmethod
    def from_formula(cls, re_formula, re_grouping, data, a_cov=None):
        re_arr = patsy.dmatrix(re_formula, data=data, return_type='dataframe').values
        gr_arr = data[re_grouping].values
        return cls(re_arr, gr_arr, a_cov)
        
    def _make_z_matrix(self, re_arr):
        z_rows, z_cols = make_zinds(self.n_obs, self.n_revars, self.level_indices)
        z_data = re_arr.reshape(-1, order='C')
        z_size = (self.n_obs, self.n_ranefs)
        return coo_to_csc(z_rows, z_cols, z_data, z_size, return_array=True)


    def _update_gdata_icov(self, G0, inv, out):
        g = G0.reshape(-1, order='F')
        tile_1d(g, self.n_levels, out=out)
        
    def _update_gdata_acov(self, G0, inv, out):
        A = self.a_inv if inv else self.a_cov
        sparse_dense_kron(A, G0, out=(out, self.g_indices, self.g_indptr))

    def update_gdata(self, theta, inv=False, out=None):
        G0 = invech(theta)
        if inv:
            G0 = np.linalg.inv(G0)
        
        out = self.g_data if out is None else out
        
        if self.iid_level_cov:
            self._update_gdata_icov(G0, inv, out)
        else:
            self._update_gdata_acov(G0, inv, out)
        return out
    
    def update_gcov(self, theta, inv=False, G=None):
        G = self.G if G is None else G
        out = G.data
        self.update_gdata(theta, inv, out)
        return G
    
    def get_logdet(self, theta, out=0.0):
        G0 = invech(theta)
        out += self.n_levels * np.linalg.slogdet(G0)[1] + self.lda_const
        return out

class RandomEffects:
    def __init__(self, re_terms, data, a_covs=None):
        self._initialize_terms(re_terms, data, a_covs)
        self._setup_attributes()

    def _initialize_terms(self, re_terms, data, a_covs):
        a_covs = [None] * len(re_terms) if a_covs is None else a_covs
        terms = []
        for (fr, gr), a in zip(re_terms, a_covs):
            terms.append(RandomEffectTerm.from_formula(fr, gr, data, a))
        self.terms = terms

    def _setup_attributes(self):
        self.n_terms = len(self.terms)
        self.n_levels = np.array([term.n_levels for term in self.terms], dtype=np.int32)
        self.n_revars = np.array([term.n_revars for term in self.terms], dtype=np.int32)
        self.n_ranefs = np.array([term.n_ranefs for term in self.terms], dtype=np.int32)
        self.n_gcovnz = np.array([term.n_gcovnz for term in self.terms], dtype=np.int32)
        
        self.n_pars = self.n_revars * (self.n_revars + 1) // 2
        self.ranef_sl = np.r_[0, self.n_ranefs.cumsum()]
        self.theta_sl = np.r_[0, self.n_pars.cumsum()]
        self.gdata_sl = np.r_[0, self.n_gcovnz.cumsum()]
        
        self.theta = np.concatenate([term.theta_init for term in self.terms])
        self.Z = sp.sparse.hstack([term.Z for term in self.terms], format='csc')
        self.G = sp.sparse.block_diag([term.G for term in self.terms], format='csc')
        self.g_data = self.G.data
        
    def update_gdata(self, theta, inv=False, out=None):
        out =  self.g_data if out is None else out
        for i, term in enumerate(self.terms):
            theta_i = theta[self.theta_sl[i]:self.theta_sl[i+1]]
            out_i = out[self.gdata_sl[i]:self.gdata_sl[i+1]]
            term.update_gdata(theta_i, inv, out_i) 
        return out
    
    def update_gcov(self, theta, inv=False, G=None):
        G = self.G if G is None else G
        out = G.data
        self.update_gdata(theta, inv, out)
        return G

    def lndet_gmat(self, theta, out=0.0):
        for i, term in enumerate(self.terms):
            theta_i = theta[self.theta_sl[i]:self.theta_sl[i+1]]
            out = term.get_logdet(theta_i, out)
        return out

class BaseMME:
    def __init__(self, X, y, re_mod, R=None):
        self.Z = re_mod.Z
        self.X = X
        self.y = y
        self.G = re_mod.G
        self.R = R 
        self.re_mod = re_mod
        
        self.n_ranef = self.Z.shape[1]
        self.n_fixef = X.shape[1]
        self.n_obs = y.shape[0]
        
        self._initialize_matrices()
        self._setup_cholesky()
    
    def _initialize_matrices(self):
        raise NotImplementedError("Subclasses must implement this method")
    
    def _setup_cholesky(self):
        raise NotImplementedError("Subclasses must implement this method")
    
    def update_crossprods(self, theta):
        raise NotImplementedError("Subclasses must implement this method")
    
    def update_chol(self, theta):
        raise NotImplementedError("Subclasses must implement this method")
    
    def update_chol_diag(self, theta):
        raise NotImplementedError("Subclasses must implement this method")
    
    def _loglike(self, theta, reml=True):
        l = self.update_chol_diag(theta)
        ytPy = l[-1]**2
        logdetG = self.re_mod.lndet_gmat(theta)
        logdetR = np.log(theta[-1]) * self.n_obs
        if reml:
            logdetC = np.sum(2 * np.log(l[:-1]))
            ll = logdetR + logdetC + logdetG + ytPy
        else:
            logdetV = np.sum(2 * np.log(l[:self.n_ranef]))
            ll = logdetR + logdetV + logdetG + ytPy
        return ll
    
    
class MMEBlocked(BaseMME):
    def __init__(self, X, y, re_mod, R=None, sparse_threshold=0.4):
        self.Zt = re_mod.Z.T.tocsc()
        self.Xy = np.hstack([X, y])
        self.sparse_threshold = sparse_threshold
        super().__init__(X, y, re_mod, R)
    
    def _initialize_matrices(self):
        if self.R is None:
            self._initialize_unweighted()
        else:
            self._initialize_weighted()
        self.C = self.ZtRZ + self.G
    
    def _initialize_unweighted(self):
        self.ZtR = None
        self.ZtZ = self.Zt.dot(self.Z)
        self.ZtRZ = self.ZtZ.copy()
        self.ZtXy = self.Zt.dot(self.Xy)
        self.ZtRXy = self.ZtXy.copy()
        self.XytXy = self.Xy.T.dot(self.Xy)
        self.XytRXy = self.XytXy.copy()
    
    def _initialize_weighted(self):
        self.ZtR = self.Zt.dot(self.R).tocsc()
        self.ZtZ = None
        self.ZtRZ = self.ZtR.dot(self.Z).tocsc()
        self.ZtXy = None
        self.ZtRXy = self.ZtR.dot(self.Xy)
        self.XytXy = None
        self.XytRXy = self.Xy.T.dot(self.R.dot(self.Xy))
    
    def _setup_cholesky(self):
        self.chol_fac = sksparse.cholmod.analyze(sp.sparse.csc_matrix(self.C), ordering_method="best")
        self._p = np.argsort(self.chol_fac.P())
        self.dg = np.zeros(self.n_fixef + self.n_ranef + 1, dtype=np.double)
        
    def update_crossprods(self, theta):
        if self.R is None:
            self._update_crossprods_scalar(theta)
        else:
            self._update_crossprods_nontrv(theta)
    
    def _update_crossprods_scalar(self, theta):
        scale = 1 / theta[-1]
        self.ZtRZ = self.ZtZ * scale
        self.ZtRXy = self.ZtXy * scale
        self.XytRXy = self.XytXy * scale
        
    def _update_crossprods_nontrv(self, theta):
        self._update_rcov(theta)
        csc_matmul(self.Zt, self.R, self.ZtR)
        csc_matmul(self.ZtR, self.Z, self.ZtRZ)
        self.ZtRXy = self.ZtR.dot(self.Xy)
        self.XytRXy = self.Xy.T.dot(self.R.dot(self.Xy))
    
    def _update_rcov(self, theta):
        # Implement this method to update R based on theta
        raise NotImplementedError("_update_rcov method needs to be implemented")
    
    def update_chol(self, theta):
        self.update_crossprods(theta)
        Ginv = self.re_mod.update_gcov(theta, inv=True, G=self.G)
        cs_add_inplace(self.ZtRZ, Ginv, self.C)
        
        if sparsity(self.C) < self.sparse_threshold:
            return self._chol_sparse(self.C)
        else:
            return self._chol_dense(self.C)
    
    def _chol_sparse(self, C):
        self.chol_fac.cholesky_inplace(sp.sparse.csc_matrix(C))
        L11 = self.chol_fac.L()[self._p,:][:,self._p].toarray()
        L21 = self.chol_fac.apply_Pt(
                self.chol_fac.solve_L(
                    self.chol_fac.apply_P(self.ZtRXy), False)).T
        L22 = np.linalg.cholesky(self.XytRXy - L21.dot(L21.T))
        return L11, L21, L22
    
    def _chol_dense(self, C):
        C_dense = C.toarray()
        L11 = np.linalg.cholesky(C_dense)
        L21 = sp.linalg.solve_triangular(L11, self.ZtRXy, trans=0, lower=True).T
        L22 = np.linalg.cholesky(self.XytRXy - L21.dot(L21.T))
        return L11, L21, L22
    
    def update_chol_diag(self, theta, out=None):
        out = self.dg if out is None else out
        L11, L21, L22 = self.update_chol(theta)
        out[:self.n_ranef] = np.diag(L11)
        out[self.n_ranef:] = np.diag(L22)
        return out
        
class MMEUnBlocked(BaseMME):
    def __init__(self, X, y, re_mod, R=None):
        super().__init__(X, y, re_mod, R)
    
    def _initialize_matrices(self):
        ZXyt = sp.sparse.hstack([self.Z, sp.sparse.csc_array(self.X), 
                                 sp.sparse.csc_array(self.y)], format='csr').T
        self.M = ZXyt.dot(ZXyt.T)
        self.M_work = self.M.copy()
        self.G_aug = sp.sparse.block_diag([self.G, sp.sparse.eye(self.n_fixef + 1) * 0.0], format="csc")
    
    def _setup_cholesky(self):
        self.chol_fac = sksparse.cholmod.analyze(sp.sparse.csc_matrix(self.M_work + self.G_aug))
    
    def update_crossprods(self, theta):
        self.M_work = self.M.copy()
        self.M_work.data /= theta[-1]
    
    def update_chol(self, theta):
        self.update_crossprods(theta)
        G_aug = self.re_mod.update_gcov(theta, inv=True, G=self.G_aug)
        M = cs_add_inplace(self.M_work, G_aug, self.M_work)
        return self.mme_chol(M)
    
    def mme_chol(self, M, use_sparse=True, sparse_threshold=0.4):
        if (sparsity(M) < sparse_threshold) and use_sparse:
            M = sp.sparse.csc_matrix(M)
            self.chol_fac.cholesky_inplace(M)
            L = self.chol_fac.L().toarray()
        else:
            L = np.linalg.cholesky(M.toarray())
        return L
    
    def update_chol_diag(self, theta):
        L = self.update_chol(theta)
        return np.diag(L)
    
    
class LMM2(object):
    
    def __init__(self,  formula, data, residual_formula=None, use_blocked=True):
        model_info = parse_random_effects(formula)
        re_terms = model_info["re_terms"]
        re_mod = RandomEffects(re_terms, data=data)
        y_vars = model_info["y_vars"]
        fe_form = model_info["fe_form"]
        X = formulaic.model_matrix(fe_form, data).values
        y = data[y_vars].values.reshape(-1, 1)
        R = None
        if use_blocked:
            self.mme = MMEBlocked(X, y, re_mod, R) 
        else:
            self.mme = MMEUnBlocked(X, y, re_mod, R) 


    def loglike(self, theta, reml=True):
        return self.mme._loglike(theta, reml)
        