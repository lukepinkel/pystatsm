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


        
class MMEU:
    def __init__(self, Z, X, y, G, re_mod, R=None):
        self.Z = Z
        self.X = X
        self.y = y
        self.G = G
        self.R = R 
        self.re_mod = re_mod
        
        self.Zt = Z.T.tocsc()
        self.n_ranef = Z.shape[1]
        self.n_fixef = X.shape[1]
        self.n_obs = y.shape[0]
        
        # Block X and y together
        self.Xy = np.hstack([X, y])
        
        if R is None:
            self.ZtR = None
            self.ZtZ = self.Zt.dot(Z)
            self.ZtRZ = self.ZtZ.copy()
            
            self.ZtXy = self.Zt.dot(self.Xy)
            self.ZtRXy = self.ZtXy.copy()
            
            self.XytXy = self.Xy.T.dot(self.Xy)
            self.XytRXy = self.XytXy.copy()
            
        else:
            self.ZtR = self.Zt.dot(R).tocsc()
            self.ZtZ = None
            self.ZtRZ = self.ZtR.dot(Z).tocsc()
            
            self.ZtXy = None
            self.ZtRXy = self.ZtR.dot(self.Xy)
            
            self.XytXy = None
            self.XytRXy = self.Xy.T.dot(R.dot(self.Xy))
        
        self.C = self.ZtRZ + self.G
        
        self.chol_fac = sksparse.cholmod.analyze(sp.sparse.csc_matrix(self.C), 
                                                 ordering_method="best")
        self._p = np.argsort(self.chol_fac.P())
        self.dg = np.zeros(self.n_fixef+self.n_ranef+1, dtype=np.double)
        
    def update_crossprods_scalar(self, theta):
        self.ZtRZ = self.ZtZ / theta[-1]
        self.ZtRXy = self.ZtXy / theta[-1]
        self.XytRXy = self.XytXy / theta[-1]
        
    def update_crossprods_nontrv(self, theta):
        self._update_rcov(theta)  # Assuming this method exists to update R
        csc_matmul(self.Zt, self.R, self.ZtR)
        csc_matmul(self.ZtR, self.Z, self.ZtRZ)
        self.ZtRXy = self.ZtR.dot(self.Xy)
        self.XytRXy = self.Xy.T.dot(self.R.dot(self.Xy))
        
    def update_crossprods(self, theta):
        if self.R is None:
            self.update_crossprods_scalar(theta)
        else:
            self.update_crossprods_nontrv(theta)
    
    def update_chol(self, theta):
        self.update_crossprods(theta)
        Ginv = self.re_mod.update_gcov(theta, inv=True, G=self.G)
        cs_add_inplace(self.ZtRZ, Ginv, self.C)
        self.chol_fac.cholesky_inplace(sp.sparse.csc_matrix(self.C))
        L11 = self.chol_fac.L()[self._p,:][:,self._p].toarray()
        L21 = self.chol_fac.apply_Pt(
                self.chol_fac.solve_L(
                    self.chol_fac.apply_P(self.ZtRXy), False)).T
        L22 = np.linalg.cholesky(self.XytRXy - L21.dot(L21.T))
        return L11, L21, L22
    
    def update_chol_diag(self, theta, out=None):
        out = self.dg if out is None else out
        self.update_crossprods(theta)
        Ginv = self.re_mod.update_gcov(theta, inv=True, G=self.G)
        cs_add_inplace(self.ZtRZ, Ginv, self.C)
        self.chol_fac.cholesky_inplace(sp.sparse.csc_matrix(self.C))
        L11 = self.chol_fac.L()[self._p,:][:,self._p].toarray()
        L21 = self.chol_fac.apply_Pt(
                self.chol_fac.solve_L(
                    self.chol_fac.apply_P(self.ZtRXy), False)).T
        L22 = np.linalg.cholesky(self.XytRXy - L21.dot(L21.T))
        out[:self.n_ranef] = np.diag(L11)
        out[self.n_ranef:] = np.diag(L22)
        return out
        
    
class LMM2(object):
    
    def __init__(self,  formula, data, residual_formula=None):
        model_info = parse_random_effects(formula)
        re_terms = model_info["re_terms"]
        re_mod = RandomEffects(re_terms, data=data)
        y_vars = model_info["y_vars"]
        fe_form = model_info["fe_form"]
        X = formulaic.model_matrix(fe_form, data)
        fe_vars = X.columns
        X = X.values
        y = data[y_vars].values
        if y.ndim==1:
            y = y.reshape(-1, 1)
        Z = re_mod.Z
        G = re_mod.G
        
        X_sp = sp.sparse.csc_array(X)
        y_sp = sp.sparse.csc_array(y)
        
        ZXyt = sp.sparse.hstack([Z, X_sp, y_sp], format='csr').T
        ZXyt_work = ZXyt.copy()
        M = ZXyt.dot(ZXyt.T)
        M_work = M.copy()
        self.model_info = model_info
        self.re_mod = re_mod
        self.fe_vars = fe_vars
        self.y_vars = y_vars
        self.G = G
        self.Z = Z
        self.X = X
        self.y = y
        self.ZXyt = ZXyt
        self.M = M
        self.ZXyt_work = ZXyt_work

        self.data = data
        self.n_rt = self.Z.shape[1]
        self.zero_mat = sp.sparse.eye(X.shape[1]+1)*0.0
        self.G_aug = sp.sparse.block_diag([self.G, self.zero_mat], format="csc")
        self.M_work = M_work + self.G_aug
        self.chol_fac = sksparse.cholmod.analyze(sp.sparse.csc_matrix(self.M_work))
        
        self.mme = MMEU(self.Z, self.X, self.y, self.G, self.re_mod)
        
    def update_gmat(self, theta, inv=False):
        G = self.re_mod.update_gcov(theta, inv=inv, G=self.G)
        return G
    
    def update_gmat_aug(self, theta, inv=False):
        G_aug = self.re_mod.update_gcov(theta, inv=inv, G=self.G_aug)
        return G_aug
        
    def lndet_gmat(self, theta):
        lnd = self.re_mod.lndet_gmat(theta)
        return lnd
    
    def update_mme(self, theta):
        G_aug = self.update_gmat_aug(theta, inv=True)
        #TODO sort this out for the nontrivial R situation
        M_work = self.M.copy()
        M_work.data /= theta[-1]
        
        M_work = cs_add_inplace(M_work, G_aug, M_work)
        return M_work
    
    def mme_chol(self, M, use_sparse=True, use_fac=True, sparse_threshold=0.4):
        #TODO: Actually take advantage of chol pattern
        if (sparsity(M) < sparse_threshold) and use_sparse:
            M = sp.sparse.csc_matrix(M)
            if use_fac:
                self.chol_fac.cholesky_inplace(M)
                L = self.chol_fac.L().toarray()
            else:
                L = cholesky(M).L().toarray()
        else:
            L = np.linalg.cholesky(M.toarray())
        return L

    def loglike(self, theta, reml=True, use_sw=False, use_sparse=True,
                use_mme=True):
        if use_mme:
            l = self.mme.update_chol_diag(theta)
        else:
            M = self.update_mme(theta)
            L = self.mme_chol(M, use_sparse=use_sparse)
            l = np.diag(L)
        ytPy = l[-1]**2
        logdetG = self.lndet_gmat(theta)
        #TODO Actually implement this with resid and memory sharing
        logdetR = np.log(theta[-1]) * self.Z.shape[0]
        if reml:
            logdetC = np.sum(2*np.log(l[:-1]))
            ll = logdetR + logdetC + logdetG + ytPy
        else:
            logdetV = np.sum(2*np.log(l[:self.n_rt]))
            ll = logdetR + logdetV + logdetG + ytPy
        return ll
        
        
        
    
class LMM2(object):
    
    def __init__(self,  formula, data, residual_formula=None):
        model_info = parse_random_effects(formula)
        re_terms = model_info["re_terms"]
        re_mod = RandomEffects(re_terms, data=data)
        y_vars = model_info["y_vars"]
        fe_form = model_info["fe_form"]
        X = formulaic.model_matrix(fe_form, data)
        fe_vars = X.columns
        X = X.values
        y = data[y_vars].values
        if y.ndim==1:
            y = y.reshape(-1, 1)
        Z = re_mod.Z
        G = re_mod.G
        
        X_sp = sp.sparse.csc_array(X)
        y_sp = sp.sparse.csc_array(y)
        
        ZXyt = sp.sparse.hstack([Z, X_sp, y_sp], format='csr').T
        ZXyt_work = ZXyt.copy()
        M = ZXyt.dot(ZXyt.T)
        M_work = M.copy()
        self.model_info = model_info
        self.re_mod = re_mod
        self.fe_vars = fe_vars
        self.y_vars = y_vars
        self.G = G
        self.Z = Z
        self.X = X
        self.y = y
        self.ZXyt = ZXyt
        self.M = M
        self.ZXyt_work = ZXyt_work

        self.data = data
        self.n_rt = self.Z.shape[1]
        self.zero_mat = sp.sparse.eye(X.shape[1]+1)*0.0
        self.G_aug = sp.sparse.block_diag([self.G, self.zero_mat], format="csc")
        self.M_work = M_work + self.G_aug
        self.chol_fac = sksparse.cholmod.analyze(sp.sparse.csc_matrix(self.M_work))
        
    def update_gmat(self, theta, inv=False):
        G = self.re_mod.update_gcov(theta, inv=inv, G=self.G)
        return G
    
    def update_gmat_aug(self, theta, inv=False):
        G_aug = self.re_mod.update_gcov(theta, inv=inv, G=self.G_aug)
        return G_aug
        
    def lndet_gmat(self, theta):
        lnd = self.re_mod.lndet_gmat(theta)
        return lnd
    
    def update_mme(self, theta):
        G_aug = self.update_gmat_aug(theta, inv=True)
        #TODO sort this out for the nontrivial R situation
        M_work = self.M.copy()
        M_work.data /= theta[-1]
        
        M_work = cs_add_inplace(M_work, G_aug, M_work)
        return M_work
    
    def mme_chol(self, M, use_sparse=True, use_fac=True):
        #TODO: Actually take advantage of chol pattern
        if sparsity(M) < 0.05 and use_sparse:
            M = sp.sparse.csc_matrix(M)
            if use_fac:
                self.chol_fac.cholesky_inplace(M)
                L = self.chol_fac.L().toarray()
            else:
                L = cholesky(M).L().toarray()
        else:
            L = np.linalg.cholesky(M.toarray())
        return L

    def loglike(self, theta, reml=True, use_sw=False, use_sparse=True):
        M = self.update_mme(theta)
        L = self.mme_chol(M, use_sparse=use_sparse)
        l = np.diag(L)
        ytPy = l[-1]**2
        logdetG = self.lndet_gmat(theta)
        #TODO Actually implement this with resid and memory sharing
        logdetR = np.log(theta[-1]) * self.Z.shape[0]
        if reml:
            logdetC = np.sum(2*np.log(l[:-1]))
            ll = logdetR + logdetC + logdetG + ytPy
        else:
            logdetV = np.sum(2*np.log(l[:self.n_rt]))
            ll = logdetR + logdetV + logdetG + ytPy
        return ll
        
    