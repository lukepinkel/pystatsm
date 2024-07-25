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
import sksparse
from sksparse.cholmod import cholesky
from abc import ABCMeta, abstractmethod

from ..utilities.python_wrappers import (sparse_dot,
                                         sparse_pattern_trace, 
                                         coo_to_csc, 
                                         sparse_dense_kron,
                                         sparse_dense_kron_inplace,
                                         ds_kron, 
                                         ds_kron_inplace,
                                         cs_matmul_inplace,
                                         cs_add_inplace,
                                         tile_1d)
from ..utilities.linalg_operations import (invech_chol, 
                                           _invech, invech,
                                           _vech, vech,
                                           _vec)
from ..utilities.formula import parse_random_effects
from ..utilities.indexing_utils import vech_inds_reverse
from ..utilities.param_transforms import CholeskyCov, CombinedTransform

def sizes_to_inds(sizes):
    return np.r_[0, np.cumsum(sizes)]
    
def sizes_to_slices(sizes):
    inds = sizes_to_inds(sizes)
    slices = [slice(x, y) for x,y in list(zip(inds[:-1], inds[1:]))]
    return slices

def make_ginds_bdiag(n_rv, n_lv):
    gdim = n_rv * n_lv
    g_cols = np.repeat(np.arange(gdim), n_rv)
    g_rows = np.repeat(np.arange(n_lv)*n_rv, n_rv * n_rv) + np.tile(np.arange(n_rv), gdim)
    return g_rows, g_cols

def logpdet(arr):
    u = np.linalg.eigvalsh(arr)
    pld = np.sum(np.log(u[u>0]))
    return pld


def make_zinds(n_ob, n_rv, cols):
    z_rows = np.repeat(np.arange(n_ob), n_rv)
    z_cols = np.repeat(cols * n_rv, n_rv)  + np.tile(np.arange(n_rv), n_ob)
    return z_rows, z_cols

def make_remod_mat(arr1, arr2, return_array=False):
    cols, u = pd.factorize(arr1, sort=False)
    n_lv = len(u)
    n_ob, n_rv = arr2.shape
    z_rows, z_cols = make_zinds(n_ob, n_rv, cols)
    z_data = arr2.reshape(-1, order='C')
    z_size = (n_ob, n_lv * n_rv)
    Z = coo_to_csc(z_rows, z_cols, z_data, z_size, return_array=True)
    return Z, n_rv, n_lv


def sparsity(a):
    return a.nnz / np.prod(a.shape)


class BaseCovarianceStructure(metaclass=ABCMeta):

    @abstractmethod
    def params_to_cov(self, params):
        pass

    @abstractmethod
    def dcov_dparams(self, params, i):
        pass

    @abstractmethod
    def d2cov_dparams(self, params, i, j):
        pass

    @abstractmethod
    def get_logdet(self, params):
        pass

    
class UnstructuredCovariance(BaseCovarianceStructure):
    
    def __init__(self, n_vars):
       
        n_pars = int((n_vars + 1) * n_vars // 2)
        theta_init = _vech(np.eye(n_vars))
        d_theta = np.zeros_like(theta_init)
        r_inds, c_inds = vech_inds_reverse(np.arange(n_pars),  n_vars)
        d_mask = r_inds == c_inds
        dCov = np.zeros((n_vars, n_vars))
        self.n_vars = n_vars
        self.n_pars = n_pars
        self.theta_init = theta_init
        self.d_theta = d_theta
        self.r_inds = r_inds
        self.c_inds = c_inds
        self.d_mask = d_mask
        self.dCov = dCov
        self.theta_init = theta_init

    def params_to_cov(self, params):
        cov = _invech(params)
        return cov

    def dcov_dparams(self, params, i, out=None):
        out = self.dCov if out is None else out
        out *= 0.0
        r, c = self.r_inds[i], self.c_inds[i]
        out[r, c] = out[c, r] = 1.0
        return out

    def d2cov_dparams(self, params, i, j, out=None):
        out = self.dCov if out is None else out
        out *= 0.0
        return out

    def get_logdet(self, params):
        cov = _invech(params)
        _, lnd = np.linalg.slogdet(cov)
        return lnd

class BaseProductCovariance(object):
    
    def __init__(self, n_rv, n_lv):
        self.n_rv = n_rv
        self.n_lv = n_lv
        self.nr = self.nc = n_rv * n_lv
        self.unstructured_cov = UnstructuredCovariance(n_rv)
        self.G = self._make_initial_matrix()
        self.g_data = self.G.data
        self.g_indices = self.G.indices
        self.g_indptr = self.G.indptr
        self.G_derivs = []
        
    @abstractmethod
    def _make_initial_matrix(self):
        pass

    @abstractmethod
    def _update_gdata(self, G0, inv, out):
        pass
    
    @abstractmethod
    def dcov_dparams(self):
        pass
    
    def update_gdata(self, params, inv=False, out=None):
        G0 = invech(params)
        if inv:
            G0 = np.linalg.inv(G0)
        out = self.g_data if out is None else out
        self._update_gdata(G0, inv, out)
        return out
    
    def params_to_cov(self, params):
       self.update_gdata(params)
       return self.G
   
    def precompute_derivatives(self):
        for i in range(self.unstructured_cov.n_pars):
            self.G_derivs.append(self.dcov_dparams(None, i))
   

    
    
class KronIG(BaseProductCovariance):
    
    def __init__(self, n_rv, n_lv):
        self.a_cov = None
        self.a_inv = None
        self.lda_const = 0.0
        super().__init__(n_rv, n_lv)
        self.precompute_derivatives()
    
    def _make_initial_matrix(self):
        n_lv, n_rv, nr = self.n_lv, self.n_rv, self.nr
        G0 = np.eye(n_rv)        
        g_data = tile_1d(G0.reshape(-1, order='F'), n_lv)
        
        g_cols = np.repeat(np.arange(nr), n_rv)
        g_rows = np.repeat(np.arange(n_lv)*n_rv, n_rv * n_rv) + np.tile(np.arange(n_rv), nr)

        G = coo_to_csc(g_rows, g_cols, g_data, (nr, nr), return_array=True)
        self.G = G
        self.g_rows, self.g_cols = g_rows, g_cols
        return G
    
    def _update_gdata(self, G0, inv, out):
        g = G0.reshape(-1, order='F')
        tile_1d(g, self.n_lv, out=out)
      
    def update_gcov(self, params, inv=False, G=None):
        G = self.G if G is None else G
        out = G.data
        self.update_gdata(params, inv, out)
        return G
    
    def get_logdet(self, params, out=0.0):
        G0 = invech(params)
        out += self.n_lv * np.linalg.slogdet(G0)[1]
        return out
    
    def dcov_dparams(self, params, i):
        dgi = _vec(self.unstructured_cov.dcov_dparams(params, i).copy())
        dgi = tile_1d(dgi, self.n_lv)
        csc_data = (dgi, (self.g_rows, self.g_cols))
        csc_shape = (self.nr, self.nc)
        dGi = sp.sparse.csc_array(csc_data, shape=csc_shape, dtype=np.double)
        dGi.eliminate_zeros()
        dGi.indptr = dGi.indptr.astype(np.int32)
        dGi.indices = dGi.indices.astype(np.int32)

        return dGi
 
    
#TODO: class KronGI(BaseProductCovariance):
  

    
class KronAG(BaseProductCovariance):
    
    def __init__(self, n_rv, n_lv, a_cov, a_inv):
        self.a_cov = a_cov
        self.a_inv = a_inv
        self.lda_const = logpdet(a_cov.toarray()) * n_rv
        super().__init__(n_rv, n_lv)
    
    def _make_initial_matrix(self):
        n_rv = self.n_rv
        G0 = np.eye(n_rv)        
        G = sparse_dense_kron(self.a_cov, G0)
        return G
    
    def _update_gdata(self, G0, inv, out):
        A = self.a_inv if inv else self.a_cov
        sparse_dense_kron_inplace(A, G0, out)
      
    def update_gcov(self, params, inv=False, G=None):
        G = self.G if G is None else G
        self.update_gdata(params, inv, G)
        return G
    
    def get_logdet(self, params, out=0.0):
        G0 = invech(params)
        out += self.n_levels * np.linalg.slogdet(G0)[1] + self.lda_const
        return out
    
    def dcov_dparams(self, params, i):
        dG0_i = self.unstructured_cov.dcov_dparams(params, i).copy()
        dGi = sparse_dense_kron(self.a_cov, dG0_i)
        dGi.eliminate_zeros()
        return dGi
    
class KronGA(BaseProductCovariance):
    
    def __init__(self, n_rv, n_lv, a_cov, a_inv):
        self.a_cov = a_cov
        self.a_inv = a_inv
        self.lda_const = logpdet(a_cov.toarray()) * n_rv
        super().__init__(n_rv, n_lv)
    
    def _make_initial_matrix(self):
        n_rv = self.n_rv
        G0 = np.eye(n_rv)        
        G = ds_kron(G0, self.a_cov)
        return G
    
    def _update_gdata(self, G0, inv, out):
        A = self.a_inv if inv else self.a_cov
        ds_kron_inplace(G0, A, out)
      
    def update_gcov(self, params, inv=False, G=None):
        G = self.G if G is None else G
        self.update_gdata(params, inv, G)
        return G
    
    def get_logdet(self, params, out=0.0):
        G0 = invech(params)
        out += self.n_levels * np.linalg.slogdet(G0)[1] + self.lda_const
        return out    
    
    def dcov_dparams(self, params, i):
        dG0_i = self.unstructured_cov.dcov_dparams(params, i).copy()
        dGi = ds_kron(dG0_i, self.a_cov)
        dGi.eliminate_zeros()
        return dGi
  
class RandomEffectTerm:
    def __init__(self, re_arr, gr_arr, a_cov=None, reparam=CholeskyCov):
        self.level_indices, self.levels = pd.factorize(gr_arr, sort=False)
        self.n_levels = len(self.levels)
        self.n_obs, self.n_revars = re_arr.shape
        self.n_ranefs = self.n_revars * self.n_levels 
        
        if a_cov is None:
            self.cov_structure = KronIG(self.n_revars, self.n_levels)
        elif sp.sparse.issparse(a_cov):
            a_inv = np.linalg.pinv(a_cov.toarray())#TODO: Fix this mess
            self.cov_structure = KronAG(self.n_revars, self.n_levels, a_cov, a_inv)
        
            
        
        self.G = self.cov_structure.G
        self.g_data = self.G.data
        self.g_indices = self.G.indices
        self.g_indptr = self.G.indptr
        self.n_gcovnz = self.G.nnz

        self.Z = self._make_z_matrix(re_arr)
        self.Z.eliminate_zeros()
        self.Z.sort_indices()
        self.theta_init = self.cov_structure.unstructured_cov.theta_init
        self.n_pars = self.cov_structure.unstructured_cov.n_pars
        
        self.reparam = reparam(self.n_revars)  

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

    def update_gdata(self, theta, inv=False, out=None):
        self.cov_structure.update_gdata(theta, inv, out)
        return out
    
    
    def update_gcov(self, theta, inv=False, G=None):
        G = self.G if G is None else G
        out = G.data
        self.cov_structure.update_gdata(theta, inv, out)
        return G
    
    def get_logdet(self, theta, out=0.0):
        out = self.cov_structure.get_logdet(theta, out)
        return out


class DiagResidualCovTerm:
    def __init__(self, re_arr, gr_arr, a_cov, reparam=CholeskyCov):
        n = gr_arr.shape[0]
        
        self.n_ranefs = self.n_obs = self.n_levels = n
        self.n_revars  = 1
        
        self.cov_structure = KronIG(1, n)
        self.G = self.cov_structure.G
        self.G_data = self.G.data
        self.G_indices = self.G.indices
        self.G_indptr = self.G.indptr
        self.n_gcovnz = self.G.nnz
        
        self.theta_init = self.cov_structure.unstructured_cov.theta_init
        self.n_pars = self.cov_structure.unstructured_cov.n_pars
        
        self.reparam = reparam(self.n_revars)


    @classmethod
    def from_formula(cls, re_grouping, data, a_cov=None):
        gr_arr = data[re_grouping].values
        return cls(None, gr_arr, a_cov)
        
    def update_gdata(self, theta, inv=False, out=None):
        self.cov_structure.update_gdata(theta, inv, out)
        return out
    
    def update_gcov(self, theta, inv=False, G=None):
        G = self.G if G is None else G
        out = G.data
        self.cov_structure.update_gdata(theta, inv, out)
        return G
    
    
    def get_logdet(self, theta, out=0.0):
        out = self.cov_structure.get_logdet(theta, out)
        return out
    
    
class RandomEffects:
    def __init__(self, re_terms, data, a_covs=None, resid_cov=None):
        self._create_terms(re_terms, data, a_covs, resid_cov)
        self._construct_matrices()

    def _create_terms(self, re_terms, data, a_covs, resid_cov):
        
        #G Specific
        a_covs = [None] * len(re_terms) if a_covs is None else a_covs
        tterms, theta = [], []
        n_gterms = len(re_terms)
        n_tterms = n_gterms + 1
        
        #G specific
        for i in range(n_gterms):
            (fr, gr), a = re_terms[i], a_covs[i]
            gterm = RandomEffectTerm.from_formula(fr, gr, data, a)
            tterms.append(gterm)
            theta.append(gterm.theta_init)
        
        rterm = DiagResidualCovTerm(None, np.arange(data.shape[0]), None)
       
        tterms.append(rterm)
        theta.append(rterm.theta_init)

        gterms = tterms[:-1]
        ranef_sl = sizes_to_slices([term.n_ranefs for term in gterms])
        gdata_sl = sizes_to_slices([term.n_gcovnz for term in gterms])
        
        n_pars = np.array([term.n_pars for term in tterms], dtype=np.int32)
        n_gpars = n_pars[:-1]
        
        theta_sl = sizes_to_slices(n_pars)
        
        n_gpar = np.sum(n_gpars)
        n_par = np.sum(n_pars)
        
        theta_to_term = np.repeat(np.arange(n_tterms), n_pars)
        
        theta = np.concatenate(theta)
        
        self.tterms, self.gterms = tterms, gterms
        self.n_tterms, self.n_gterms = n_tterms, n_gterms
        self.theta, self.theta_sl, self.theta_to_term = theta, theta_sl, theta_to_term
        
        self.ranef_sl, self.gdata_sl = ranef_sl, gdata_sl
        self.n_gpars, self.n_gpar = n_gpars, n_gpar
        self.n_par, self.n_pars = n_par, n_pars
        self.reparams = [term.reparam for term in self.tterms]        
        self.reparam = CombinedTransform(self.reparams, self.theta_sl)
        

    def _construct_matrices(self):
        self.Z = sp.sparse.hstack([term.Z for term in self.gterms], format='csc')
        self.G = sp.sparse.block_diag([term.G for term in self.gterms], format='csc')
        self.R = self.tterms[-1].G
        self.g_data = self.G.data
        self.r_data = self.R.data
        self.G_derivs = [deriv for term in self.tterms for deriv in term.cov_structure.G_derivs]
        
     
    def update_gdata(self, theta, inv=False, out=None):
        out = self.g_data if out is None else out
        for i, term in enumerate(self.gterms):
            theta_i = theta[self.theta_sl[i]]
            out_i = out[self.gdata_sl[i]]
            term.update_gdata(theta_i, inv, out_i) 
        return out
    
    
    def update_rdata(self, theta, inv=False, out=None):
        out = self.r_data if out is None else out
        r_term = self.tterms[-1]
        theta_r = theta[self.theta_sl[-1]]
        r_term.update_gdata(theta_r, inv, out)
        return out
        
    
    def update_gcov(self, theta, inv=False, G=None):
        G = self.G if G is None else G
        out = G.data
        self.update_gdata(theta, inv, out)
        return G
    
    def update_gcov_reparam(self, eta, inv=False, G=None):
        theta = self.reparam.rvs(eta)
        G = self.update_gcov(theta, inv, G)
        return G
    
    def update_rcov(self, theta, inv=False, R=None):
        R = self.R if R is None else R
        out = R.data
        self.update_rdata(theta, inv, out)
        return R
    
    def update_rcov_reparam(self, eta, inv=False, R=None):
        theta = self.reparam.rvs(eta)
        R = self.update_rcov(theta, inv, R)
        return R
    
    def lndet_gmat(self, theta, out=0.0):
        for i, term in enumerate(self.gterms):
            theta_i = theta[self.theta_sl[i]]
            out = term.get_logdet(theta_i, out)
        return out
    
    def lndet_rmat(self, theta, out=0.0):
        r_term = self.tterms[-1]
        theta_r = theta[self.theta_sl[-1]]
        out += r_term.get_logdet(theta_r)
        return out


    
class BaseMME:
    def __init__(self, X, y, re_mod):
        self.Z = re_mod.Z
        self.X = X
        self.y = y
        self.G = re_mod.G
        self.R = re_mod.R
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
        logdetR = self.re_mod.lndet_rmat(theta)#logdetR = np.log(theta[-1]) * self.n_obs
        if reml:
            logdetC = np.sum(2 * np.log(l[:-1]))
            ll = logdetR + logdetC + logdetG + ytPy
        else:
            logdetV = np.sum(2 * np.log(l[:self.n_ranef]))
            ll = logdetR + logdetV + logdetG + ytPy
        return ll
    
    def _loglike_reparam(self, eta, reml=True):
        theta = self.re_mod.reparam.rvs(eta)
        return self._loglike(theta, reml)
    
    
    
    
class MMEBlocked(BaseMME):
    def __init__(self, X, y, re_mod, sparse_threshold=0.4):
        self.Zt = re_mod.Z.T.tocsc()
        self.Xy = np.hstack([X, y])
        self.sparse_threshold = sparse_threshold
        self._scalar_resid = isinstance(re_mod.tterms[-1], DiagResidualCovTerm)
        super().__init__(X, y, re_mod)
    
    def _initialize_matrices(self):
        if self._scalar_resid:
            self._initialize_unweighted()
        else:
            self._initialize_weighted()
        self.C = self.ZtRZ + self.G
    
    def _initialize_unweighted(self):
        ZtR = self.Zt.dot(self.R).tocsc()
        ZtRdR = ZtR.copy() #so worst caase
        ZtR_dR_RZ = sp.sparse.csc_array.dot(ZtRdR, ZtR.T).tocsc()
        
        ZtRZ = sp.sparse.csc_array.dot(ZtR, self.Zt.T).tocsc()
        ZtZ =  sp.sparse.csc_array.dot(self.Zt, self.Zt.T).tocsc()
       
        ZtRXy = ZtR.dot(self.Xy)
        ZtXy = sp.sparse.csc_array.dot(self.Zt, self.Xy)
        
        XytRXy = (self.R.dot(self.Xy)).T.dot(self.Xy)
        XytXy = self.Xy.T.dot(self.Xy)
        
        self.ZtR, self.ZtRdR, self.ZtR_dR_RZ = ZtR, ZtRdR, ZtR_dR_RZ
        self.ZtRZ, self.ZtZ = ZtRZ, ZtZ
        self.ZtRXy, self.ZtXy = ZtRXy, ZtXy
        self.XytRXy, self.XytXy = XytRXy, XytXy
        
    def _initialize_weighted(self):
         ZtR = self.Zt.dot(self.R).tocsc()
         ZtRdR = ZtR.copy() #so worst caase
         ZtR_dR_RZ = sp.sparse.csc_array.dot(ZtRdR, ZtR.T).tocsc()
         
         ZtRZ = sp.sparse.csc_array.dot(ZtR, self.Zt.T).tocsc()
         ZtZ = None
        
         ZtRXy = ZtR.dot(self.Xy)
         ZtXy = None
         
         XytRXy = (self.R.dot(self.Xy)).T.dot(self.Xy)
         XytXy = None
         
         self.ZtR, self.ZtRdR, self.ZtR_dR_RZ = ZtR, ZtRdR, ZtR_dR_RZ
         self.ZtRZ, self.ZtZ = ZtRZ, ZtZ
         self.ZtRXy, self.XtXy = ZtRXy, ZtXy
         self.XytRXy, self.XytXy = XytRXy, XytXy
    
    def _setup_cholesky(self):
        self.chol_fac = sksparse.cholmod.analyze(sp.sparse.csc_matrix(self.C), ordering_method="best")
        self._p = np.argsort(self.chol_fac.P())
        self.dg = np.zeros(self.n_fixef + self.n_ranef + 1, dtype=np.double)
        
    def update_crossprods(self, theta):
        if self._scalar_resid:
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
        cs_matmul_inplace(self.Zt, self.R, self.ZtR)
        cs_matmul_inplace(self.ZtR, self.Z, self.ZtRZ)
        self.ZtRXy = self.ZtR.dot(self.Xy)
        self.XytRXy = self.Xy.T.dot(self.R.dot(self.Xy))
    
    def _update_rcov(self, theta, inv=False):
        self.re_mod.update_rcov(theta, inv, self.R)
        
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
        L11 = self.chol_fac.L()
        L11 = L11[self._p,:][:,self._p]
        L11 = L11.toarray()
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
    
    
    def _chol_sparse_diag(self, C):
        self.chol_fac.cholesky_inplace(sp.sparse.csc_matrix(C))
        L11 = self.chol_fac.L()
        #L11 = L11[self._p,:][:,self._p]
        #L11_diag = L11.diagonal()
        L11_diag = L11[self._p, self._p]
        L21 = self.chol_fac.apply_Pt(
                self.chol_fac.solve_L(
                    self.chol_fac.apply_P(self.ZtRXy), False)).T
        L22_diag = np.diag(np.linalg.cholesky(self.XytRXy - L21.dot(L21.T)))
        return L11_diag, L21, L22_diag
    
    def _chol_dense_diag(self, C):
        C_dense = C.toarray()
        L11 = np.linalg.cholesky(C_dense)
        L21 = sp.linalg.solve_triangular(L11, self.ZtRXy, trans=0, lower=True).T
        L22_diag = np.diag(np.linalg.cholesky(self.XytRXy - L21.dot(L21.T)))
        L11_diag = np.diag(L11)
        return L11_diag, L21, L22_diag
    
    def update_chol_diag(self, theta, out=None):
        out = self.dg if out is None else out
        
        self.update_crossprods(theta)
        Ginv = self.re_mod.update_gcov(theta, inv=True, G=self.G)
        cs_add_inplace(self.ZtRZ, Ginv, self.C)
        
        if sparsity(self.C) < self.sparse_threshold:
            L11_diag, L21, L22_diag = self._chol_sparse_diag(self.C)
        else:
            L11_diag, L21, L22_diag = self._chol_dense_diag(self.C)
        out[:self.n_ranef] = L11_diag
        out[self.n_ranef:] = L22_diag
        return out
    
    def _gradient(self, theta, reml=True):
        self.update_crossprods(theta)
        Ginv = self.re_mod.update_gcov(theta, inv=True, G=self.G)
        C = cs_add_inplace(self.ZtRZ, Ginv, self.C)
        self.chol_fac.cholesky_inplace(sp.sparse.csc_matrix(C))

        ZtRZ = self.ZtRZ
        ZtRXy = self.ZtRXy
        ZtRX = ZtRXy[:, :-1]
        ZtRy = ZtRXy[:, [-1]]
        XtRX = self.XytRXy[:-1, :-1]
        XtRy = self.XytRXy[:-1, [-1]]

        L_zz = self.chol_fac.apply_Pt(self.chol_fac.solve_L(
            self.chol_fac.apply_P(sp.sparse.csc_matrix(ZtRZ)), False)).T
        
        L_zxyt =  self .chol_fac.apply_Pt(self.chol_fac.solve_L(
            self.chol_fac.apply_P(sp.sparse.csc_matrix(ZtRXy)), False))
        #L22 = np.linalg.cholesky(self.XytRXy - L_zxyt.T.dot(L_zxyt))
        
        L_zxt = L_zxyt[:, :-1]
        L_zyt = L_zxyt[:, -1]
        L_zx = L_zxt.T.tocsc()
        L_zy = L_zyt.T.tocsc()


        #T_zz = ZtRZ - L_zz.dot(L_zz.T)
        T_zx = ZtRX - L_zz.dot(L_zx.T)
        T_xx = XtRX - L_zx.dot(L_zx.T)
        T_xx_inv = np.linalg.inv(T_xx)
        u_xy = XtRy - L_zx.dot(L_zy.T)
        u_zy = ZtRy - L_zz.dot(L_zy.T)
        v_y = u_zy - T_zx.dot(T_xx_inv.dot(u_xy))
        T2B = _vec(np.asarray(T_xx_inv))
        grad = np.zeros_like(theta)

        # Gradient for G parameters
        for i in range(self.re_mod.n_gpar):
            dGi = self.re_mod.G_derivs[i]
            k = self.re_mod.theta_to_term[i]
            sl = self.re_mod.ranef_sl[k]
           
            T_zxi = T_zx[sl]
            
            L_zzi = L_zz[sl]
            L_zzit= L_zzi.T.tocsc()
            ZtRZi = ZtRZ[sl,sl]
            
            #trprd1 = sp.sparse.csc_array.dot(T_zz[sl, sl], dGi).diagonal()
            
            T1 = ZtRZi.dot(dGi).diagonal().sum() - sparse_pattern_trace(L_zzit, dGi) 
            #L_zzi = L_zz[sl].T.tocsc()
            #ZtRZi = ZtRZ[sl,sl]
            #trprd2 = np.zeros(dGi.nnz)
            #dGi_coo = dGi.tocoo()
            #for c in range(dGi_coo.nnz):
            #    j, k = dGi_coo.row[c], dGi_coo.col[c]
            #    trprd2[c] = ZtRZi[j, k] - sparse_dot(L_zzi[:, j], L_zzi[:, k])
            
            #assert(np.allclose(trprd1, trprd2))
            
            #T1 = sp.sparse.csc_array.dot(T_zz[sl, sl], dGi).diagonal().sum()
            
            T2A = T_zxi.T.dot(dGi.dot(T_zxi))
            T2A =  _vec(np.asarray(T2A))
            T2 = -np.dot(T2A, T2B)
            if reml:
                T3 = -np.asarray(np.dot(dGi.dot(v_y[sl]).T, v_y[sl])).flatten()[0]
            else:
                T3 = 0.0
            grad[i] = T1 + T2 + T3
        #scuffed 
        
        self._update_rcov(theta, inv=True)
        Rinv, X, Z, y = self.R, self.X, self.Z, self.y
        b = T_xx_inv.dot(u_xy)
        G = self.re_mod.update_gcov(theta, inv=False, G=self.G)
        u = np.asarray(G.dot(v_y))
        r = y - X.dot(b) - Z.dot(u)
        Py = Rinv.dot(r)
        MZtRX = self.chol_fac.solve_A(sp.sparse.csc_matrix(ZtRX))
        T1A = Rinv.diagonal().sum()
        cs_matmul_inplace(self.Zt, Rinv, self.ZtR)
        
        for i in range(self.re_mod.n_gpar, self.re_mod.n_par):
            dRi = self.re_mod.G_derivs[i]
            cs_matmul_inplace(self.ZtR, dRi, self.ZtRdR)
            cs_matmul_inplace(self.ZtRdR, self.ZtR.T.tocsc(), self.ZtR_dR_RZ)
            ZtR_dR_RZ = self.ZtR_dR_RZ
            XtR = Rinv.dot(X).T
            T1B = (self.chol_fac.solve_A(
                    sp.sparse.csc_matrix(ZtR_dR_RZ)).diagonal()).sum()

            
            T1 = T1A-T1B
            T2 = np.dot((dRi.dot(Py)).T, Py)[0,0]
            if reml:
                T3A = (dRi.dot(XtR.T)).T.dot(XtR.T)
                T3B = -2.0 * sp.sparse.csc_array.dot(((self.ZtRdR).dot(XtR.T)).T, MZtRX)
                T3C = (ZtR_dR_RZ.dot(MZtRX)).T.dot(MZtRX)
                T3 = np.trace(T_xx_inv.dot((T3A + T3B + T3C)))
            else:
                T3 = 0.0
            grad[i] = T1 - T2 - T3

        return grad
    
    def _gradient_reparam(self, eta, reml=True):
        theta = self.re_mod.reparam.rvs(eta)
        return self._gradient(theta, reml)
    
    
        
    
class LMM2(object):
    
    def __init__(self,  formula, data, residual_formula=None, mme_kws=None):
        model_info = parse_random_effects(formula)
        re_terms = model_info["re_terms"]
        re_mod = RandomEffects(re_terms, data=data)
        y_vars = model_info["y_vars"]
        fe_form = model_info["fe_form"]
        X = formulaic.model_matrix(fe_form, data).values
        y = data[y_vars].values.reshape(-1, 1)
        mme_kws = {} if mme_kws is None else mme_kws
        self.mme = MMEBlocked(X, y, re_mod, **mme_kws) 
 

    def loglike(self, theta, reml=True):
        return self.mme._loglike(theta, reml)
    
    def gradient(self, theta, reml=True):
        return self.mme._gradient(theta, reml)
        
    def loglike_reparam(self, eta, reml=True):
        return self.mme._loglike_reparam(eta, reml)
    
    def gradient_reparam(self, eta, reml=True):
        dl_dtheta = self.mme._gradient_reparam(eta, reml)
        dtheta_deta = self.mme.re_mod.reparam.jac_rvs(eta)
        g = np.dot(dtheta_deta, dl_dtheta)
        return g
         
