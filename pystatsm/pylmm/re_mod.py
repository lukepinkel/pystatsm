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
import matplotlib as mpl
import matplotlib.pyplot as plt
from abc import ABCMeta, abstractmethod
from scipy.interpolate import interp1d

from ..utilities.python_wrappers import (sparse_dot,
                                         sparse_pattern_trace,
                                         block_diag_self_dot,
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
                                           _vec,
                                           cholesky,
                                           analyze)
from ..utilities.formula import parse_random_effects
from ..utilities.indexing_utils import vech_inds_reverse
from ..utilities.param_transforms import CholeskyCov, CombinedTransform
from ..utilities.numerical_derivs import so_gc_cd

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
    Z = coo_to_csc(z_rows, z_cols, z_data, z_size, return_array=return_array)
    return Z, n_rv, n_lv


def sparsity(a):
    return a.nnz / np.prod(a.shape)


def _resolve_a(a_cov):
    n = a_cov.shape[0]
    if sp.sparse.issparse(a_cov):
        a_csc = a_cov.tocsc()
        fac = cholesky(sp.sparse.csc_matrix(a_csc))
        a_inv = sp.sparse.csc_matrix(fac.solve_A(sp.sparse.eye(n, format='csc')))
        return a_csc, a_inv
    A = np.asarray(a_cov)
    L = np.linalg.cholesky(A)
    A_inv = sp.linalg.cho_solve((L, True), np.eye(n))
    return sp.sparse.csc_matrix(A), sp.sparse.csc_matrix(A_inv)


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

class BaseProductCovariance(metaclass=ABCMeta):

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

    def update_gcov(self, params, inv=False, G=None):
        G = self.G if G is None else G
        self.update_gdata(params, inv, G.data)
        return G

    def params_to_cov(self, params):
       self.update_gdata(params)
       return self.G

    def precompute_derivatives(self):
        for i in range(self.unstructured_cov.n_pars):
            self.G_derivs.append(self.dcov_dparams(None, i))

    def precompute_grad_caches(self, ZtRZ, sl):
        pass

    def accumulate_gradient(self, par_offset, sl, ZtRZ, L_zz, T_zx, v_y,
                            T_xx_inv, reml, grad):
        L_zz_blk_t = L_zz[sl].T.tocsc()
        ZtRZ_blk = ZtRZ[sl, sl]
        T_zx_blk = T_zx[sl]
        v_y_blk = v_y[sl]
        T2B = _vec(np.asarray(T_xx_inv))
        for j in range(self.unstructured_cov.n_pars):
            dGj = self.G_derivs[j]
            T1 = ZtRZ_blk.dot(dGj).diagonal().sum() - sparse_pattern_trace(L_zz_blk_t, dGj)
            T3 = -float(np.asarray(dGj.dot(v_y_blk).T.dot(v_y_blk)).flatten()[0])
            if reml:
                T2A = _vec(np.asarray(T_zx_blk.T.dot(dGj.dot(T_zx_blk))))
                T2 = -np.dot(T2A, T2B)
            else:
                T2 = 0.0
            grad[par_offset + j] = T1 + T2 + T3




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

    def precompute_grad_caches(self, ZtRZ, sl):
        ng, nv = self.n_lv, self.n_rv
        offset = sl.start
        indptr, indices = ZtRZ.indptr, ZtRZ.indices
        cols = offset + np.arange(ng * nv, dtype=np.int64)
        col_starts = indptr[cols].astype(np.int64)
        col_ends = indptr[cols + 1].astype(np.int64)
        in_term = np.empty(ng * nv, dtype=np.int64)
        for k in range(ng * nv):
            s, e = col_starts[k], col_ends[k]
            in_term[k] = s + np.searchsorted(indices[s:e], offset, side='left')
        a = np.arange(nv, dtype=np.int64)[None, :, None]
        self._zr_index_map = in_term.reshape(ng, nv)[:, None, :] + a

    def accumulate_gradient(self, par_offset, sl, ZtRZ, L_zz, T_zx, v_y,
                            T_xx_inv, reml, grad, L_zz_sl=None):
        ng, nv = self.n_lv, self.n_rv
        ZtRZ_sum = ZtRZ.data[self._zr_index_map].sum(axis=0)
        L_blk = (L_zz_sl if L_zz_sl is not None else L_zz[sl]).tocsr()
        LL_sum = block_diag_self_dot(L_blk, ng, nv)
        T3 = T_zx[sl].reshape(ng, nv, -1)
        v_blk = v_y[sl].reshape(ng, nv)
        bb_sum = v_blk.T.dot(v_blk)
        M_sum = ZtRZ_sum - LL_sum - bb_sum
        if reml:
            M_sum = M_sum - np.einsum('jap,pq,jbq->ab', T3, T_xx_inv, T3,
                                      optimize=True)
        ucov = self.unstructured_cov
        ri, ci = ucov.r_inds, ucov.c_inds
        mult = np.where(ucov.d_mask, 1.0, 2.0)
        grad[par_offset:par_offset + ucov.n_pars] = M_sum[ri, ci] * mult



class KronAG(BaseProductCovariance):

    def __init__(self, n_rv, n_lv, a_cov, a_inv):
        self.a_cov = a_cov
        self.a_inv = a_inv
        self.lda_const = logpdet(a_cov.toarray()) * n_rv
        super().__init__(n_rv, n_lv)
        self.precompute_derivatives()

    def _make_initial_matrix(self):
        n_rv = self.n_rv
        G0 = np.asfortranarray(np.eye(n_rv))
        G = sparse_dense_kron(self.a_cov, G0)
        return G

    def _update_gdata(self, G0, inv, out):
        A = self.a_inv if inv else self.a_cov
        sparse_dense_kron_inplace(A, G0, out)

    def get_logdet(self, params, out=0.0):
        G0 = invech(params)
        out += self.n_lv * np.linalg.slogdet(G0)[1] + self.lda_const
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
        self.precompute_derivatives()

    def _make_initial_matrix(self):
        n_rv = self.n_rv
        G0 = np.eye(n_rv)
        G = ds_kron(G0, self.a_cov)
        return G

    def _update_gdata(self, G0, inv, out):
        A = self.a_inv if inv else self.a_cov
        ds_kron_inplace(G0, A, out)

    def get_logdet(self, params, out=0.0):
        G0 = invech(params)
        out += self.n_lv * np.linalg.slogdet(G0)[1] + self.lda_const
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
        else:
            a_cov, a_inv = _resolve_a(a_cov)
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
        dm = patsy.dmatrix(re_formula, data=data, return_type='dataframe')
        gr_arr = data[re_grouping].values
        obj = cls(dm.values, gr_arr, a_cov)
        obj.re_formula = re_formula
        obj.re_grouping = re_grouping
        obj.var_names = list(dm.columns)
        return obj

    def _make_z_matrix(self, re_arr):
        z_rows, z_cols = make_zinds(self.n_obs, self.n_revars, self.level_indices)
        z_data = re_arr.reshape(-1, order='C')
        z_size = (self.n_obs, self.n_ranefs)
        return coo_to_csc(z_rows, z_cols, z_data, z_size, return_array=True)

    def update_gdata(self, theta, inv=False, out=None):
        if out is None:
            out = self.G.data
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
        if out is None:
            out = self.G.data
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

    def accumulate_gradient(self, mme, theta, reml, grad, T_xx_inv, u_xy, v_y,
                            **kwargs):
        sigma2 = float(theta[-1])
        b = T_xx_inv.dot(u_xy)
        G = mme.re_mod.update_gcov(theta, inv=False, G=mme.G)
        u = np.asarray(G.dot(v_y))
        r = mme.y - mme.X.dot(b) - mme.Z.dot(u)
        y_dot_r = float(np.asarray(mme.y).reshape(-1).dot(
            np.asarray(r).reshape(-1)))
        n_eff = mme.n_obs - mme.n_fixef if reml else mme.n_obs
        n_gpar = mme.re_mod.n_gpar
        theta_dot_grad = float(np.dot(theta[:n_gpar], grad[:n_gpar]))
        grad[n_gpar] = (n_eff / sigma2 - y_dot_r / sigma2**2
                        - theta_dot_grad / sigma2)


class RandomEffects:
    def __init__(self, re_terms, data, a_covs=None):
        self._create_terms(re_terms, data, a_covs)
        self._construct_matrices()

    def _create_terms(self, re_terms, data, a_covs):

        a_covs = [None] * len(re_terms) if a_covs is None else a_covs
        tterms, theta = [], []
        n_gterms = len(re_terms)
        n_tterms = n_gterms + 1

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
        logdet_C, L22_diag = self.update_chol_diag(theta)
        ytPy = L22_diag[-1] ** 2
        logdetG = self.re_mod.lndet_gmat(theta)
        logdetR = self.re_mod.lndet_rmat(theta)
        if reml:
            logdetXtVinvX = 2.0 * np.sum(np.log(L22_diag[:-1]))
            ll = logdetR + logdet_C + logdetXtVinvX + logdetG + ytPy
        else:
            ll = logdetR + logdet_C + logdetG + ytPy
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
        self.update_crossprods(self.re_mod.theta)
        for k, term in enumerate(self.re_mod.gterms):
            term.cov_structure.precompute_grad_caches(
                self.ZtRZ, self.re_mod.ranef_sl[k])

    def _initialize_matrices(self):
        if self._scalar_resid:
            self._initialize_unweighted()
        else:
            self._initialize_weighted()
        self.C = self.ZtRZ + self.G

    def _initialize_unweighted(self):
        ZtR = self.Zt.dot(self.R).tocsc()
        ZtRdR = ZtR.copy()
        ZtR_dR_RZ = sp.sparse.csc_array.dot(ZtRdR, ZtR.T).tocsc()

        ZtZ = sp.sparse.csc_array.dot(self.Zt, self.Zt.T).tocsc()
        ZtZ.sort_indices()
        ZtRZ = ZtZ.copy()                       # persistent buffer
        ZtRZ.sort_indices()

        ZtXy = sp.sparse.csc_array.dot(self.Zt, self.Xy)
        ZtRXy = ZtXy.copy()                     # ndarray buffer

        XytXy = self.Xy.T.dot(self.Xy)
        XytRXy = XytXy.copy()

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
         self.ZtRXy, self.ZtXy = ZtRXy, ZtXy
         self.XytRXy, self.XytXy = XytRXy, XytXy

    def _setup_cholesky(self):
        self.chol_fac = analyze(self.C, ordering_method="best")
        self._p = np.argsort(self.chol_fac.P())

    def update_crossprods(self, theta):
        if self._scalar_resid:
            self._update_crossprods_scalar(theta)
        else:
            self._update_crossprods_nontrv(theta)

    def _update_crossprods_scalar(self, theta):
        scale = 1.0 / theta[-1]
        np.multiply(self.ZtZ.data, scale, out=self.ZtRZ.data)
        np.multiply(self.ZtXy, scale, out=self.ZtRXy)
        np.multiply(self.XytXy, scale, out=self.XytRXy)

    def _update_crossprods_nontrv(self, theta):
        self._update_rcov(theta, inv=True)
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
        self.chol_fac.cholesky_inplace(C)
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
        self.chol_fac.cholesky_inplace(C)
        logdet_C = float(self.chol_fac.slogdet()[1])
        L21 = self.chol_fac.apply_Pt(
                self.chol_fac.solve_L(
                    self.chol_fac.apply_P(self.ZtRXy), False)).T
        L22_diag = np.diag(np.linalg.cholesky(self.XytRXy - L21.dot(L21.T)))
        return logdet_C, L22_diag

    def _chol_dense_diag(self, C):
        C_dense = C.toarray()
        L11 = np.linalg.cholesky(C_dense)
        logdet_C = 2.0 * float(np.sum(np.log(np.diag(L11))))
        L21 = sp.linalg.solve_triangular(L11, self.ZtRXy, trans=0, lower=True).T
        L22_diag = np.diag(np.linalg.cholesky(self.XytRXy - L21.dot(L21.T)))
        return logdet_C, L22_diag

    def update_chol_diag(self, theta):
        self.update_crossprods(theta)
        Ginv = self.re_mod.update_gcov(theta, inv=True, G=self.G)
        cs_add_inplace(self.ZtRZ, Ginv, self.C)
        if sparsity(self.C) < self.sparse_threshold:
            return self._chol_sparse_diag(self.C)
        return self._chol_dense_diag(self.C)

    def _gradient(self, theta, reml=True):
        self.update_crossprods(theta)
        Ginv = self.re_mod.update_gcov(theta, inv=True, G=self.G)
        C = cs_add_inplace(self.ZtRZ, Ginv, self.C)
        self.chol_fac.cholesky_inplace(C)

        ZtRZ = self.ZtRZ
        ZtRXy = self.ZtRXy
        ZtRX = ZtRXy[:, :-1]
        ZtRy = ZtRXy[:, [-1]]
        XtRX = self.XytRXy[:-1, :-1]
        XtRy = self.XytRXy[:-1, [-1]]

        L_zz = self.chol_fac.apply_Pt(self.chol_fac.solve_L(
            self.chol_fac.apply_P(ZtRZ), False)).T
        L_zxyt = self.chol_fac.apply_Pt(self.chol_fac.solve_L(
            self.chol_fac.apply_P(ZtRXy), False))
        L_zxt = L_zxyt[:, :-1]
        L_zyt = L_zxyt[:, [-1]]

        T_zx = ZtRX - L_zz.dot(L_zxt)
        T_xx = XtRX - L_zxt.T.dot(L_zxt)
        T_xx_inv = np.linalg.inv(T_xx)
        u_xy = XtRy - L_zxt.T.dot(L_zyt)
        u_zy = ZtRy - L_zz.dot(L_zyt)
        v_y = u_zy - T_zx.dot(T_xx_inv.dot(u_xy))
        grad = np.zeros_like(theta)

        for k in range(self.re_mod.n_gterms):
            term = self.re_mod.gterms[k]
            term.cov_structure.accumulate_gradient(
                self.re_mod.theta_sl[k].start, self.re_mod.ranef_sl[k],
                ZtRZ, L_zz, T_zx, v_y, T_xx_inv, reml, grad)


        self.re_mod.tterms[-1].accumulate_gradient(
            self, theta, reml, grad,
            T_xx_inv=T_xx_inv, u_xy=u_xy, v_y=v_y, ZtRX=ZtRX)
        return grad

    def _resid_gradient_trace(self, theta, reml, grad, T_xx_inv, u_xy, v_y, ZtRX):
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
        ZtRT = self.ZtR.T.tocsc()
        XtR = Rinv.dot(X).T

        for i in range(self.re_mod.n_gpar, self.re_mod.n_par):
            dRi = self.re_mod.G_derivs[i]
            cs_matmul_inplace(self.ZtR, dRi, self.ZtRdR)
            cs_matmul_inplace(self.ZtRdR, ZtRT, self.ZtR_dR_RZ)
            ZtR_dR_RZ = self.ZtR_dR_RZ
            T1B = (self.chol_fac.solve_A(
                    sp.sparse.csc_matrix(ZtR_dR_RZ)).diagonal()).sum()

            T1 = T1A - T1B
            T2 = np.dot((dRi.dot(Py)).T, Py)[0, 0]
            if reml:
                T3A = (dRi.dot(XtR.T)).T.dot(XtR.T)
                T3B = -2.0 * sp.sparse.csc_array.dot(((self.ZtRdR).dot(XtR.T)).T, MZtRX)
                T3C = (ZtR_dR_RZ.dot(MZtRX)).T.dot(MZtRX)
                T3 = np.trace(T_xx_inv.dot((T3A + T3B + T3C)))
            else:
                T3 = 0.0
            grad[i] = T1 - T2 - T3

    def _gradient_reparam(self, eta, reml=True):
        theta = self.re_mod.reparam.rvs(eta)
        return self._gradient(theta, reml)

    def _gradient_v2(self, theta, reml=True):
        self.update_crossprods(theta)
        Ginv = self.re_mod.update_gcov(theta, inv=True, G=self.G)
        C = cs_add_inplace(self.ZtRZ, Ginv, self.C)
        self.chol_fac.cholesky_inplace(C)

        ZtRZ = self.ZtRZ
        ZtRXy = self.ZtRXy
        ZtRX = ZtRXy[:, :-1]
        ZtRy = ZtRXy[:, [-1]]
        XtRX = self.XytRXy[:-1, :-1]
        XtRy = self.XytRXy[:-1, [-1]]

        MZtRX = self.chol_fac.solve_A(ZtRX)
        MZtRy = self.chol_fac.solve_A(ZtRy)

        T_zx = ZtRX - ZtRZ.dot(MZtRX)
        T_xx = XtRX - ZtRX.T.dot(MZtRX)
        T_xx_inv = np.linalg.inv(T_xx)
        u_xy = XtRy - ZtRX.T.dot(MZtRy)
        u_zy = ZtRy - ZtRZ.dot(MZtRy)
        v_y = u_zy - T_zx.dot(T_xx_inv.dot(u_xy))

        grad = np.zeros_like(theta)
        for k in range(self.re_mod.n_gterms):
            term = self.re_mod.gterms[k]
            sl = self.re_mod.ranef_sl[k]
            Mk = self.chol_fac.apply_Pt(self.chol_fac.solve_L(
                self.chol_fac.apply_P(ZtRZ[:, sl]), False))
            L_zz_k = Mk.T   # (n_ranef_k, n_ranef): only this term's row block
            term.cov_structure.accumulate_gradient(
                self.re_mod.theta_sl[k].start, sl,
                ZtRZ, None, T_zx, v_y, T_xx_inv, reml, grad,
                L_zz_sl=L_zz_k)

        self.re_mod.tterms[-1].accumulate_gradient(
            self, theta, reml, grad,
            T_xx_inv=T_xx_inv, u_xy=u_xy, v_y=v_y, ZtRX=ZtRX)
        return grad

    def _gradient_v2_reparam(self, eta, reml=True):
        theta = self.re_mod.reparam.rvs(eta)
        return self._gradient_v2(theta, reml)

    def vinv_apply(self, vec, theta):
        if not self._scalar_resid:
            raise NotImplementedError("vinv_apply requires scalar residual")
        inv_s2 = 1.0 / float(theta[-1])
        Rinv_v = vec * inv_s2
        rhs = np.asarray(self.Zt.dot(Rinv_v)).reshape(-1, 1)
        Cinv = np.asarray(self.chol_fac.solve_A(rhs)).reshape(-1)
        return Rinv_v - inv_s2 * np.asarray(self.Z.dot(Cinv)).reshape(-1)

    def p_apply(self, vec, theta, T_xx_inv, reml=True):
        Vv = self.vinv_apply(vec, theta)
        if not reml:
            return Vv
        correction = self.X.dot(T_xx_inv.dot(self.X.T.dot(Vv)))
        return Vv - self.vinv_apply(correction, theta)




class _VarCorrReparam:

    def __init__(self, re_mod):
        diag_ix, off_ix, off_da, off_db = [], [], [], []
        for k in range(re_mod.n_gterms):
            term = re_mod.gterms[k]
            ucov = term.cov_structure.unstructured_cov
            sl = re_mod.theta_sl[k]
            ab_to_local = {(int(ucov.r_inds[j]), int(ucov.c_inds[j])): j
                           for j in range(ucov.n_pars)}
            for j in range(ucov.n_pars):
                a, b = int(ucov.r_inds[j]), int(ucov.c_inds[j])
                if a == b:
                    diag_ix.append(sl.start + j)
                else:
                    off_ix.append(sl.start + j)
                    off_da.append(sl.start + ab_to_local[(a, a)])
                    off_db.append(sl.start + ab_to_local[(b, b)])
        self.diag_ix = np.asarray(diag_ix, dtype=np.int64)
        self.off_ix = np.asarray(off_ix, dtype=np.int64)
        self.off_da = np.asarray(off_da, dtype=np.int64)
        self.off_db = np.asarray(off_db, dtype=np.int64)

    def fwd(self, theta):
        tau = np.asarray(theta, dtype=float).copy()
        if len(self.off_ix):
            d_a = theta[self.off_da]
            d_b = theta[self.off_db]
            denom = np.sqrt(np.maximum(d_a * d_b, 1e-30))
            corr = np.clip(theta[self.off_ix] / denom, -0.999999, 0.999999)
            tau[self.off_ix] = np.arctanh(corr)
        return tau

    def rvs(self, tau):
        theta = np.asarray(tau, dtype=float).copy()
        if len(self.off_ix):
            d_a = tau[self.off_da]
            d_b = tau[self.off_db]
            theta[self.off_ix] = (np.sqrt(np.maximum(d_a * d_b, 0.0))
                                  * np.tanh(tau[self.off_ix]))
        return theta

    def jac_rvs(self, tau):
        n = int(np.asarray(tau).shape[0])
        J = np.eye(n)
        if not len(self.off_ix):
            return J
        o, a, b = self.off_ix, self.off_da, self.off_db
        ta, tb, to = tau[a], tau[b], tau[o]
        safe = (ta > 0) & (tb > 0)
        tiny = np.finfo(float).tiny
        sqrt_ab = np.where(safe, np.sqrt(ta * tb), 0.0)
        tanh_o = np.tanh(to)
        sech2 = 1.0 - tanh_o * tanh_o
        ratio_ba = np.sqrt(np.where(safe, tb / np.maximum(ta, tiny), 0.0))
        ratio_ab = np.sqrt(np.where(safe, ta / np.maximum(tb, tiny), 0.0))
        J[o, o] = sqrt_ab * sech2
        J[o, a] = 0.5 * tanh_o * ratio_ba
        J[o, b] = 0.5 * tanh_o * ratio_ab
        return J

    def jvp_T(self, tau, g):
        g_tau = np.asarray(g, dtype=float).copy()
        if not len(self.off_ix):
            return g_tau
        o, a, b = self.off_ix, self.off_da, self.off_db
        ta, tb, to = tau[a], tau[b], tau[o]
        safe = (ta > 0) & (tb > 0)
        tiny = np.finfo(float).tiny
        sqrt_ab = np.where(safe, np.sqrt(ta * tb), 0.0)
        tanh_o = np.tanh(to)
        sech2 = 1.0 - tanh_o * tanh_o
        ratio_ba = np.sqrt(np.where(safe, tb / np.maximum(ta, tiny), 0.0))
        ratio_ab = np.sqrt(np.where(safe, ta / np.maximum(tb, tiny), 0.0))
        g_tau[o] = g[o] * sqrt_ab * sech2
        np.add.at(g_tau, a, g[o] * 0.5 * tanh_o * ratio_ba)
        np.add.at(g_tau, b, g[o] * 0.5 * tanh_o * ratio_ab)
        return g_tau



class LMM2(object):

    def __init__(self, formula, data, mme_kws=None):

        model_info = parse_random_effects(formula)
        re_terms = model_info["re_terms"]
        re_mod = RandomEffects(re_terms, data=data)
        y_vars = model_info["y_vars"]
        fe_form = model_info["fe_form"]
        X_df = formulaic.model_matrix(fe_form, data)
        X = X_df.values
        y = data[y_vars].values.reshape(-1, 1)
        mme_kws = {} if mme_kws is None else mme_kws
        self.mme = MMEBlocked(X, y, re_mod, **mme_kws)
        self.formula = formula
        self.fe_form = fe_form
        self.y_vars = y_vars
        self.fe_names = list(X_df.columns)
        self.re_terms = re_terms

    def loglike(self, theta, reml=True):
        return self.mme._loglike(theta, reml)

    def gradient(self, theta, reml=True):
        return self.mme._gradient(theta, reml)

    def gradient_v2(self, theta, reml=True):
        return self.mme._gradient_v2(theta, reml)

    def loglike_reparam(self, eta, reml=True):
        return self.mme._loglike_reparam(eta, reml)

    def gradient_reparam(self, eta, reml=True):
        dl_dtheta = self.mme._gradient_reparam(eta, reml)
        dtheta_deta = self.mme.re_mod.reparam.jac_rvs(eta)
        g = np.dot(dl_dtheta, dtheta_deta)
        return g

    def gradient_v2_reparam(self, eta, reml=True):
        dl_dtheta = self.mme._gradient_v2_reparam(eta, reml)
        dtheta_deta = self.mme.re_mod.reparam.jac_rvs(eta)
        return np.dot(dl_dtheta, dtheta_deta)


    def _factor_C(self, theta):
        self.mme.update_crossprods(theta)
        Ginv = self.mme.re_mod.update_gcov(theta, inv=True, G=self.mme.G)
        C = cs_add_inplace(self.mme.ZtRZ, Ginv, self.mme.C)
        self.mme.chol_fac.cholesky_inplace(C)
        return Ginv

    def compute_effects(self, theta=None):
        theta = self.theta if theta is None else theta
        self._factor_C(theta)
        ZtRX = self.mme.ZtRXy[:, :-1]
        ZtRy = self.mme.ZtRXy[:, [-1]]
        XtRX = self.mme.XytRXy[:-1, :-1]
        XtRy = self.mme.XytRXy[:-1, [-1]]
        ZtRZ = self.mme.ZtRZ
        M_X = self.mme.chol_fac.solve_A(ZtRX)
        M_y = self.mme.chol_fac.solve_A(ZtRy)
        T_xx = XtRX - ZtRX.T.dot(M_X)
        T_xx_inv = np.linalg.inv(T_xx)
        u_xy = XtRy - ZtRX.T.dot(M_y)
        u_zy = ZtRy - ZtRZ.dot(M_y)
        T_zx = ZtRX - ZtRZ.dot(M_X)
        v_y = u_zy - T_zx.dot(T_xx_inv.dot(u_xy))
        beta = T_xx_inv.dot(u_xy).reshape(-1)
        G = self.mme.re_mod.update_gcov(theta, inv=False, G=self.mme.G)
        u = G.dot(v_y).reshape(-1)
        return beta, T_xx_inv, u

    def predict(self, X=None, Z=None, beta=None, u=None):
        X = self.mme.X if X is None else X
        Z = self.mme.Z if Z is None else Z
        beta = self.beta if beta is None else beta
        u = self.u if u is None else u
        return X.dot(beta) + Z.dot(u)

    def vinvcrossprod(self, A, B, theta=None):
        theta = self.theta if theta is None else theta
        self._factor_C(theta)
        self.mme._update_rcov(theta, inv=True)
        Rinv = self.mme.R
        ZtRA = self.mme.Zt.dot(Rinv.dot(A))
        ZtRB = self.mme.Zt.dot(Rinv.dot(B))
        M_B = self.mme.chol_fac.solve_A(ZtRB)
        return A.T.dot(Rinv.dot(B)) - ZtRA.T.dot(M_B)

    def fit(self, reml=True, method='l-bfgs-b', opt_kws=None,
            theta_init=None):
        re_mod = self.mme.re_mod
        theta_init = re_mod.theta if theta_init is None else theta_init
        eta_init = re_mod.reparam.fwd(theta_init)
        opt_kws = {} if opt_kws is None else opt_kws
        opt = sp.optimize.minimize(self.loglike_reparam, eta_init, args=(reml,),
                                   jac=self.gradient_reparam, method=method,
                                   options=opt_kws)
        theta_hat = re_mod.reparam.rvs(opt.x)
        self._post_fit(theta_hat, opt, reml)
        return self

    def _post_fit(self, theta_hat, opt, reml):
        self.theta = theta_hat
        self.eta = opt.x
        self.opt = opt
        self.reml = reml
        self.beta, self.XtVinvX_inv, self.u = self.compute_effects(theta_hat)
        self.se_beta = np.sqrt(np.diag(self.XtVinvX_inv))

        H = so_gc_cd(self.gradient, theta_hat, args=(reml,))
        self.H_theta = H
        self.Hinv_theta = np.linalg.pinv(H / 2.0)
        self.se_theta = np.sqrt(np.diag(self.Hinv_theta))

        n_obs = self.mme.n_obs
        n_fe = self.mme.n_fixef
        n_pars = len(theta_hat)
        nll = float(opt.fun)
        if reml:
            llconst = (n_obs - n_fe) * np.log(2 * np.pi)
            n_eff, d = n_obs - n_fe, n_pars
        else:
            llconst = n_obs * np.log(2 * np.pi)
            n_eff, d = n_obs, n_fe + n_pars
        self.llconst, self.nll = llconst, nll
        self.ll = llconst + nll
        self.llf = -self.ll / 2.0
        self.AIC = self.ll + 2 * d
        self.AICC = self.ll + 2 * d * n_eff / (n_eff - d - 1)
        self.BIC = self.ll + d * np.log(n_eff)
        self.CAIC = self.ll + d * (np.log(n_eff) + 1)

        re_mod = self.mme.re_mod
        self.re_covs, self.re_corrs = {}, {}
        param_names = list(self.fe_names)
        for k in range(re_mod.n_gterms):
            G_k = invech(theta_hat[re_mod.theta_sl[k]])
            self.re_covs[k] = G_k
            v = np.diag(np.sqrt(1 / np.diag(G_k)))
            self.re_corrs[k] = v.dot(G_k).dot(v)
            term = re_mod.gterms[k]
            ucov = term.cov_structure.unstructured_cov
            for j in range(ucov.n_pars):
                a, b = ucov.r_inds[j], ucov.c_inds[j]
                param_names.append(f"{term.re_grouping}:G[{a}][{b}]")
        param_names.append('resid_cov')
        self.param_names = param_names
        self.params = np.concatenate([self.beta, theta_hat])
        self.se_params = np.concatenate([self.se_beta, self.se_theta])
        res = pd.DataFrame(np.vstack([self.params, self.se_params]).T,
                           index=param_names, columns=['estimate', 'SE'])
        res['t'] = res['estimate'] / res['SE']
        res['p'] = sp.stats.t(n_obs - n_fe).sf(np.abs(res['t']))
        res['degfree'] = float(n_obs - n_fe)
        self.res = res
        self.sumstats = pd.DataFrame(
            {'value': [self.ll, self.llf, self.AIC, self.AICC, self.BIC, self.CAIC]},
            index=['ll', 'llf', 'AIC', 'AICC', 'BIC', 'CAIC'])


    def confint(self, alpha=0.05):
        df = self.mme.n_obs - self.mme.n_fixef
        q = sp.stats.t(df).ppf(1.0 - alpha / 2.0)
        lo = self.beta - q * self.se_beta
        hi = self.beta + q * self.se_beta
        return pd.DataFrame({'lo': lo, 'hi': hi}, index=self.fe_names)

    def random_effects(self, as_frame='per_term'):
        re_mod = self.mme.re_mod
        if as_frame == 'per_term':
            out = {}
            for k in range(re_mod.n_gterms):
                term = re_mod.gterms[k]
                sl = re_mod.ranef_sl[k]
                nv, ng = term.n_revars, term.n_levels
                cols = [f'v{j}' for j in range(nv)]
                out[k] = pd.DataFrame(self.u[sl].reshape(ng, nv), columns=cols)
            return out
        if as_frame == 'multi':
            frames = []
            for k in range(re_mod.n_gterms):
                term = re_mod.gterms[k]
                sl = re_mod.ranef_sl[k]
                nv, ng = term.n_revars, term.n_levels
                cols = [f'v{j}' for j in range(nv)]
                df = pd.DataFrame(self.u[sl].reshape(ng, nv), columns=cols)
                df.index = pd.MultiIndex.from_product(
                    [[k], np.arange(ng)], names=['term', 'level'])
                frames.append(df)
            return pd.concat(frames)
        if as_frame == 'long':
            se = self.random_effects_se()
            rows = []
            for k in range(re_mod.n_gterms):
                term = re_mod.gterms[k]
                sl = re_mod.ranef_sl[k]
                nv, ng = term.n_revars, term.n_levels
                u_blk = self.u[sl].reshape(ng, nv)
                se_blk = se[sl].reshape(ng, nv)
                for j in range(ng):
                    for a in range(nv):
                        rows.append((k, j, f'v{a}', u_blk[j, a], se_blk[j, a]))
            return pd.DataFrame(rows, columns=['term', 'level', 'var',
                                                'estimate', 'SE']).set_index(
                ['term', 'level', 'var'])
        raise ValueError(f"unknown as_frame={as_frame!r}")

    def random_effects_array(self):
        return np.asarray(self.u)

    def random_effects_se(self, theta=None, full=True):
        theta = self.theta if theta is None else theta
        self._factor_C(theta)
        ZtRX = self.mme.ZtRXy[:, :-1]
        M_X = self.mme.chol_fac.solve_A(ZtRX)
        T_xx_inv = self.XtVinvX_inv
        var_x = np.einsum('ik,kl,il->i', M_X, T_xx_inv, M_X, optimize=True)

        if not full:
            return np.sqrt(np.clip(var_x, 0.0, None))
        n_ranef = self.mme.Z.shape[1]
        diag_Cinv = np.zeros(n_ranef)
        e = np.zeros((n_ranef, 1))
        for i in range(n_ranef):
            e[i, 0] = 1.0
            x = self.mme.chol_fac.solve_A(e)
            diag_Cinv[i] = float(np.asarray(x).reshape(-1)[i])
            e[i, 0] = 0.0
        return np.sqrt(np.clip(diag_Cinv + var_x, 0.0, None))

    def variance_decomposition(self, theta=None):
        theta = self.theta if theta is None else theta
        eta_fe = np.asarray(self.mme.X.dot(self.beta)).reshape(-1)
        v_fe = float(np.var(eta_fe))
        re_mod = self.mme.re_mod
        G_full = re_mod.update_gcov(theta)
        Z = re_mod.Z
        n = self.mme.n_obs
        v_re = {}
        for k in range(re_mod.n_gterms):
            sl = re_mod.ranef_sl[k]
            Z_k = Z[:, sl]
            G_k = G_full[sl, sl]
            ZG_k = Z_k.dot(G_k)
            v_re[k] = float(Z_k.multiply(ZG_k).sum() / n)
        v_resid = float(theta[-1])
        return {'v_fe': v_fe, 'v_re': v_re, 'v_resid': v_resid}

    def r_squared(self, theta=None):
        comps = self.variance_decomposition(theta=theta)
        v_fe = comps['v_fe']
        v_re = comps['v_re']
        v_resid = comps['v_resid']
        total = v_fe + sum(v_re.values()) + v_resid
        return {
            'marginal':    v_fe / total,
            'conditional': (v_fe + sum(v_re.values())) / total,
            'per_term':    {k: v / total for k, v in v_re.items()},
            'components':  comps,
            'total':       total,
        }

    def hessian(self, theta=None, reml=True):
        raise NotImplementedError

    def _dC_dtheta(self, theta, C):
        re_mod = self.mme.re_mod
        X, Z, Zt = self.mme.X, self.mme.Z, self.mme.Zt
        self._factor_C(theta)
        self.mme._update_rcov(theta, inv=True)
        Rinv = self.mme.R
        chol_fac = self.mme.chol_fac
        RinvX = Rinv.dot(X)
        ZtRinvX = Zt.dot(RinvX)
        M_ZtRinvX = chol_fac.solve_A(ZtRinvX)
        J = []
        for k in range(re_mod.n_gterms):
            term = re_mod.gterms[k]
            sl = re_mod.ranef_sl[k]
            Zk = Z[:, sl]
            RinvZk = Rinv.dot(Zk)
            XtRZk = X.T.dot(RinvZk.toarray() if sp.sparse.issparse(RinvZk) else RinvZk)
            ZtRZk = Zt.dot(RinvZk)
            M_ZtRZk = chol_fac.solve_A(ZtRZk)
            if sp.sparse.issparse(M_ZtRZk):
                M_ZtRZk = M_ZtRZk.toarray()
            XtVinvZk = XtRZk - ZtRinvX.T.dot(M_ZtRZk)
            CXtVZk = C.dot(XtVinvZk)
            for dGj in term.cov_structure.G_derivs:
                J.append(CXtVZk.dot(dGj.dot(CXtVZk.T)))
        RinvZ_MZX = Rinv.dot(Z).dot(M_ZtRinvX)
        if sp.sparse.issparse(RinvZ_MZX):
            RinvZ_MZX = RinvZ_MZX.toarray()
        VinvX = RinvX - RinvZ_MZX
        CVinvX = C.dot(VinvX.T)
        J.append(CVinvX.dot(CVinvX.T))
        return J

    def approx_degfree(self, L_list=None, theta=None, beta=None,
                       method='satterthwaite', reml=True):
        n_fe = self.mme.n_fixef
        L_list = [np.eye(n_fe)[[i]] for i in range(n_fe)] if L_list is None else L_list
        theta = self.theta if theta is None else theta
        beta = self.beta if beta is None else beta
        self._factor_C(theta)
        C = np.linalg.inv(self.vinvcrossprod(self.mme.X, self.mme.X, theta))
        if hasattr(self, 'Hinv_theta'):
            Vtheta = self.Hinv_theta
        else:
            Vtheta = np.linalg.pinv(so_gc_cd(self.gradient, theta, args=(reml,)) / 2.0)
        J = self._dC_dtheta(theta, C)
        out = []
        for L in L_list:
            L = np.atleast_2d(np.asarray(L, dtype=float))
            LCLt = L.dot(C).dot(L.T)
            u, Q = np.linalg.eigh(LCLt)
            order = np.argsort(u)[::-1]
            u, Q = u[order], Q[:, order]
            q = int(np.linalg.matrix_rank(L))
            P = Q.T.dot(L)
            t2 = (P.dot(beta)) ** 2 / np.maximum(u, np.finfo(float).tiny)
            f = float(np.sum(t2) / q)
            D = np.array([[float(x.dot(Ji).dot(x)) for Ji in J] for x in P[:q]])
            nu_d = np.array([D[i].dot(Vtheta).dot(D[i]) for i in range(q)])
            nu_m = u[:q] ** 2 / np.maximum(nu_d, np.finfo(float).tiny)
            keep = nu_m > 2
            E = float(np.sum(nu_m[keep] / (nu_m[keep] - 2.0)))
            nu = float(2.0 * E / (E - q)) if E > q else np.nan
            p = float(sp.stats.f(q, nu).sf(f)) if np.isfinite(nu) else np.nan
            out.append({'F': f, 'df1': q, 'df2': nu, 'p': p})
        return out

    def profile(self, n_points=11, tb=3.0):
        if not hasattr(self, 'theta'):
            raise RuntimeError("Call .fit() before .profile().")
        reparam = _VarCorrReparam(self.mme.re_mod)
        tau_hat = reparam.fwd(self.theta)
        n_theta = len(tau_hat)
        var_set = set(reparam.diag_ix.tolist()) | {n_theta - 1}
        llmax = self.nll
        J_inv = np.linalg.inv(reparam.jac_rvs(tau_hat))
        var_tau = J_inv.dot(self.Hinv_theta).dot(J_inv.T)
        se_tau = np.sqrt(np.diag(var_tau))
        thetas = np.zeros((n_theta * n_points, n_theta))
        zetas = np.zeros(n_theta * n_points)
        k = 0
        for i in range(n_theta):
            t_mle = tau_hat[i]
            width = tb * max(se_tau[i], 1e-3)
            lb = max(t_mle - width, 1e-6) if i in var_set else t_mle - width
            ub = t_mle + width
            for t0 in np.linspace(lb, ub, n_points):
                theta_r, ll_r = self._fit_with_fixed_tau(
                    reparam, tau_hat, i, t0)
                LR = max(ll_r - llmax, 0.0)
                zetas[k] = np.sqrt(LR) * np.sign(t0 - t_mle)
                thetas[k] = theta_r
                k += 1
        ix = np.repeat(np.arange(n_theta), n_points)
        return thetas, zetas, ix

    def _fit_with_fixed_tau(self, reparam, tau_init, fixed_idx, fixed_value,
                            method='trust-constr'):
        n = len(tau_init)
        free = np.ones(n, dtype=bool)
        free[fixed_idx] = False
        var_set = set(reparam.diag_ix.tolist()) | {n - 1}
        bounds = []
        for i in range(n):
            if not free[i]:
                continue
            if i in var_set:
                bounds.append((1e-6, None))
            else:
                bounds.append((None, None))

        reml_flag = self.reml

        def fg(tau_free):
            tau = tau_init.copy()
            tau[free] = tau_free
            tau[fixed_idx] = fixed_value
            theta = reparam.rvs(tau)
            ll = self.loglike(theta, reml=reml_flag)
            g_theta = self.gradient(theta, reml=reml_flag)
            g_tau = reparam.jvp_T(tau, g_theta)
            return ll, g_tau[free]

        x0 = tau_init[free].copy()
        opt = sp.optimize.minimize(fg, x0, jac=True, method=method,
                                   bounds=bounds,options=dict(verbose=3,
                                                              initial_tr_radius=0.5))
        tau_r = tau_init.copy()
        tau_r[free] = opt.x
        tau_r[fixed_idx] = fixed_value
        return reparam.rvs(tau_r), float(opt.fun)

    def plot_profile(self, thetas, zetas, ix, quantiles=None, figsize=(14, 4)):
        if quantiles is None:
            quantiles = np.array([60, 70, 80, 90, 95, 99])
            quantiles = np.concatenate([(100 - quantiles[::-1]) / 2,
                                        100 - (100 - quantiles) / 2])
        theta = self.theta.copy()
        se_theta = self.se_theta.copy()
        n_thetas = thetas.shape[1]
        q = sp.stats.norm(0, 1).ppf(np.array(quantiles) / 100)
        fig, axes = plt.subplots(figsize=figsize, ncols=n_thetas, sharey=True)
        plt.subplots_adjust(wspace=0.05, left=0.05, right=0.95)
        for i in range(n_thetas):
            ax = axes[i]
            x = thetas[ix == i, i]
            y = zetas[ix == i]
            trunc = (y > -5) & (y < 5)
            x, y = x[trunc], y[trunc]
            f_interp = interp1d(y, x, fill_value='extrapolate')
            xq = f_interp(q)
            ax.plot(x, y)
            ax.set_xlim(x.min(), x.max())
            ax.axhline(0, color='k')
            sgs = np.zeros((len(q), 2, 2))
            sgs[:, 0, 0] = sgs[:, 1, 0] = xq
            sgs[:, 1, 1] = q
            xqt = theta[i] + q * se_theta[i]
            ax.axvline(theta[i], color='k')
            norm = mpl.colors.TwoSlopeNorm(vcenter=0, vmin=q.min(), vmax=q.max())
            lc = mpl.collections.LineCollection(sgs, cmap=plt.cm.bwr, norm=norm)
            lc.set_array(q)
            lc.set_linewidth(2)
            ax.add_collection(lc)
            ax.scatter(xqt, np.zeros_like(xqt), c=q, cmap=plt.cm.bwr,
                       norm=norm, s=20)
        ax.set_ylim(-5, 5)
        return fig, axes

    def profile_confint(self, thetas, zetas, ix, alpha=0.05):
        z = float(sp.stats.norm.ppf(1.0 - alpha / 2.0))
        n_thetas = thetas.shape[1]
        lo, hi = np.zeros(n_thetas), np.zeros(n_thetas)
        for i in range(n_thetas):
            x = thetas[ix == i, i]
            y = zetas[ix == i]
            f_inv = interp1d(y, x, kind='linear', fill_value='extrapolate')
            lo[i], hi[i] = float(f_inv(-z)), float(f_inv(z))
        return pd.DataFrame({'lo': lo, 'hi': hi})

    def lrt(self, other):
        if self.reml != getattr(other, 'reml', None):
            raise ValueError("LRT requires both models fit with same reml")
        df = max(0, self.theta.size - other.theta.size)
        stat = -(self.ll - other.ll)
        p = sp.stats.chi2(df).sf(stat) if df > 0 else 1.0
        return pd.DataFrame({'stat': [stat], 'df': [df], 'p': [p]})

