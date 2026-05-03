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
from abc import ABCMeta, abstractmethod

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
    """Return (csc a_cov, csc a_inv) using a Cholesky factorization. Sparse
    a_cov is inverted via sksparse cholmod against a sparse identity; dense
    a_cov goes through cho_solve. Replaces the prior `pinv(a_cov.toarray())`
    path which failed quietly for indefinite or large matrices."""
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

class _DataView:
    """Minimal stand-in for a sparse matrix exposing only `.data`. The C-level
    *_kron_inplace kernels look at the third arg's `.data` attribute and
    nothing else; wrapping the output buffer here lets us keep the canonical
    `_update_gdata(G0, inv, out)` contract (out = data array) without forcing
    the kernels to know about ndarray vs sparse-matrix."""
    __slots__ = ('data',)
    def __init__(self, data):
        self.data = data


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
        """Write the structured covariance entries into the data buffer
        `out` (a 1-D ndarray with the same layout as `self.G.data`)."""
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
        """Refresh G's data in place at the given params. Default uses
        `_update_gdata` against G's data buffer; subclasses with non-standard
        layouts can override."""
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
        """Hook for one-time, structure-aware caches that need ZtRZ's pattern.
        Default does nothing; subclasses override when they can amortize work."""
        pass

    def accumulate_gradient(self, par_offset, sl, ZtRZ, L_zz, T_zx, v_y,
                            T_xx_inv, reml, grad):
        """Compute grad[par_offset : par_offset + n_pars] for this term.
        Default: per-parameter contraction using self.G_derivs. Subclasses
        override for structure-aware fast paths (e.g. block-diagonal)."""
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
        """Cache positions in ZtRZ.data for the (j, a, b) entries of each
        diagonal block. KronIG within-term means each column has exactly nv
        in-term rows at [offset+j*nv .. offset+(j+1)*nv), block-diagonal across
        levels. Cross-term coupling (other G-terms in a multi-term Z) puts
        rows < offset before the in-term run after sort_indices, so we find
        the in-term start per column via searchsorted."""
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
        """If `L_zz_sl` is supplied it is the row-slice of L_zz already
        restricted to this term — used by the v2 gradient path to avoid ever
        materializing global L_zz. When None, slice from the global L_zz
        (legacy v1 path)."""
        ng, nv = self.n_lv, self.n_rv

        # _zr_index_map is built once in MMEBlocked.__init__; if it's missing,
        # we have a setup bug — let AttributeError fire rather than silently
        # falling back to a slower path.
        ZtRZ_sum = ZtRZ.data[self._zr_index_map].sum(axis=0)

        L_blk = (L_zz_sl if L_zz_sl is not None else L_zz[sl]).tocsr()
        LL_sum = block_diag_self_dot(L_blk, ng, nv)

        T_blk = np.asarray(T_zx[sl].toarray() if sp.sparse.issparse(T_zx) else T_zx[sl])
        T3 = T_blk.reshape(ng, nv, -1)
        v_blk = np.asarray(v_y[sl].toarray() if sp.sparse.issparse(v_y) else v_y[sl]).reshape(ng, nv)
        bb_sum = v_blk.T.dot(v_blk)
        M_sum = ZtRZ_sum - LL_sum - bb_sum
        if reml:
            M_sum = M_sum - np.einsum('jap,pq,jbq->ab', T3, T_xx_inv, T3,
                                      optimize=True)
        # np.matrix leaks from upstream sparse-dense ops; strip so the
        # element-wise multiply below doesn't dispatch to matrix.__mul__.
        M_sum = np.asarray(getattr(M_sum, 'A', M_sum))

        ucov = self.unstructured_cov
        ri, ci = ucov.r_inds, ucov.c_inds
        mult = np.where(ucov.d_mask, 1.0, 2.0)
        grad[par_offset:par_offset + ucov.n_pars] = M_sum[ri, ci] * mult


#TODO: class KronGI(BaseProductCovariance):



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
        # `out` is a data buffer; wrap so the kernel's `C.data` access works.
        A = self.a_inv if inv else self.a_cov
        sparse_dense_kron_inplace(A, G0, _DataView(out))

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
        ds_kron_inplace(G0, A, _DataView(out))

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
        # When `out` is None we mutate self.G.data in place; return that buffer
        # rather than the None we were handed, so the value is usable.
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
        # When `out` is None we mutate self.G.data in place; return that buffer
        # rather than the None we were handed.
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
        """Residual-variance gradient (scalar i.i.d. R = σ² I) via the
        γ-reparametrization identity. With γ_k = θ_G,k / σ², the σ² piece of
        -2 log L has the closed form

            ∂(-2 log L)/∂σ² (γ-param) = n_eff/σ² − y'r/σ⁴

        where n_eff = n−p (REML) or n (ML) and r = y − Xβ̂ − Zû. Chain rule
        back to (θ_G, σ²) gives

            ∂f/∂σ² = n_eff/σ² − y'r/σ⁴ − (1/σ²) θ_G · g_θ_G

        so we never form solve_A on the n_ranef×n_ranef block. Identity
        requires G linear in θ_G (vech parametrization)."""
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
        # `resid_cov` was a placeholder for non-scalar residual covariance and
        # was never wired up; removed to keep the API honest. Adding it back
        # means populating the residual term branch below with a real class.
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
        self.update_crossprods(self.re_mod.theta)
        # ZtRZ already sorted in _initialize_unweighted (scalar resid path);
        # for the non-trivial path it comes out sorted from cs_matmul_inplace.
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
        # ZtZ / ZtXy / XytXy are theta-independent (R = σ²I drops out into a
        # scalar). We sort ZtZ once and keep persistent ZtRZ / ZtRXy / XytRXy
        # buffers with the same structure; _update_crossprods_scalar then
        # writes data in place via np.multiply, avoiding both allocation and
        # the per-call sort_indices that re-running `ZtZ * scale` triggers.
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
        self.dg = np.zeros(self.n_fixef + self.n_ranef + 1, dtype=np.double)

    def update_crossprods(self, theta):
        if self._scalar_resid:
            self._update_crossprods_scalar(theta)
        else:
            self._update_crossprods_nontrv(theta)

    def _update_crossprods_scalar(self, theta):
        # In-place data updates on the persistent ZtRZ / ZtRXy / XytRXy buffers
        # set up in _initialize_unweighted. Avoids reallocating sparse objects
        # and the sort_indices that ZtZ * scalar would otherwise trigger.
        scale = 1.0 / theta[-1]
        np.multiply(self.ZtZ.data, scale, out=self.ZtRZ.data)
        np.multiply(self.ZtXy, scale, out=self.ZtRXy)
        np.multiply(self.XytXy, scale, out=self.XytRXy)

    def _update_crossprods_nontrv(self, theta):
        # ZtRZ etc. need R^{-1} (the weighted normal-equations form), so the
        # _update_rcov call must invert R in-place before downstream products.
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
        L_zxyt = self.chol_fac.apply_Pt(self.chol_fac.solve_L(
            self.chol_fac.apply_P(sp.sparse.csc_matrix(ZtRXy)), False))

        L_zxt = L_zxyt[:, :-1]
        # List-index keeps 2D in scipy 1.15+ where csc_array[:, -1] now returns 1D coo_array.
        L_zyt = L_zxyt[:, [-1]]
        L_zx = L_zxt.T.tocsc()
        L_zy = L_zyt.T.tocsc()

        T_zx = ZtRX - L_zz.dot(L_zx.T)
        T_xx = XtRX - L_zx.dot(L_zx.T)
        T_xx_inv = np.linalg.inv(T_xx)
        u_xy = XtRy - L_zx.dot(L_zy.T)
        u_zy = ZtRy - L_zz.dot(L_zy.T)
        v_y = u_zy - T_zx.dot(T_xx_inv.dot(u_xy))
        grad = np.zeros_like(theta)

        #TODO: get rid of this for loop
        for k in range(self.re_mod.n_gterms):
            term = self.re_mod.gterms[k]
            term.cov_structure.accumulate_gradient(
                self.re_mod.theta_sl[k].start, self.re_mod.ranef_sl[k],
                ZtRZ, L_zz, T_zx, v_y, T_xx_inv, reml, grad)

        # Residual gradient: dispatched on the residual term's structure.
        # DiagResidualCovTerm uses a closed-form γ-reparam identity (cheap);
        # any future general residual term should override this with the
        # trace-based path below (`_resid_gradient_trace`).
        self.re_mod.tterms[-1].accumulate_gradient(
            self, theta, reml, grad,
            T_xx_inv=T_xx_inv, u_xy=u_xy, v_y=v_y, ZtRX=ZtRX)
        return grad

    def _resid_gradient_trace(self, theta, reml, grad, T_xx_inv, u_xy, v_y, ZtRX):
        """General trace-based residual gradient. One solve_A on the
        n_ranef×n_ranef block per residual parameter. This is the fallback
        for residual covariance structures that aren't a scalar variance —
        kept here so future ResidualCovTerm subclasses can delegate to it."""
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

    # ---- v2 gradient: per-term Schur form (skip global L_zz materialization)

    def _gradient_v2(self, theta, reml=True):
        """Same gradient as `_gradient`, computed without ever forming the
        global L_zz = solve_L(P · ZtRZ). Uses Henderson-Schur identities

            T_zx = ZtRX − ZtRZ · MZtRX
            T_xx = XtRX − ZtRX' · MZtRX
            u_xy = XtRy − ZtRX' · MZtRy
            u_zy = ZtRy − ZtRZ · MZtRy

        with MZtRX = solve_A(ZtRX), MZtRy = solve_A(ZtRy) — both small RHS.
        For each G-term k we solve_L on ZtRZ[:, sl_k] only (its column block),
        avoiding the global n_ranef × n_ranef solve_L. Sum of per-term solve_L
        cost is the same as the global solve_L total work, but skips
        intermediate sparse-matrix object construction and L_zz slicing in
        accumulate_gradient. Math fidelity is exact."""
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

        # Schur substitutions: solve_A on small dense RHS (n_ranef × n_fixef
        # and n_ranef × 1) — cheap. Returns dense ndarray.
        MZtRX = self.chol_fac.solve_A(np.asarray(ZtRX))
        MZtRX = np.asarray(MZtRX.toarray() if sp.sparse.issparse(MZtRX) else MZtRX)
        MZtRy = self.chol_fac.solve_A(np.asarray(ZtRy))
        MZtRy = np.asarray(MZtRy.toarray() if sp.sparse.issparse(MZtRy) else MZtRy)

        T_zx = np.asarray(ZtRX) - np.asarray(ZtRZ.dot(MZtRX))
        T_xx = np.asarray(XtRX) - np.asarray(ZtRX).T.dot(MZtRX)
        T_xx_inv = np.linalg.inv(T_xx)
        u_xy = np.asarray(XtRy) - np.asarray(ZtRX).T.dot(MZtRy)
        u_zy = np.asarray(ZtRy) - np.asarray(ZtRZ.dot(MZtRy))
        v_y = u_zy - T_zx.dot(T_xx_inv.dot(u_xy))

        grad = np.zeros_like(theta)
        for k in range(self.re_mod.n_gterms):
            term = self.re_mod.gterms[k]
            sl = self.re_mod.ranef_sl[k]
            ZtRZ_k = ZtRZ[:, sl]
            Mk = self.chol_fac.apply_Pt(self.chol_fac.solve_L(
                self.chol_fac.apply_P(sp.sparse.csc_matrix(ZtRZ_k)), False))
            L_zz_k = Mk.T  # shape (n_ranef_k, n_ranef) — only this term's rows
            term.cov_structure.accumulate_gradient(
                self.re_mod.theta_sl[k].start, sl,
                ZtRZ, None, T_zx, v_y, T_xx_inv, reml, grad,
                L_zz_sl=L_zz_k)

        # Residual gradient via the same dispatch as v1.
        self.re_mod.tterms[-1].accumulate_gradient(
            self, theta, reml, grad,
            T_xx_inv=T_xx_inv, u_xy=u_xy, v_y=v_y, ZtRX=ZtRX)
        return grad

    def _gradient_v2_reparam(self, eta, reml=True):
        theta = self.re_mod.reparam.rvs(eta)
        return self._gradient_v2(theta, reml)

    # ---- V^{-1} and P operators (caller must refresh chol_fac at theta) -----

    def vinv_apply(self, vec, theta):
        """Apply V^{-1} to vec via Woodbury, where V = ZGZ' + R. Assumes
        scalar residual (R = sigma2 I) and that chol_fac currently factors
        C = Z'R^{-1}Z + G^{-1} at this theta. sigma2 is read from theta[-1]."""
        if not self._scalar_resid:
            raise NotImplementedError("vinv_apply requires scalar residual")
        inv_s2 = 1.0 / float(theta[-1])
        Rinv_v = vec * inv_s2
        rhs = np.asarray(self.Zt.dot(Rinv_v)).reshape(-1, 1)
        Cinv = self.chol_fac.solve_A(rhs)
        Cinv = np.asarray(Cinv.toarray() if sp.sparse.issparse(Cinv) else Cinv).reshape(-1)
        return Rinv_v - inv_s2 * np.asarray(self.Z.dot(Cinv)).reshape(-1)

    def p_apply(self, vec, theta, T_xx_inv, reml=True):
        """Apply the projection P = V^{-1} - V^{-1}X (X'V^{-1}X)^{-1} X'V^{-1}
        for REML, or just V^{-1} for ML. Caller supplies T_xx_inv =
        (X'V^{-1}X)^{-1}, typically from compute_effects()."""
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
        """Dense J = dtheta/dtau. Identity on variance and residual rows;
        on each correlation row o with parents (a, b), J[o, *] has at most
        three nonzeros at columns (a, b, o)."""
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
        J[o, o] = sqrt_ab * sech2  # overwrites the np.eye() 1.0 (correct: 0 if unsafe)
        J[o, a] = 0.5 * tanh_o * ratio_ba
        J[o, b] = 0.5 * tanh_o * ratio_ab
        return J

    def jvp_T(self, tau, g):
        """Compute J.T @ g without materializing J. Used in profile-likelihood
        chain-rule gradients to avoid the n×n dense build per evaluation."""
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


class LMMSummary:
    """Lightweight container for an LMM2 fit summary. Holds three DataFrames
    (`fe`, `re`, `ic`) and pretty-prints via __repr__/_repr_html_."""

    def __init__(self, fe, re, ic, header=None, n_obs=None, formula=None):
        self.fe, self.re, self.ic = fe, re, ic
        self.header = header or 'Linear Mixed Model'
        self.n_obs, self.formula = n_obs, formula

    def __repr__(self):
        sep = '=' * 78
        meta = []
        if self.formula is not None: meta.append(f'Formula: {self.formula}')
        if self.n_obs is not None: meta.append(f'N: {self.n_obs}')
        return '\n'.join([
            sep, self.header, *meta, sep,
            'Fixed effects:', self.fe.to_string(float_format='%.4f'), '',
            'Random-effects parameters:',
            self.re.to_string(float_format='%.4f'), '',
            'Information criteria:',
            self.ic.to_string(float_format='%.2f'), sep,
        ])

    def __str__(self):
        return self.__repr__()

    def _repr_html_(self):
        meta = ''
        if self.formula is not None: meta += f'<p><b>Formula:</b> {self.formula}</p>'
        if self.n_obs is not None: meta += f'<p><b>N:</b> {self.n_obs}</p>'
        return ('<div><h3>{header}</h3>{meta}'
                '<h4>Fixed effects</h4>{fe}'
                '<h4>Random-effects parameters</h4>{re}'
                '<h4>Information criteria</h4>{ic}</div>').format(
                    header=self.header, meta=meta,
                    fe=self.fe.to_html(float_format='%.4f'),
                    re=self.re.to_html(float_format='%.4f'),
                    ic=self.ic.to_html(float_format='%.4f'))


class LMM2(object):

    def __init__(self, formula, data, mme_kws=None):
        # `residual_formula` was a placeholder for heteroscedastic / structured
        # residual covariance (e.g. AR(1)) but never had an implementation
        # behind it; removed until the corresponding ResidualCovTerm subclasses
        # exist. The single DiagResidualCovTerm path is the only one wired up.
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
        self.fe_names = list(X_df.columns) if hasattr(X_df, 'columns') else \
            [f"X{i}" for i in range(X.shape[1])]
        self.re_terms = re_terms

    # ---- objectives ---------------------------------------------------------

    def loglike(self, theta, reml=True):
        return self.mme._loglike(theta, reml)

    def gradient(self, theta, reml=True):
        return self.mme._gradient(theta, reml)

    def gradient_v2(self, theta, reml=True):
        """Per-term Schur-form gradient — alternative implementation that
        skips global L_zz materialization. Mathematically equivalent to
        `gradient`; provided for benchmarking against the legacy path."""
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
        """Update C = Z'R^{-1}Z + G^{-1} at the given theta and refactor
        the in-place Cholesky on it. Used by every method that needs
        chol_fac at a particular theta."""
        self.mme.update_crossprods(theta)
        Ginv = self.mme.re_mod.update_gcov(theta, inv=True, G=self.mme.G)
        C = cs_add_inplace(self.mme.ZtRZ, Ginv, self.mme.C)
        self.mme.chol_fac.cholesky_inplace(sp.sparse.csc_matrix(C))
        return Ginv

    def compute_effects(self, theta=None):
        if theta is None:
            theta = getattr(self, 'theta', self.mme.re_mod.theta)
        self._factor_C(theta)
        ZtRX = self.mme.ZtRXy[:, :-1]
        ZtRy = self.mme.ZtRXy[:, [-1]]
        XtRX = self.mme.XytRXy[:-1, :-1]
        XtRy = self.mme.XytRXy[:-1, [-1]]
        ZtRZ = self.mme.ZtRZ
        #TODO: remove these if else nightmares. Again, upstream failure
        M_X = self.mme.chol_fac.solve_A(sp.sparse.csc_matrix(ZtRX))
        M_y = self.mme.chol_fac.solve_A(sp.sparse.csc_matrix(ZtRy))
        if sp.sparse.issparse(M_X): M_X = np.asarray(M_X.toarray())
        if sp.sparse.issparse(M_y): M_y = np.asarray(M_y.toarray())
        ZtRX_arr = np.asarray(ZtRX.toarray() if sp.sparse.issparse(ZtRX) else ZtRX)
        ZtRy_arr = np.asarray(ZtRy.toarray() if sp.sparse.issparse(ZtRy) else ZtRy)
        XtRX_arr = np.asarray(XtRX.toarray() if sp.sparse.issparse(XtRX) else XtRX)
        XtRy_arr = np.asarray(XtRy.toarray() if sp.sparse.issparse(XtRy) else XtRy)
        T_xx = XtRX_arr - ZtRX_arr.T.dot(M_X)
        T_xx_inv = np.linalg.inv(T_xx)
        u_xy = XtRy_arr - ZtRX_arr.T.dot(M_y)
        u_zy = ZtRy_arr - np.asarray(ZtRZ.dot(M_y))
        T_zx = ZtRX_arr - np.asarray(ZtRZ.dot(M_X))
        v_y = u_zy - T_zx.dot(T_xx_inv.dot(u_xy))
        beta = T_xx_inv.dot(u_xy).reshape(-1)
        G = self.mme.re_mod.update_gcov(theta, inv=False, G=self.mme.G)
        u = np.asarray(G.dot(v_y)).reshape(-1)
        return beta, T_xx_inv, u

    def predict(self, X=None, Z=None, beta=None, u=None):
        X = self.mme.X if X is None else np.asarray(X)
        Z = self.mme.Z if Z is None else Z
        beta = self.beta if beta is None else beta
        u = self.u if u is None else u
        return X.dot(beta) + np.asarray(Z.dot(u)).reshape(-1)

    def vinvcrossprod(self, A, B, theta=None):
        theta = self.theta if hasattr(self, 'theta') and theta is None else theta
        self._factor_C(theta)
        self.mme._update_rcov(theta, inv=True)
        Rinv = self.mme.R
        A2 = A.toarray() if sp.sparse.issparse(A) else np.asarray(A)
        B2 = B.toarray() if sp.sparse.issparse(B) else np.asarray(B)
        ZtRA = self.mme.Zt.dot(Rinv.dot(A2))
        ZtRB = self.mme.Zt.dot(Rinv.dot(B2))
        M_B = self.mme.chol_fac.solve_A(sp.sparse.csc_matrix(ZtRB))
        if sp.sparse.issparse(M_B): M_B = np.asarray(M_B.toarray())
        AtRB = A2.T.dot(Rinv.dot(B2))
        ZtRA_arr = np.asarray(ZtRA.toarray() if sp.sparse.issparse(ZtRA) else ZtRA)
        return np.asarray(AtRB) - ZtRA_arr.T.dot(M_B)

    # ---- fit pipeline -------------------------------------------------------

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
        from ..utilities.numerical_derivs import so_gc_cd
        self.theta = theta_hat
        self.eta = opt.x
        self.opt = opt
        self.reml = reml
        self.beta, self.XtVinvX_inv, self.u = self.compute_effects(theta_hat)
        self.se_beta = np.sqrt(np.diag(self.XtVinvX_inv))

        try:
            H = self.average_information(theta_hat, reml=reml)
        except NotImplementedError:
            H = so_gc_cd(self.gradient, theta_hat, args=(reml,))
        self.H_theta = H
        self.Hinv_theta = np.linalg.pinv(H / 2.0)
        self.se_theta = np.sqrt(np.clip(np.diag(self.Hinv_theta), 0, None))

        n_obs = self.mme.n_obs
        n_fe = self.mme.n_fixef
        n_pars = len(theta_hat)
        nll = float(opt.fun)  # -2 logL_(R)EML evaluated at θ̂
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
        self.AICC = (self.ll + 2 * d * n_eff / (n_eff - d - 1)
                     if n_eff - d - 1 > 0 else np.nan)
        self.BIC = self.ll + d * np.log(n_eff)
        self.CAIC = self.ll + d * (np.log(n_eff) + 1)

        re_mod = self.mme.re_mod
        self.re_covs, self.re_corrs = {}, {}
        for k in range(re_mod.n_gterms):
            theta_k = theta_hat[re_mod.theta_sl[k]]
            G_k = invech(theta_k)
            self.re_covs[k] = G_k
            std = np.sqrt(np.clip(np.diag(G_k), 1e-30, None))
            self.re_corrs[k] = G_k / np.outer(std, std)

        self._build_summary_frames()

    def _build_summary_frames(self):
        from ..utilities.output import get_param_table
        re_mod = self.mme.re_mod
        df_t = self.mme.n_obs - self.mme.n_fixef
        self.summary_fe = get_param_table(self.beta, self.se_beta, degfree=df_t,
                                          index=self.fe_names,
                                          parameter_label='estimate')
        labels, ests, ses = [], [], []
        for k in range(re_mod.n_gterms):
            term = re_mod.gterms[k]
            ucov = term.cov_structure.unstructured_cov
            sl = re_mod.theta_sl[k]
            gname = getattr(term, 're_grouping', f'g{k}')
            vnames = getattr(term, 'var_names',
                             [f'v{i}' for i in range(term.n_revars)])
            for j in range(ucov.n_pars):
                a, b = int(ucov.r_inds[j]), int(ucov.c_inds[j])
                labels.append(f"{gname}.var({vnames[a]})" if a == b
                              else f"{gname}.cov({vnames[a]},{vnames[b]})")
                ests.append(self.theta[sl.start + j])
                ses.append(self.se_theta[sl.start + j])
        labels.append('resid.var')
        ests.append(self.theta[-1]); ses.append(self.se_theta[-1])
        re_full = get_param_table(np.asarray(ests), np.asarray(ses),
                                  degfree=np.inf, index=labels,
                                  parameter_label='estimate')
        # Variance components are bounded ≥ 0 — z/p columns are misleading.
        self.summary_re = re_full.drop(columns=['t', 'p'])
        self.ic = pd.DataFrame(
            {'value': [self.ll, self.llf, self.AIC, self.AICC, self.BIC, self.CAIC]},
            index=['ll', 'llf', 'AIC', 'AICC', 'BIC', 'CAIC'])

    def summary(self):
        """Return an `LMMSummary` object holding fixed-effect, random-effect,
        and information-criterion tables. Pretty-prints in the terminal and
        renders as HTML in notebooks."""
        return LMMSummary(self.summary_fe, self.summary_re, self.ic,
                          header=f'Linear Mixed Model ({"REML" if self.reml else "ML"})',
                          n_obs=self.mme.n_obs, formula=self.formula)

    def confint(self, alpha=0.05):
        """Two-sided (1 - alpha) Wald CIs for fixed effects (t-distributed
        on n_obs - n_fixef df)."""
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
            se = None
            try:
                se = self.random_effects_se()
            except Exception:
                pass
            rows = []
            for k in range(re_mod.n_gterms):
                term = re_mod.gterms[k]
                sl = re_mod.ranef_sl[k]
                nv, ng = term.n_revars, term.n_levels
                u_blk = self.u[sl].reshape(ng, nv)
                se_blk = (se[sl].reshape(ng, nv) if se is not None
                          else np.full((ng, nv), np.nan))
                for j in range(ng):
                    for a in range(nv):
                        rows.append((k, j, f'v{a}', u_blk[j, a], se_blk[j, a]))
            return pd.DataFrame(rows, columns=['term', 'level', 'var',
                                                'estimate', 'SE']).set_index(
                ['term', 'level', 'var'])
        raise ValueError(f"unknown as_frame={as_frame!r}")

    def random_effects_array(self):
        """û as a flat 1-D numpy array (length sum_k ng_k * nv_k)."""
        return np.asarray(self.u)

    def random_effects_se(self, theta=None, full=True):
        theta = self.theta if theta is None else theta
        self._factor_C(theta)
        # M_X = C^{-1} ZtRX
        ZtRX = self.mme.ZtRXy[:, :-1]
        #TODO: this shit agian
        ZtRX_arr = np.asarray(ZtRX.toarray() if sp.sparse.issparse(ZtRX)
                              else ZtRX)
        M_X = self.mme.chol_fac.solve_A(sp.sparse.csc_matrix(ZtRX))
        if sp.sparse.issparse(M_X): M_X = np.asarray(M_X.toarray())
        else: M_X = np.asarray(M_X)
        T_xx_inv = self.XtVinvX_inv
        # diag(M_X T_xx_inv M_X^T) = einsum('ik,kl,il->i', M_X, T_xx_inv, M_X)
        var_x = np.einsum('ik,kl,il->i', M_X, T_xx_inv, M_X, optimize=True)

        if not full:
            return np.sqrt(np.clip(var_x, 0.0, None))

        # diag(C^{-1}) via n_ranef single-column solves. For large problems
        # this is the slow part; sksparse 0.5+ doesn't expose Takahashi-style
        # selected inverse so we pay one solve per column.
        n_ranef = self.mme.Z.shape[1]
        diag_Cinv = np.zeros(n_ranef)
        e = np.zeros(n_ranef)
        for i in range(n_ranef):
            e[i] = 1.0
            x = self.mme.chol_fac.solve_A(sp.sparse.csc_matrix(e.reshape(-1, 1)))
            xi = (np.asarray(x.toarray()).reshape(-1)[i]
                  if sp.sparse.issparse(x)
                  else np.asarray(x).reshape(-1)[i])
            diag_Cinv[i] = float(xi)
            e[i] = 0.0
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

    # ---- second-order: Hessian and df approximations -----------------------

    def hessian(self, theta=None, reml=True):
        raise NotImplementedError("Nah dawg")

    def average_information(self, theta=None, reml=True):
        if not self.mme._scalar_resid:
            raise NotImplementedError("nah fam")
        theta = (np.asarray(self.theta if hasattr(self, 'theta')
                            else self.mme.re_mod.theta)
                 if theta is None else np.asarray(theta))

        re_mod = self.mme.re_mod
        n_obs = self.mme.n_obs
        n_pars = int(len(theta))
        n_ranef = re_mod.Z.shape[1]
        X, Z = self.mme.X, self.mme.Z
        y = self.mme.y.reshape(-1)

        beta, T_xx_inv, u = self.compute_effects(theta)  # refreshes chol_fac
        r = y - X.dot(beta) - np.asarray(Z.dot(u)).reshape(-1)
        Py = r / float(theta[-1])  # scalar resid: Rinv @ r = r / sigma2
        v_y = np.asarray(Z.T.dot(Py)).reshape(-1)

        # u_mat[:, i] = A_i @ Py. For G-params, A_i = Z dG_i Z'; for resid, A = I.
        u_mat = np.zeros((n_obs, n_pars))
        for k in range(re_mod.n_gterms):
            term = re_mod.gterms[k]
            ucov = term.cov_structure.unstructured_cov
            sl = re_mod.ranef_sl[k]
            par_sl = re_mod.theta_sl[k]
            ng, nv = term.n_levels, term.n_revars
            v_y_blk = v_y[sl].reshape(ng, nv)
            for jp in range(ucov.n_pars):
                a, b = int(ucov.r_inds[jp]), int(ucov.c_inds[jp])
                blk = np.zeros((ng, nv))
                if a == b:
                    blk[:, a] = v_y_blk[:, a]
                else:
                    blk[:, a] = v_y_blk[:, b]
                    blk[:, b] = v_y_blk[:, a]
                full = np.zeros(n_ranef)
                full[sl] = blk.reshape(-1)
                u_mat[:, par_sl.start + jp] = np.asarray(Z.dot(full)).reshape(-1)
        u_mat[:, -1] = Py

        Pu_mat = np.empty_like(u_mat)
        for j in range(n_pars):
            Pu_mat[:, j] = self.mme.p_apply(u_mat[:, j], theta, T_xx_inv,
                                             reml=reml)

        AI = 0.5 * u_mat.T.dot(Pu_mat)
        AI = 0.5 * (AI + AI.T)
        return 2.0 * AI

    def _fd_dC_dtheta(self, theta, eps=None):
        """Central-difference list [dC/dθ_i] where C = (X'V^{-1}X)^{-1}.
        Each evaluation re-factorizes chol_fac at theta ± eps, so cost scales
        linearly with len(theta). Used by approx_degfree."""
        eps = (np.finfo(float).eps) ** (1.0 / 3.0) if eps is None else eps
        n_pars = theta.size
        X = self.mme.X
        out = []
        for i in range(n_pars):
            tp = theta.copy(); tp[i] += eps
            tm = theta.copy(); tm[i] -= eps
            Cp = np.linalg.inv(np.asarray(self.vinvcrossprod(X, X, tp)))
            Cm = np.linalg.inv(np.asarray(self.vinvcrossprod(X, X, tm)))
            out.append((Cp - Cm) / (2.0 * eps))
        # Restore chol_fac at the original theta so subsequent calls aren't
        # left holding a perturbed factorisation.
        self._factor_C(theta)
        return out

    def approx_degfree(self, L_list=None, theta=None, beta=None,
                       method='satterthwaite', reml=True):
        # Only Satterthwaite is implemented; Kenward-Roger would need the
        # second-derivative correction term (W_2 in Kenward & Roger 1997)
        # which we don't have a closed form for here.
        if method == 'kenward-roger':
            raise NotImplementedError("Kenward-Roger not implemented; "
                                      "use method='satterthwaite'.")
        if method != 'satterthwaite':
            raise ValueError(f"unknown method={method!r}; "
                             f"expected 'satterthwaite' or 'kenward-roger'.")

        n_fe = self.mme.n_fixef
        if L_list is None:
            L_list = [np.eye(n_fe)[[i]] for i in range(n_fe)]
        theta = self.theta if theta is None else theta
        beta = self.beta if beta is None else beta

        self._factor_C(theta)
        C = np.asarray(np.linalg.inv(np.asarray(self.vinvcrossprod(self.mme.X,
                                                                    self.mme.X,
                                                                    theta))))

        # Vtheta = (½ H)^{-1}. Reuse cached Hinv_theta if we've fit; else
        # use the exact AI matrix (NotImplementedError fallback to numerical H).
        if hasattr(self, 'Hinv_theta'):
            Vtheta = self.Hinv_theta
        else:
            try:
                H = self.average_information(theta=theta, reml=reml)
            except NotImplementedError:
                from ..utilities.numerical_derivs import so_gc_cd
                H = so_gc_cd(self.gradient, theta, args=(reml,))
            Vtheta = np.linalg.pinv(H / 2.0)

        # J_i = dC/dθ_i via central differences.
        J = self._fd_dC_dtheta(theta)

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
            # Per-direction Satterthwaite df.
            D = np.array([[float(x.dot(Ji).dot(x)) for Ji in J]
                          for x in P[:q]])
            nu_d = np.array([D[i].dot(Vtheta).dot(D[i]) for i in range(q)])
            nu_m = u[:q] ** 2 / np.maximum(nu_d, np.finfo(float).tiny)
            keep = nu_m > 2
            E = float(np.sum(nu_m[keep] / (nu_m[keep] - 2.0)))
            nu = float(2.0 * E / (E - q)) if E > q else float('nan')
            p = (float(sp.stats.f(q, nu).sf(f)) if np.isfinite(nu)
                 else float('nan'))
            out.append({'F': f, 'df1': q, 'df2': nu, 'p': p})
        return out

    # ---- profile likelihood -----------------------------------------------

    def profile_likelihood(self, n_points=11, n_se=3.0, parameters=None,
                           method='l-bfgs-b', verbose=False):

        if not hasattr(self, 'theta'):
            raise RuntimeError("Call .fit() before .profile_likelihood().")
        reparam = _VarCorrReparam(self.mme.re_mod)
        tau_hat = reparam.fwd(self.theta)

        # SE(tau) via chain rule from Hinv_theta. Var(tau) = J^{-1} V_theta J^{-T}
        # where J = dtheta/dtau.
        J = reparam.jac_rvs(tau_hat)
        try:
            J_inv = np.linalg.inv(J)
            var_tau = J_inv.dot(self.Hinv_theta).dot(J_inv.T)
        except np.linalg.LinAlgError:
            var_tau = self.Hinv_theta
        se_tau = np.sqrt(np.clip(np.diag(var_tau), 0, None))

        n = len(tau_hat)
        if parameters is None:
            parameters = list(range(n))
        ll_full = self.nll  # -2 log L_(R)EML at the fitted theta_hat
        var_set = set(reparam.diag_ix.tolist()) | {n - 1}

        rows = []
        for i in parameters:
            center = float(tau_hat[i])
            width = n_se * max(float(se_tau[i]), 1e-3)
            lo = max(center - width, 1e-6) if i in var_set else center - width
            hi = center + width
            grid = np.linspace(lo, hi, n_points)
            for t0 in grid:
                if verbose:
                    print(f"  profile param {i}, tau={t0:.4f}", flush=True)
                theta_r, ll_r = self._fit_with_fixed_tau(
                    reparam, tau_hat, i, t0, method=method)
                LR = ll_r - ll_full
                LR = max(LR, 0.0)
                zeta = float(np.sign(t0 - center) * np.sqrt(LR))
                rows.append({'param': i, 'tau': float(t0),
                             'theta_i': float(theta_r[i]),
                             'LR': float(LR), 'zeta': zeta,
                             'theta': theta_r.copy()})
        return pd.DataFrame(rows)

    def _fit_with_fixed_tau(self, reparam, tau_init, fixed_idx, fixed_value,
                            method='l-bfgs-b'):
        """One profile point: optimize -2 log L over tau with tau[fixed_idx]
        held at `fixed_value`. Uses the chain-rule gradient through reparam."""
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
            # dL/dtau = J.T @ dL/dtheta  (J = dtheta/dtau), without building J.
            g_tau = reparam.jvp_T(tau, g_theta)
            return ll, g_tau[free]

        x0 = tau_init[free].copy()
        opt = sp.optimize.minimize(fg, x0, jac=True, method=method,
                                   bounds=bounds, options={'maxiter': 200})
        tau_r = tau_init.copy()
        tau_r[free] = opt.x
        tau_r[fixed_idx] = fixed_value
        return reparam.rvs(tau_r), float(opt.fun)

    def plot_profile(self, profile_df=None, parameters=None, quantiles=None,
                     figsize=None, axes=None, sharey=True,
                     line_kws=None, scatter_kws=None, cmap='bwr',
                     show_wald=True, show_ci_bands=True,
                     param_names=None, n_points=21, n_se=3.0,
                     zeta_lim=5.0, colorbar=False):
        """Plot profile-likelihood ζ(θᵢ) curves with confidence bands and
        Wald-CI overlay scatter (one panel per parameter).

        Parameters
        ----------
        profile_df : DataFrame from profile_likelihood(), or None to compute.
        parameters : iterable of int or None
            Subset of parameters to plot. Defaults to all in profile_df.
        quantiles : array-like in (0, 100) or None
            Confidence levels for the colored bands. Default
            {60, 70, 80, 90, 95, 99} (two-sided, mirrored).
        figsize : tuple or None
            Figure size; default scales with the number of panels.
        axes : array of Axes or None
            Pre-existing axes (e.g. from a parent layout). If None, a fresh
            figure is created.
        sharey : bool
            Share y-axis across panels.
        line_kws, scatter_kws : dict
            Extra kwargs for the profile line and the Wald scatter.
        cmap : str
            Colormap for the CI bands and Wald points.
        show_wald : bool
            Overlay Wald-CI scatter at θ̂ + q·SE(θ̂) on the zero line.
        show_ci_bands : bool
            Draw the colored confidence-level segments at the band quantiles.
        param_names : list[str] or None
            Custom axis titles, indexed by full parameter index. If None,
            uses 'θ[i]' or, when available, the entries of summary_re.
        n_points, n_se : ints / floats
            Forwarded to profile_likelihood() if profile_df is None.
        zeta_lim : float
            Truncate the y-axis to ±zeta_lim and drop points outside that.
        colorbar : bool
            Add a colorbar showing the band quantiles.

        Returns (fig, axes).
        """
        #What the fuck
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        from scipy.interpolate import interp1d

        if profile_df is None:
            profile_df = self.profile_likelihood(
                n_points=n_points, n_se=n_se, parameters=parameters)

        if parameters is None:
            parameters = sorted(int(p) for p in profile_df['param'].unique())

        if quantiles is None:
            q_levels = np.array([60.0, 70.0, 80.0, 90.0, 95.0, 99.0])
            quantiles = np.concatenate([(100.0 - q_levels[::-1]) / 2.0,
                                         100.0 - (100.0 - q_levels) / 2.0])
        quantiles = np.asarray(quantiles, dtype=float)
        q = sp.stats.norm(0, 1).ppf(quantiles / 100.0)

        n_panels = len(parameters)
        if axes is None:
            if figsize is None:
                figsize = (max(3.5 * n_panels, 5.0), 3.6)
            fig, axes = plt.subplots(ncols=n_panels, figsize=figsize,
                                     sharey=sharey)
            if n_panels == 1:
                axes = np.array([axes])
            plt.subplots_adjust(wspace=0.08, left=0.08, right=0.95,
                                bottom=0.15)
        else:
            axes = np.atleast_1d(axes)
            fig = axes.flat[0].figure

        line_defaults = {'color': 'C0', 'lw': 1.8}
        line_defaults.update(line_kws or {})
        scatter_defaults = {'s': 28, 'edgecolor': 'k', 'linewidth': 0.4,
                            'zorder': 5}
        scatter_defaults.update(scatter_kws or {})

        norm = mpl.colors.TwoSlopeNorm(vcenter=0, vmin=q.min(), vmax=q.max())
        cmap_obj = plt.get_cmap(cmap)

        # Default axis labels: prefer summary_re if available.
        if param_names is None and hasattr(self, 'summary_re'):
            try:
                names_full = list(self.summary_re.index)
                param_names = {i: names_full[i] for i in range(len(names_full))}
            except Exception:
                param_names = None

        for ax, p in zip(axes.flat, parameters):
            sub = (profile_df[profile_df['param'] == p]
                   .sort_values('tau').reset_index(drop=True))
            x = np.asarray(sub['theta_i'].values, dtype=float)
            y = np.asarray(sub['zeta'].values, dtype=float)
            keep = (y > -zeta_lim) & (y < zeta_lim) & np.isfinite(y) & np.isfinite(x)
            x, y = x[keep], y[keep]

            ax.plot(x, y, **line_defaults)
            ax.axhline(0, color='k', lw=0.7)
            if hasattr(self, 'theta'):
                ax.axvline(float(self.theta[p]), color='k', lw=0.7)

            label = (param_names.get(p, f'θ[{p}]')
                     if isinstance(param_names, dict) else
                     (param_names[p] if param_names else f'θ[{p}]'))
            ax.set_title(label, fontsize=11)
            ax.set_xlabel(label)
            if p == parameters[0]:
                ax.set_ylabel(r'$\zeta(\theta)$')

            if x.size < 3:
                continue

            if show_ci_bands:
                # Need ζ to be increasing in θ_i for inversion. Sort by ζ
                # and clip to monotone range.
                order = np.argsort(y)
                ys, xs = y[order], x[order]
                _, uniq = np.unique(ys, return_index=True)
                ys, xs = ys[uniq], xs[uniq]
                try:
                    rtef_inv = interp1d(ys, xs, kind='linear',
                                     bounds_error=False, fill_value=np.nan)
                    xq = f_inv(q)
                    valid = np.isfinite(xq)
                    if valid.any():
                        sgs = np.zeros((int(valid.sum()), 2, 2))
                        sgs[:, 0, 0] = sgs[:, 1, 0] = xq[valid]
                        sgs[:, 0, 1] = 0.0
                        sgs[:, 1, 1] = q[valid]
                        lc = mpl.collections.LineCollection(sgs, cmap=cmap_obj,
                                                            norm=norm)
                        lc.set_array(q[valid])
                        lc.set_linewidth(2)
                        ax.add_collection(lc)
                except Exception:
                    pass

            if show_wald and hasattr(self, 'theta') and hasattr(self, 'se_theta'):
                xqw = float(self.theta[p]) + q * float(self.se_theta[p])
                ax.scatter(xqw, np.zeros_like(xqw), c=q, cmap=cmap_obj,
                           norm=norm, **scatter_defaults)

            ax.set_ylim(-zeta_lim, zeta_lim)

        if colorbar:
            sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=axes.tolist() if hasattr(axes, 'tolist')
                                else list(axes.flat),
                                fraction=0.025, pad=0.02)
            cbar.set_label('z-quantile')

        return fig, axes

    def profile_confint(self, alpha=0.05, profile_df=None, n_points=11,
                        n_se=3.0, parameters=None):
        """Profile-likelihood CIs at level (1 - alpha). Inverts the profile
        zeta to find the parameter values where zeta = ±z_{1 - alpha/2}.
        Returns DataFrame indexed by parameter with columns
            ['theta_lo', 'theta_hi', 'tau_lo', 'tau_hi'].
        Reports both the natural-scale (theta) and the tau-scale CI bounds.
        Pass an existing `profile_df` to avoid recomputing."""
        if profile_df is None:
            profile_df = self.profile_likelihood(
                n_points=n_points, n_se=n_se, parameters=parameters)
        z = float(sp.stats.norm.ppf(1.0 - alpha / 2.0))
        from scipy.interpolate import interp1d
        out_rows = {}
        for i, sub in profile_df.groupby('param'):
            sub = sub.sort_values('tau').reset_index(drop=True)
            # zeta is monotone in tau (locally) — interpolate inverse.
            tau_at_zeta = interp1d(sub['zeta'].values, sub['tau'].values,
                                   kind='linear', bounds_error=False,
                                   fill_value='extrapolate')
            theta_at_tau = interp1d(sub['tau'].values, sub['theta_i'].values,
                                    kind='linear', bounds_error=False,
                                    fill_value='extrapolate')
            tau_lo = float(tau_at_zeta(-z))
            tau_hi = float(tau_at_zeta(z))
            out_rows[int(i)] = {'tau_lo': tau_lo, 'tau_hi': tau_hi,
                                'theta_lo': float(theta_at_tau(tau_lo)),
                                'theta_hi': float(theta_at_tau(tau_hi))}
        return pd.DataFrame(out_rows).T

    def lrt(self, other):
        """Likelihood-ratio test: 2(ll_other - ll_self) ~ chi2(df) where
        df = #pars in self minus #pars in other (positive when self is the
        richer model). Both models must have been fitted with the same reml."""
        if self.reml != getattr(other, 'reml', None):
            raise ValueError("LRT requires both models fit with same reml")
        df = max(0, self.theta.size - other.theta.size)
        stat = -(self.ll - other.ll)
        p = sp.stats.chi2(df).sf(stat) if df > 0 else 1.0
        return pd.DataFrame({'stat': [stat], 'df': [df], 'p': [p]})

