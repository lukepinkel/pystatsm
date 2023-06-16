#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 07:59:44 2023

@author: lukepinkel
"""

import numpy as np
import scipy as sp
import pandas as pd

from ..utilities.linalg_operations import  _vech, _vec, _invec, _invech
from ..utilities.func_utils import handle_default_kws, triangular_number
from .derivatives import _dloglike_mu, _d2loglike_mu0, _d2loglike_mu1, _d2loglike_mu2 #from .cov_derivatives import _d2sigma, _dsigma, _dloglike, _d2loglike
from .param_table import ParameterTable
from .model_data import ModelData

def _sparse_post_mult(A, S):
    prod = S.T.dot(A.T)
    prod = prod.T
    return prod




pd.set_option("mode.chained_assignment", None)




class ModelSpecification(object):
    
    def __init__(self, formula, data, group_col):
        group_ids = np.unique(data[group_col])
        n_groups = len(group_ids)
        data[group_col] = data[group_col].replace(dict(zip(group_ids, np.arange(n_groups)))) 
        self.n_groups = n_groups
        self.create_group_tables(formula, data, group_col)
        
    def create_group_tables(self, formula, data, group_col):
        self.ptables, self.ftables , self.ptable_objects= {}, {}, {}
        self.p_templates, self.indexers = {}, {}
        self.sample_covs = {}
        self.sample_means = {}
        self.mat_cols, self.mat_rows, self.mat_dims = {}, {}, {}
        for i in range(self.n_groups):
            dfi = data.loc[data[group_col]==i].iloc[:, :-1]
            sample_cov = dfi.cov(ddof=0)
            sample_mean = pd.DataFrame(dfi.mean()).T
            ptable_object = ParameterTable(formula, sample_cov, sample_mean)
            pt, ix, mr, mc, md = ptable_object.construct_model_mats(ptable_object.ptable,
                                                        ptable_object.var_names, 
                                                        ptable_object.lav_order, 
                                                        ptable_object.obs_order)
            self.mat_rows[i] = mr
            self.mat_cols[i] = mc
            self.mat_dims[i] = md
            self.p_templates[i], self.indexers[i] = pt, ix
            self. ptable_objects[i] = ptable_object
            self.ptables[i] = ptable_object.ptable
            self.ftables[i] = ptable_object.ftable
            self.sample_covs[i] = sample_cov.values
            self.sample_means[i] = sample_mean.values
        self.ftable = pd.concat(self.ftables, axis=0).reset_index()
        self.frees = {}
        self.q = len((self.ptable_objects[0]).lav_order)
        self.p = len((self.ptable_objects[0]).obs_order)

        for i in range(self.n_groups):
            self.frees[i] = self.p_templates[i][self.indexers[i].flat_indices]


    
    def harmonize_group_tables(self, equal_loadings=True, equal_regression=True, 
                               equal_means=False):
        self.n_free = self.ptable_objects[0].n_free
        self.shared = np.zeros(len(self.n_free), dtype=bool)
        self.n_shared = 0
        if equal_loadings:
            self.shared[0] = True
        if equal_regression:
            self.shared[1] = True
        if equal_means:
            self.shared[4] = True
            self.shared[5] = True
        self.unshared = ~self.shared
        self.n_shared =self. n_free[self.shared].sum()
        self.n_unshared = self.n_free[self.unshared].sum()
        self.n_total_free = self.n_shared + self.n_unshared*self.n_groups
        self.n_free = self.n_shared + self.n_unshared
        
        ftable = pd.concat(self.ftables, axis=0).reset_index()
        ftable["group_ind"] = 0
        ftable["group_free"] = 0
        ftable["duplicate"] = False

        ix0 = ftable["level_0"] == 0
        offset = 0
        for i in range(self.n_groups):
            group_ix = ftable["level_0"] == i
            ftable.loc[group_ix,  "group_ind"] = 1+np.arange(self.n_free) + offset
            ftable.loc[group_ix,  "group_free"] = ftable.loc[group_ix,  "free"] + offset
            if equal_loadings:
                ix = (ftable["mat"] == 0)
                ftable.loc[ix&(group_ix), "group_ind"] = ftable.loc[ix0&ix, "group_ind"].values
                ftable.loc[ix&(group_ix), "group_free"] = ftable.loc[ix0&ix, "group_free"].values
                if i>0:
                    ftable.loc[ix & group_ix, "duplicate"] = True
            if equal_regression:
                ix = ftable["mat"] == 1
                ftable.loc[ix & group_ix, "group_ind"] = ftable.loc[ix & ix0, "group_ind"].values
                ftable.loc[ix & group_ix, "group_free"] = ftable.loc[ix & ix0, "group_free"].values
                if i>0:
                    ftable.loc[ix & group_ix, "duplicate"] = True
            if equal_means:
                ix = ftable["mat"] == 4
                ftable.loc[ix & group_ix, "group_ind"] = ftable.loc[ix & ix0, "group_ind"].values
                ftable.loc[ix & group_ix, "group_free"] = ftable.loc[ix & ix0, "group_ind"].values
                ix = ftable["mat"] == 5
                ftable.loc[ix & group_ix, "group_ind"] = ftable.loc[ix & ix0, "group_ind"].values
                ftable.loc[ix & group_ix, "group_free"] = ftable.loc[ix & ix0, "group_free"].values
                if i>0:
                    ftable.loc[ix & group_ix, "duplicate"] = True

            offset += self.n_unshared
        
        utables = {}
        for i in range(self.n_groups):
            group_ix = ftable["level_0"] == i
            utable = ftable.loc[group_ix]
            utable = utable.iloc[self.indexers[i].unique_indices]
            utables[i] = utable
        utable = pd.concat(utables, axis=0).drop(["level_0", "level_1"], axis=1).reset_index()
        utable = utable[~utable["group_free"].duplicated()]
        free_to_group_free = {}
        nrows = self.n_free
        ncols = self.n_total_free
        d = np.ones(self.n_free)
        for i in range(self.n_groups):
            row = ftable.loc[ftable["level_0"]==i, "ind"].values
            col = (ftable.loc[ftable["level_0"]==i, "group_ind"]-1).values
            free_to_group_free[i] = sp.sparse.csc_array((d, (row, col)), shape=(nrows, ncols))
            
        rows = utable["group_free"].values-1
        cols = utable["group_ind"].values-1
        nrows = rows.max()+1
        ncols = cols.max()+1
        d = np.ones(len(rows))
        group_free_totheta = sp.sparse.csc_array((d, (rows, cols)), shape=(nrows, ncols))
        
        rows = ftable["group_free"].values-1
        cols = ftable["group_ind"].values-1
        ix = np.sort(np.unique(np.vstack([rows, cols]).T,axis=0, return_index=True)[1])
        rows, cols = rows[ix], cols[ix]
        nrows = rows.max()+1
        ncols = cols.max()+1
        d = np.ones(len(rows))
        theta_to_group_free = sp.sparse.csc_array((d, (rows, cols)), shape=(nrows, ncols))
        #theta_to_group_free = group_free_totheta.T
        self.group_free_totheta = group_free_totheta
        self.free_to_group_free = free_to_group_free
        self.theta_to_group_free = theta_to_group_free
        self.ftable = ftable
        self.free = np.zeros(self.n_total_free)
        self.utable = utable
        indexer = self.indexers[0]
        qp = len(indexer.unique_indices)
        J = sp.sparse.csc_array((np.ones(qp), (np.arange(qp), indexer.unique_indices)),
                                shape=(qp, indexer.unique_indices.max()+1))
        self.free_to_theta = J
        for i in range(self.n_groups):
            self.free = self.free + self.free_to_group_free[i].T.dot(self.frees[i])


class SEM:
    
    def __init__(self, formula, data, group_col=None, model_spec_kws=None):
        default_model_spec_kws = dict(extension_kws=dict(fix_lv_var=False))
        model_spec_kws = handle_default_kws(model_spec_kws, default_model_spec_kws)
        if group_col is None:
            group_col = "groups"
            data["groups"] = 0
        self.mspec = ModelSpecification(formula, data, group_col=group_col)
        self.mspec.harmonize_group_tables()
        self.indexer = self.mspec.indexers[0]

        self.model_data = ModelData(data=data)
        bounds = self.mspec.ftable[~self.mspec.ftable["group_ind"].duplicated()][["lb", "ub"]]
        bounds = bounds.values
        self.bounds = [tuple(x) for x in bounds.tolist()]
        self.bounds_theta = [tuple(x) for x in bounds[self.mspec.group_free_totheta.tocoo().col.astype(int)]]
        self.p_templates = self.mspec.p_templates

    
        self.free = self.mspec.free
        self.p, self.q = self.mspec.p, self.mspec.q
        self.p2, self.q2 = triangular_number(self.p), triangular_number(self.q)
        self.nf = len(self.indexer.flat_indices)
        self.nf2 = triangular_number(self.nf)
        self.nt = len(self.indexer.first_locs)
        self.make_derivative_matrices()
        self.theta = self.mspec.group_free_totheta.dot(self.free)
        self.n_par = len(self.p_templates[0])
        self.ll_const =  1.8378770664093453 * self.p
        self.n_groups =self. mspec.n_groups
        self.indexers = self.mspec.indexers
        self.gsizes = self.model_data.data_df["group"].value_counts().values
        self.gweights = self.gsizes / np.sum(self.gsizes)

    def make_derivative_matrices(self):
        self.indexer.create_derivative_arrays([(0, 0), (1, 0), (2, 0), (1 ,1), (2, 1), (5, 0), (5, 1)])
        self.dA = self.indexer.dA
        self.dSm = np.zeros((self.p2+self.p, self.nf))
        self.d2Sm = np.zeros((self.p2+self.p, self.nf, self.nf))
        self.m_size = self.indexer.block_sizes                    
        self.m_kind = self.indexer.block_indices                 
        self.d2_kind = self.indexer.block_pair_types             
        self.d2_inds = self.indexer.colex_descending_inds  
        self.J_theta = sp.sparse.csc_array(
            (np.ones(self.nf), (np.arange(self.nf), self.indexer.unique_locs)), 
             shape=(self.nf, self.nt))
        self.dSm = np.zeros((self.p2+self.p, self.nf))
        self.d2Sm = np.zeros((self.p2+self.p, self.nf, self.nf))
        s, r = np.triu_indices(self.p, k=0)
        self._vech_inds = r+s*self.p
        self.unique_locs = self.indexer.unique_locs
        self.ptable = self.mspec.ptables[0]
        self.free_table = self.mspec.ftables[0]
        self.free_names = self.free_table[["lhs", "rel","rhs"]].astype(str).agg(' '.join, axis=1).values
        self.theta_names = self.mspec.utable[["lhs", "rel","rhs"]].astype(str).agg(' '.join, axis=1).values
        self._grad_kws = dict(dA=self.dA, m_size=self.m_size, m_type=self.m_kind, 
                              vind=self._vech_inds, n=self.nf, p2=self.p2)
        self._hess_kws = dict(dA=self.dA, m_size=self.m_size, d2_inds=self.d2_inds,
                              first_deriv_type=self.m_kind,  second_deriv_type=self.d2_kind,
                              n=self.nf,  vech_inds=self._vech_inds)

     
    
    def func(self, theta, per_group=False):
        free = self._theta_to_free(theta)
        if per_group:
            f = np.zeros(self.n_groups)
            if np.iscomplexobj(theta):
                f = f.astype(complex)
        else:
            f = 0.0
        for i in range(self.n_groups):
            group_free = self._free_to_group_free(free, i)
            par = self._free_to_par(group_free, i)
            mats = self._par_to_model_mats(par, i)
            Sigma, mu = self._implied_cov_mean(*mats)
            r = (self.mspec.sample_means[i]-mu).flatten()
            V = np.linalg.inv(Sigma)
            trSV = np.trace(V.dot(self.mspec.sample_covs[i]))
            rVr = np.dot(r.T.dot(V), r) 
            if np.any(np.iscomplex(Sigma)):
                s, lnd = np.linalg.slogdet(Sigma)
                lndS = np.log(s)+lnd
            else:
                s, lndS = np.linalg.slogdet(Sigma)
            fi = (rVr + lndS + trSV) * self.gweights[i]
            if (s==-1) or (fi < -1):
                fi += np.inf
            if per_group:
                f[i] = fi
            else:
                f += fi
        return f
    
    def gradient(self, theta, per_group=False):
        free = self._theta_to_free(theta)
        if per_group:
            g = np.zeros((self.n_groups, self.mspec.n_total_free))
        else:
            g = np.zeros(self.mspec.n_total_free)
        for i in range(self.n_groups):
            group_free = self._free_to_group_free(free, i)
            par = self._free_to_par(group_free, i)
            L, B, F, P, a, b = self._par_to_model_mats(par, i)
            B = np.linalg.inv(np.eye(B.shape[0]) - B)
            LB = np.dot(L, B)
            mu = (a+LB.dot(b.T).T).reshape(-1)
            a, b = a.flatten(), b.flatten()
            Sigma = LB.dot(F).dot(LB.T) + P
            R = self.mspec.sample_covs[i] - Sigma
            Sinv = np.linalg.inv(Sigma)
            VRV = Sinv.dot(R).dot(Sinv)
            vecVRV = _vec(VRV)
            rtV = (self.mspec.sample_means[i].flatten() - mu).dot(Sinv)
            gi = np.zeros(self.nf)
            kws = self._grad_kws
            gi = _dloglike_mu(gi, L, B, F, b, a,vecVRV, rtV, **kws) * self.gweights[i]
            if per_group:
                g[i] = self.mspec.free_to_group_free[i].T.dot(gi)
            else:
                g = g + self.mspec.free_to_group_free[i].T.dot(gi)
        g = (self.mspec.theta_to_group_free.dot(g.T)).T
        return g
            

    def hessian(self, theta, per_group=False, method=0):
        free = self._theta_to_free(theta)
        if per_group:
            H = np.zeros((self.n_groups, self.mspec.n_total_free, self.mspec.n_total_free))
        else:
            H = np.zeros((self.mspec.n_total_free, self.mspec.n_total_free))
        for i in range(self.n_groups):
            group_free = self._free_to_group_free(free, i)
            par = self._free_to_par(group_free, i)
            L, B, F, P, a, b = self._par_to_model_mats(par, i)
            B = np.linalg.inv(np.eye(B.shape[0]) - B)   
            LB = np.dot(L, B)
            mu = (a+LB.dot(b.T).T).reshape(-1)
            d = (self.mspec.sample_means[i].flatten() - mu)
            a, b = a.flatten(), b.flatten()
            kws = self._hess_kws
            Sigma = LB.dot(F).dot(LB.T) + P
            S = self.mspec.sample_covs[i]
            R = S - Sigma
            Sinv = np.linalg.inv(Sigma)
            VRV = Sinv.dot(R).dot(Sinv)
            vecVRV = _vec(VRV)
            vecV = _vec(Sinv)
            Hi = np.zeros((self.nf,)*2)
            d1Sm = np.zeros((self.nf, self.p, self.p+1))
            if method == 0:
                Hi = _d2loglike_mu0(H=Hi, d1Sm=d1Sm, L=L, B=B, F=F, P=P, a=a, b=b, Sinv=Sinv, S=S, d=d, 
                               vecVRV=vecVRV, vecV=vecV, **kws)
            elif method == 1:
                Hi = _d2loglike_mu1(H=Hi, d1Sm=d1Sm, L=L, B=B, F=F, P=P, a=a, b=b, Sinv=Sinv, S=S, d=d, 
                               vecVRV=vecVRV, vecV=vecV, **kws)
            elif method == 2:
                Hi = _d2loglike_mu2(H=Hi, d1Sm=d1Sm, L=L, B=B, F=F, P=P, a=a, b=b, Sinv=Sinv, S=S, d=d, 
                               vecVRV=vecVRV, vecV=vecV, **kws)
            Hi = _sparse_post_mult(self.mspec.free_to_group_free[i].T.dot(Hi),
                                     self.mspec.free_to_group_free[i]) * self.gweights[i]
            if per_group:
                H[i] = Hi
            else:
                H = H + Hi
        J = self.mspec.theta_to_group_free
        H = _sparse_post_mult(J.dot(H), J.T)
        return H
    
    def _par_to_model_mats(self, par, i):
        slices = self.indexers[i].slices
        shapes = self.indexers[i].shapes
        L = _invec(par[slices[0]], *shapes[0])
        B = _invec(par[slices[1]], *shapes[1])
        F = _invech(par[slices[2]])
        P = _invech(par[slices[3]])
        a = _invec(par[slices[4]], *shapes[4])
        b = _invec(par[slices[5]], *shapes[5])
        return L, B, F, P, a, b

    def _free_to_par(self, free, i):
        par = self.p_templates[i].copy()
        if np.iscomplexobj(free):
            par = par.astype(complex)
        par[self.indexers[i].flat_indices] = free
        return par
    
    def _free_to_group_free(self, free, i):
        S = self.mspec.free_to_group_free[i]
        group_free = S.dot(free)
        return group_free

        
    def _theta_to_free(self, theta):
        free = self.mspec.theta_to_group_free.T.dot(theta)
        return free

    def _implied_cov_mean(self, L, B, F, P, a, b):
        B = np.linalg.inv(np.eye(B.shape[0])-B)
        LB = np.dot(L, B)
        Sigma = LB.dot(F).dot(LB.T) + P
        mu = (a+LB.dot(b.T).T).reshape(-1)
        return Sigma, mu
    
    
    def implied_sample_stats(self, free):
        L, B, F, P, a, b = self.free_to_model_mats(free)
        B = np.linalg.inv(np.eye(B.shape[0])-B)
        LB = np.dot(L, B)
        Sigma = LB.dot(F).dot(LB.T) + P
        mu = (a+LB.dot(b.T).T).reshape(-1)
        s, mu = _vech(Sigma), mu.flatten()
        sm = np.concatenate([s, mu])
        return sm
    
    def loglike(self, free, reduce=True):
        Sigma, mu = self.implied_cov_mean(free)
        L = sp.linalg.cholesky(Sigma) #np.linalg.cholesky(Sigma)
        Y = self.model_data.data - mu
        Z = sp.linalg.solve_triangular(L, Y.T, trans=1).T #np.dot(X, np.linalg.inv(L.T))
        t1 = 2.0 * np.log(np.diag(L)).sum() 
        t2 = np.sum(Z**2, axis=1)
        ll = (t1 + t2 + self.ll_const) / 2
        if reduce:
            ll = np.sum(ll)
        return ll
    
    def _fit(self, theta_init=None, minimize_kws=None, minimize_options=None, use_hess=False):
        bounds = self.bounds_theta
        func = self.func
        grad = self.gradient
        if use_hess:
            hess = self.hessian
        else:
            hess = None
        theta = self.theta if theta_init is None else theta_init
        default_minimize_options = dict(initial_tr_radius=1.0, verbose=3)
        minimize_options = handle_default_kws(minimize_options, default_minimize_options)
                
        default_minimize_kws = dict(method="trust-constr", options=minimize_options)
        minimize_kws = handle_default_kws(minimize_kws, default_minimize_kws)
        res = sp.optimize.minimize(func, x0=theta, jac=grad, hess=hess,  
                                   bounds=bounds,  **minimize_kws)
        return res
    
    def fit(self,  theta_init=None, minimize_kws=None, minimize_options=None, use_hess=False):
        res = self._fit(theta_init=theta_init, minimize_kws=minimize_kws, minimize_options=minimize_options, use_hess=use_hess)
        if np.linalg.norm(res.grad)>1e16:
            if minimize_options is None:
                minimize_options = {}
            minimize_options["initial_tr_radius"]=0.01
            res = self._fit(minimize_kws=minimize_kws, minimize_options=minimize_options, use_hess=use_hess)
        self.opt_res = res
        self.theta = res.x
        self.free = self._theta_to_free(self.theta)
        self.res = pd.DataFrame(res.x, index=self.theta_names, 
                                columns=["estimate"])
        self.res["se"] = np.sqrt(np.diag(np.linalg.inv(self.hessian(self.theta)*self.model_data.n_obs/2)))
        mats = {}
        free = self._theta_to_free(self.theta)
        mat_names = ["L", "B", "F", "P", "a", "b"]
        for i in range(self.n_groups):
            group_free = self._free_to_group_free(free, i)
            par = self._free_to_par(group_free, i)
            mlist = self._par_to_model_mats(par, i)
            mats[i] = {}
            for j,  mat in enumerate(mlist):
                mats[i][j] = pd.DataFrame(mat, index=self.mspec.mat_rows[i][j],
                                          columns=self.mspec.mat_cols[i][j])
        self.mats = mats
        
