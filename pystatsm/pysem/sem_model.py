#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 20:06:29 2023

@author: lukepinkel
"""

import numpy as np
import scipy as sp
import pandas as pd
from .cov_model import CovarianceStructure
from .fitfunctions import LikelihoodObjective
from .formula import FormulaParser#ModelSpecification
from .model_data import ModelData
from .derivatives import _dsigma_mu, _dloglike_mu, _d2sigma_mu, _d2loglike_mu0, _d2loglike_mu1, _d2loglike_mu2 #from .cov_derivatives import _d2sigma, _dsigma, _dloglike, _d2loglike
from ..utilities.func_utils import handle_default_kws, triangular_number
from ..utilities.data_utils import cov
from ..utilities.linalg_operations import _vech, _vec, _invec, _invech
from ..utilities.output import get_param_table
def _sparse_post_mult(A, S):
    prod = S.T.dot(A.T)
    prod = prod.T
    return prod





class SEM:
    
    def __init__(self, formula, data, model_spec_kws=None):
        default_model_spec_kws = dict(extension_kws=dict(fix_lv_var=False))
        model_spec_kws = handle_default_kws(model_spec_kws, default_model_spec_kws)
        
        self.model_data = ModelData(data=data)
        ov_names = self.model_data.sample_cov_df.columns
        nov = len(ov_names)
        var_order = dict(zip(ov_names, np.arange(nov)))
        
        self.model_spec = FormulaParser(formula, var_order=var_order, **model_spec_kws)
        self.model_spec.to_model_mats()
        self.model_indexer = self.model_spec.create_indexer(self.model_spec.free_mats)
        self.model_spec.update_ptable_with_data(self.model_data.sample_cov_df, self.model_data.sample_mean_df)
        self.model_spec.to_model_mats()
        self.model_spec.add_bounds_to_table()
        bounds = self.model_spec.ptable.loc[~self.model_spec.ptable["fixed"], ["lb", "ub"]]
        bounds = bounds.values
        self.bounds = [tuple(x) for x in bounds.tolist()]
        self.bounds_theta = [tuple(x) for x in bounds[self.model_indexer.unique_indices]]
        self.p_template = self.model_spec.to_param_template()
        self.p_template[np.isnan(self.p_template)] = 0.0

        self.free = self.p_template[self.model_indexer.flat_indices]
        self.p, self.q = self.model_spec.p, self.model_spec.q
        self.p2, self.q2 = triangular_number(self.p), triangular_number(self.q)
        self.nf = len(self.model_indexer.flat_indices)
        self.nf2 = triangular_number(self.nf)
        self.nt = len(self.model_indexer.first_locs)
        self.make_derivative_matrices()
        self.theta = self.free[self.model_indexer.unique_indices]
        self.n_par = len(self.p_template)
        self.ll_const =  1.8378770664093453 * self.p

        

    def make_derivative_matrices(self):
        self.model_indexer.create_derivative_arrays([(0, 0), (1, 0), (2, 0), (1 ,1), (2, 1), (5, 0), (5, 1)])
        self.dA = self.model_indexer.dA
        self.dSm = np.zeros((self.p2+self.p, self.nf))
        self.d2Sm = np.zeros((self.p2+self.p, self.nf, self.nf))
        self.m_size = self.model_indexer.block_sizes                    
        self.m_kind = self.model_indexer.block_indices                 
        self.d2_kind = self.model_indexer.block_pair_types             
        self.d2_inds = self.model_indexer.colex_descending_inds  
        self.J_theta = sp.sparse.csc_array(
            (np.ones(self.nf), (np.arange(self.nf), self.model_indexer.unique_locs)), 
             shape=(self.nf, self.nt))
        self.dSm = np.zeros((self.p2+self.p, self.nf))
        self.d2Sm = np.zeros((self.p2+self.p, self.nf, self.nf))
        s, r = np.triu_indices(self.p, k=0)
        self._vech_inds = r+s*self.p
        self.unique_locs = self.model_indexer.unique_locs
        self.ptable = self.model_spec.ptable
        self.free_table = self.model_spec.ptable.loc[~self.model_spec.ptable["fixed"]]
        self.free_names = self.free_table[["lhs", "rel","rhs"]].astype(str).agg(' '.join, axis=1).values
        self.theta_names = self.free_names[self.model_indexer.unique_indices]
        
        
    def par_to_model_mats(self, par):
         slices = self.model_indexer.slices
         shapes = self.model_indexer.shapes
         L = _invec(par[slices[0]], *shapes[0])
         B = _invec(par[slices[1]], *shapes[1])
         F = _invech(par[slices[2]])
         P = _invech(par[slices[3]])
         a = _invec(par[slices[4]], *shapes[4])
         b = _invec(par[slices[5]], *shapes[5])
         return L, B, F, P, a, b
     

    def free_to_par(self, free):
        par = self.p_template.copy()
        if np.any(np.iscomplex(free)):
            par = par.astype(complex)
        par[self.model_indexer.flat_indices] = free
        return par
    
    def theta_to_free(self, theta):
        free = theta[self.unique_locs]
        return free

    def theta_to_model_mats(self, theta):
        return self.par_to_model_mats(self.free_to_par(self.theta_to_free(theta)))
        
    def free_to_model_mats(self, free):
        par = self.free_to_par(free)
        L, B, F, P, a, b = self.par_to_model_mats(par)
        return L, B, F, P, a, b
    
    def implied_cov_mean(self, free):
        L, B, F, P, a, b = self.free_to_model_mats(free)
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
    
    def func(self, free):
        Sigma, mu = self.implied_cov_mean(free)
        r = (self.model_data.sample_mean-mu).flatten()
        Sinv = np.linalg.inv(Sigma)
        trSV = np.trace(Sinv.dot(self.model_data.sample_cov))
        rVr = np.dot(r.T.dot(Sinv), r) 
        if np.any(np.iscomplex(Sigma)):
            s, lnd = np.linalg.slogdet(Sigma)
            lndS = np.log(s)+lnd
        else:
            s, lndS = np.linalg.slogdet(Sigma)
        f = rVr + lndS + trSV
        if s==-1:
            f = np.inf
        return f
    
    def func_theta(self, theta):
        free = self.theta_to_free(theta)
        f = self.func(free)
        return f
    
    def dsigma_mu(self, free):
        L, B, F, _, a, b = self.free_to_model_mats(free)
        B = np.linalg.inv(np.eye(B.shape[0])-B)
        kws =  dict(dA=self.dA, m_size=self.m_size, m_type=self.m_kind,
                    vind=self._vech_inds, n=self.nf, p2=self.p2)
        a, b = a.flatten(), b.flatten()
        dSm = _dsigma_mu(self.dSm.copy(), L, B, F, b, a,**kws)
        return dSm
    
    def d2sigma_mu(self, free):
        L, B, F, _, a, b = self.free_to_model_mats(free)
        B = np.linalg.inv(np.eye(B.shape[0])-B)
        a, b = a.flatten(), b.flatten()
        kws =  dict(dA=self.dA, m_size=self.m_size,  m_type=self.d2_kind, 
                    d2_inds=self.d2_inds, vind=self._vech_inds, n=self.nf2, 
                    p2=self.p2)
        dSm = _d2sigma_mu(self.d2Sm.copy(), L, B, F, b, a,**kws)
        return dSm
    
    def gradient(self, free):
        L, B, F, P, a, b = self.free_to_model_mats(free)
        B = np.linalg.inv(np.eye(B.shape[0]) - B)
        LB = np.dot(L, B)
        mu = (a+LB.dot(b.T).T).reshape(-1)
        a, b = a.flatten(), b.flatten()
        kws =  dict(dA=self.dA, m_size=self.m_size, m_type=self.m_kind,
                    vind=self._vech_inds, n=self.nf, p2=self.p2)
       
        Sigma = LB.dot(F).dot(LB.T) + P
        R = self.model_data.sample_cov - Sigma
        Sinv = np.linalg.inv(Sigma)
        VRV = Sinv.dot(R).dot(Sinv)
        vecVRV = _vec(VRV)
        rtV = (self.model_data.sample_mean.flatten() - mu).dot(Sinv)
        g = np.zeros(self.nf)
        g = _dloglike_mu(g, L, B, F, b, a,vecVRV, rtV, **kws)
        return g
    
    def gradient_obs(self, free):
        Sigma, mu = self.implied_cov_mean(free)
        dmu_dfree = self.dsigma_mu(free)
        dS = dmu_dfree[:-self.p]
        dm = dmu_dfree[-self.p:]
        dS = _invech(dS.T)
        V = np.linalg.inv(Sigma)
        DS = dS.reshape(self.nf, -1, order='F')
        t1 = DS.dot(V.reshape(-1,order='F')).reshape(1, -1)
        Y = self.model_data.data - mu
        YV = Y.dot(V)
        t2 = YV.dot(dm)
        t3 = np.einsum("ij,hjk,ik->ih", YV, dS, YV, optimize=True)
        grad_obs = -2 * (-t1 / 2 + t2 + t3 / 2)
        return grad_obs
        

    
    def gradient_theta(self, theta):
        free = self.theta_to_free(theta)
        L, B, F, P, a, b = self.free_to_model_mats(free)
        B = np.linalg.inv(np.eye(B.shape[0]) - B)
        LB = np.dot(L, B)
        mu = (a+LB.dot(b.T).T).reshape(-1)
        a, b = a.flatten(), b.flatten()
        kws =  dict(dA=self.dA, m_size=self.m_size, m_type=self.m_kind,
                    vind=self._vech_inds, n=self.nf, p2=self.p2)
       
        Sigma = LB.dot(F).dot(LB.T) + P
        R = self.model_data.sample_cov - Sigma
        Sinv = np.linalg.inv(Sigma)
        VRV = Sinv.dot(R).dot(Sinv)
        vecVRV = _vec(VRV)
        rtV = (self.model_data.sample_mean.flatten() - mu).dot(Sinv)
        g = np.zeros(self.nf)
        g = _dloglike_mu(g, L, B, F, b, a,vecVRV, rtV, **kws)
        g = self.J_theta.T.dot(g)
        return g
    
    def hessian(self, free, method=0):
        L, B, F, P, a, b = self.free_to_model_mats(free)
        B = np.linalg.inv(np.eye(B.shape[0]) - B)
        LB = np.dot(L, B)
        mu = (a+LB.dot(b.T).T).reshape(-1)
        d = (self.model_data.sample_mean.flatten() - mu)
        a, b = a.flatten(), b.flatten()
        kws = dict(dA=self.dA, m_size=self.m_size, d2_inds=self.d2_inds, first_deriv_type=self.m_kind, 
                   second_deriv_type=self.d2_kind,  n=self.nf,  vech_inds=self._vech_inds)

        Sigma = LB.dot(F).dot(LB.T) + P
        S = self.model_data.sample_cov
        R = S - Sigma
        Sinv = np.linalg.inv(Sigma)
        VRV = Sinv.dot(R).dot(Sinv)
        vecVRV = _vec(VRV)
        vecV = _vec(Sinv)
        H = np.zeros((self.nf,)*2)
        d1Sm = np.zeros((self.nf, self.p, self.p+1))
        if method == 0:
            H = _d2loglike_mu0(H=H, d1Sm=d1Sm, L=L, B=B, F=F, P=P, a=a, b=b, Sinv=Sinv, S=S, d=d, 
                           vecVRV=vecVRV, vecV=vecV, **kws)
        elif method == 1:
            H = _d2loglike_mu1(H=H, d1Sm=d1Sm, L=L, B=B, F=F, P=P, a=a, b=b, Sinv=Sinv, S=S, d=d, 
                           vecVRV=vecVRV, vecV=vecV, **kws)
        elif method == 2:
            H = _d2loglike_mu2(H=H, d1Sm=d1Sm, L=L, B=B, F=F, P=P, a=a, b=b, Sinv=Sinv, S=S, d=d, 
                           vecVRV=vecVRV, vecV=vecV, **kws)
        return H
    
    def hessian_theta(self, theta, method=0):
        free = self.theta_to_free(theta)
        H = self.hessian(free, method=method)
        H = _sparse_post_mult(self.J_theta.T.dot(H), self.J_theta)
        return H

    def _fit(self, theta_init=None, minimize_kws=None, minimize_options=None, use_hess=False):
        bounds = self.bounds_theta
        func = self.func_theta
        grad = self.gradient_theta
        if use_hess:
            hess = self.hessian_theta
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
        self.free = self.theta_to_free(self.theta)
        self.res = pd.DataFrame(res.x, index=self.theta_names, 
                                columns=["estimate"])
        self.res["se"] = np.sqrt(np.diag(np.linalg.inv(self.hessian_theta(self.theta)*self.model_data.n_obs/2)))
        mlist = list(self.theta_to_model_mats(self.theta))
        mat_names = ["L", "B", "F", "P", "a", "b"]
        for i,  mat in enumerate(mlist):
            mat = pd.DataFrame(mat, index=self.model_spec.mat_row_names[i],
                               columns=self.model_spec.mat_col_names[i])
            setattr(self, mat_names[i], mat)
                    
    
    
# class SEM(CovarianceStructure):
    
#     def __init__(self, formula, data=None, sample_cov=None, sample_mean=None, n_obs=None,
#                  model_spec_kws=None, fit_function=LikelihoodObjective, mean_structure=False):
#         data = ModelData(data=data, sample_cov=sample_cov,
#                          sample_mean=sample_mean,  
#                          n_obs=n_obs, ddof=0)
        
#         default_model_spec_kws = dict(extension_kws=dict(fix_lv_var=False))
#         model_spec_kws = handle_default_kws(model_spec_kws, default_model_spec_kws)
        
#         var_order = dict(zip(data.sample_cov_df.columns,
#                              np.arange(len(data.sample_cov_df.columns))))
#         model_spec = ModelSpecification(formula, var_order=var_order, **model_spec_kws)
#         lv_ov = set(model_spec.names["lv_extended"]).difference(
#             set.union(model_spec.names["lv"], model_spec.names["y"], model_spec.names["v"]))
#         lv_ov = sorted(lv_ov, key=lambda x: model_spec.lv_order[x])
#         C = data.sample_cov_df.loc[lv_ov, lv_ov] #data[lv_ov].cov(ddof=0)
#         model_spec.fixed_mats[2].loc[lv_ov, lv_ov] = C
#         matrix_names = ["L", "B", "F", "P"]
#         matrix_order = dict(L=0, B=1, F=2, P=3)
#         init_kws = {}
#         for name in matrix_names:
#             i = matrix_order[name]
#             init_kws[f"{name}_free"]  = model_spec.free_mats[i].values
#             init_kws[f"{name}_fixed"] = model_spec.fixed_mats[i].values
#             init_kws[f"{name}_fixed_loc"] = model_spec.fixed_mats[i].values!=0
#         super().__init__(**init_kws)
#         self.model_spec = model_spec
#         self.sample_cov = data.sample_cov
#         self.data = data
#         self.means = data.sample_mean
#         self.n_obs = data.n_obs
#         self.lndetS = np.linalg.slogdet(data.sample_cov)[1]
#         self.dA, self.matrix_type = self.cov_der.dA, self.cov_der.matrix_type
#         self.r, self.c = self.cov_der.r, self.cov_der.c
#         self.d2_kind =self.cov_der.d2_kind
#         self.dS = self.cov_der.dS
#         self.d2S = self.cov_der.d2S
#         self.nf=self.nf1
#         self._vech_inds = self.cov_der._vech_inds
#         self.parameter_indices = self.cov_der.parameter_indices
#         self.matrix_dims = self.cov_der.matrix_dims

#         self.J_theta = self.cov_der.J_theta

    
#     def func(self, theta):
#         Sigma = self.implied_cov(theta)
#         n_obs, C = self.data.n_obs, self.data.sample_cov
#         lndS = np.linalg.slogdet(Sigma)[1]
#         SinvC = np.linalg.solve(Sigma, C)
#         trSinvC = np.trace(SinvC)
#         f = (lndS + trSinvC - self.lndetS - len(C)) * n_obs / 2.0
#         return f
    
#     def gradient(self, theta):
#         par = self.free_to_par(self.theta_to_free(theta))
#         L, B, F, P = self.par_to_mat(par)
#         B = np.linalg.inv(np.eye(B.shape[0]) - B)
#         LB = np.dot(L, B)
#         Sigma = LB.dot(F).dot(LB.T) + P
#         dA, r, c, deriv_type = self.dA, self.r, self.c, self.matrix_type
#         n_obs, C = self.data.n_obs, self.data.sample_cov
#         R = C - Sigma
#         Sinv = np.linalg.inv(Sigma)
#         VRV = Sinv.dot(R).dot(Sinv)
#         vecVRV = _vec(VRV)
#         g = np.zeros(self.nf)
#         g =_dloglike(g, L, B, F, vecVRV, dA, r, c, deriv_type, self.nf, self._vech_inds)
#         g = n_obs  / 2 * g
#         g = self.J_theta.T.dot(g)
#         return g

#     def hessian(self, theta):
#         par = self.free_to_par(self.theta_to_free(theta))
#         L, B, F, P = self.par_to_mat(par)
#         B = np.linalg.inv(np.eye(B.shape[0]) - B)
#         LB = np.dot(L, B)
#         Sigma = LB.dot(F).dot(LB.T) + P
#         dA, r, c, deriv_type = self.dA, self.r, self.c, self.matrix_type
#         n_obs, C = self.data.n_obs, self.data.sample_cov
#         R = C - Sigma
#         Sinv = np.linalg.inv(Sigma)
#         VRV = Sinv.dot(R).dot(Sinv)
#         vecVRV = _vec(VRV)
#         vecV = _vec(Sinv)
#         H = np.zeros((self.nf,)*2)
#         H = _d2loglike(H, L, B, F, Sinv, C, vecVRV, vecV, dA, r, c, 
#                        deriv_type, self.d2_kind, self.nf, self._vech_inds)
#         J = self.J_theta
#         H = J.T.dot(H)
#         H = _sparse_post_mult(H, J)
#         H = n_obs / 2 * H
#         return H
    
#     def func_free(self, free):
#         Sigma = self.implied_cov_free(free)
#         n_obs, C = self.data.n_obs, self.data.sample_cov
#         lndS = np.linalg.slogdet(Sigma)[1]
#         SinvC = np.linalg.solve(Sigma, C)
#         trSinvC = np.trace(SinvC)
#         f = (lndS + trSinvC - self.lndetS - len(C)) * n_obs / 2.0
#         return f
    
#     def gradient_free(self, free):
#         par = self.free_to_par(free)
#         L, B, F, P = self.par_to_mat(par)
#         B = np.linalg.inv(np.eye(B.shape[0]) - B)
#         LB = np.dot(L, B)
#         Sigma = LB.dot(F).dot(LB.T) + P
#         dA, r, c, deriv_type = self.dA, self.r, self.c, self.matrix_type
#         n_obs, C = self.data.n_obs, self.data.sample_cov
#         R = C - Sigma
#         Sinv = np.linalg.inv(Sigma)
#         VRV = Sinv.dot(R).dot(Sinv)
#         vecVRV = _vec(VRV)
#         g = np.zeros(self.nf)
#         g =_dloglike(g, L, B, F, vecVRV, dA, r, c, deriv_type, self.nf, self._vech_inds)
#         g = n_obs  / 2 * g
#         return g

#     def hessian_free(self, free):
#         par = self.free_to_par(free)
#         L, B, F, P = self.par_to_mat(par)
#         B = np.linalg.inv(np.eye(B.shape[0]) - B)
#         LB = np.dot(L, B)
#         Sigma = LB.dot(F).dot(LB.T) + P
#         dA, r, c, deriv_type = self.dA, self.r, self.c, self.matrix_type
#         n_obs, C = self.data.n_obs, self.data.sample_cov
#         R = C - Sigma
#         Sinv = np.linalg.inv(Sigma)
#         VRV = Sinv.dot(R).dot(Sinv)
#         vecVRV = _vec(VRV)
#         vecV = _vec(Sinv)
#         H = np.zeros((self.nf,)*2)
#         H = _d2loglike(H, L, B, F, Sinv, C, vecVRV, vecV, dA, r, c, 
#                        deriv_type, self.d2_kind, self.nf, self._vech_inds)
#         H = n_obs / 2 * H
#         return H
    
#     def fit(self,  minimize_kws=None, minimize_options=None, constrain=False, use_hess=False):
#         x = self.theta.copy()
#         bounds = self.make_bounds()
#         if constrain:
#             constraints = self.make_constraints()
#         else:
#             constraints = None
#         fun = self.func
#         jac = self.gradient
#         if use_hess:
#             hess = self.hessian
#         else:
#             hess=None
        
#         default_minimize_options = dict(initial_tr_radius=1.0, verbose=3)
#         minimize_options = handle_default_kws(minimize_options, default_minimize_options)
        
#         default_minimize_kws = dict(method="trust-constr", options=minimize_options)
#         minimize_kws = handle_default_kws(minimize_kws, default_minimize_kws)
        
#         res = sp.optimize.minimize(fun,x0=x, jac=jac, hess=hess, bounds=bounds,
#                                    constraints=constraints, **minimize_kws)
#         self.opt_res = res
#         self.theta = res.x
#         self.free = self.theta_to_free(self.theta)
#         self.n_params = len(res.x)
#         self.theta_hess = self.hessian(self.theta)
#         self.theta_cov = np.linalg.pinv(self.theta_hess)
#         self.theta_se = np.sqrt(np.diag(self.theta_cov))
#         self.free_hess = self.hessian_free(self.free)
#         self.free_cov = np.linalg.pinv(self.free_hess)
#         self.free_se = self.theta_to_free(self.theta_se)
#         self.L, self.B, self.F, self.P = self.to_model_mats(self.theta)
#         ov_names, lv_names = self._row_col_names["L"]
#         self.param_names = (self.model_spec.ptable.loc[self.model_spec.ptable["free"]!=0,
#                                                   ["lhs", "rel","rhs"]].astype(str).agg(' '.join, axis=1))
#         self.res = get_param_table(self.free, self.free_se, self.data.n_obs-self.n_params,
#                                    index=self.param_names)
#         self.Ldf = pd.DataFrame(self.L, index=ov_names, columns=lv_names)
#         self.Bdf = pd.DataFrame(self.B, index=lv_names, columns=lv_names)
#         self.Fdf = pd.DataFrame(self.F, index=lv_names, columns=lv_names)
#         self.Pdf = pd.DataFrame(self.P, index=ov_names, columns=ov_names)
        
#     def free_to_par(self, free):
#         par = self.p_template.copy()
#         par[self.par_to_free_ind] = free
#         return par
    
#     def theta_to_free(self, theta):
#         free = theta[self.theta_to_free_ind]
#         return free

#     def par_to_mat(self, par):
#         minds = self.parameter_indices
#         mdims = self.matrix_dims
#         L, B = _invec(par[minds[0]], *mdims[0]), _invec(par[minds[1]], *mdims[1])
#         F, P = _invech(par[minds[2]]),  _invech(par[minds[3]])
#         return L, B, F, P
    
#     def to_model_mats(self, theta):
#         par = self.free_to_par(self.theta_to_free(theta))
#         L, B, F, P = self.par_to_mat(par)
#         return L, B, F, P
    
#     def dsigma(self, theta):
#         par = self.free_to_par(self.theta_to_free(theta))
#         L, B, F, P = self.par_to_mat(par)
#         B = np.linalg.inv(np.eye(B.shape[0]) - B)
#         dS = self.dS.copy()
#         kws = self.cov_der.cov_first_deriv_kws
#         dS = _dsigma(dS, L, B, F, **kws)
#         return dS
 
#     def dsigma_free(self, free):
#         par = self.free_to_par(free)
#         L, B, F, P = self.par_to_mat(par)
#         B = np.linalg.inv(np.eye(B.shape[0]) - B)
#         dS = self.dS.copy()
#         kws = self.cov_der.cov_first_deriv_kws
#         dS = _dsigma(dS, L, B, F, **kws)
#         return dS
    
#     def implied_cov(self, theta):
#         par = self.free_to_par(self.theta_to_free(theta))
#         L, B, F, P = self.par_to_mat(par)
#         B = np.linalg.inv(np.eye(B.shape[0]) - B)
#         LB = np.dot(L, B)
#         S = LB.dot(F).dot(LB.T) + P
#         return S
    
#     def implied_cov_free(self, free):
#         par = self.free_to_par(free)
#         L, B, F, P = self.par_to_mat(par)
#         B = np.linalg.inv(np.eye(B.shape[0]) - B)
#         LB = np.dot(L, B)
#         S = LB.dot(F).dot(LB.T) + P
#         return S
        
#     def d2sigma(self, theta):
#         par = self.free_to_par(self.theta_to_free(theta))
#         L, B, F, P = self.par_to_mat(par)
#         B = np.linalg.inv(np.eye(B.shape[0]) - B)
#         d2S = self.d2S.copy()
#         kws = self.cov_der.cov_second_deriv_kws
#         d2S = _d2sigma(d2S, L, B, F, **kws)
#         return d2S
    
#     def d2sigma_free(self, free):
#         par = self.free_to_par(free)
#         L, B, F, P = self.par_to_mat(par)
#         B = np.linalg.inv(np.eye(B.shape[0]) - B)
#         d2S = self.d2S.copy()
#         kws = self.cov_der.cov_second_deriv_kws
#         d2S = _d2sigma(d2S, L, B, F, **kws)
#         return d2S
    
        
            
            
    
    
    
    