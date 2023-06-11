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
from .formula import ModelSpecification
from .model_data import ModelData
from .cov_derivatives import _d2sigma, _dsigma, _dloglike, _d2loglike
from ..utilities.func_utils import handle_default_kws
from ..utilities.data_utils import cov
from ..utilities.linalg_operations import _vech, _vec, _invec, _invech
from ..utilities.output import get_param_table
def _sparse_post_mult(A, S):
    prod = S.T.dot(A.T)
    prod = prod.T
    return prod


class SEM(CovarianceStructure):
    
    def __init__(self, formula, data=None, sample_cov=None, sample_mean=None, n_obs=None,
                 model_spec_kws=None, fit_function=LikelihoodObjective, mean_structure=False):
        data = ModelData(data=data, sample_cov=sample_cov,
                         sample_mean=sample_mean,  
                         n_obs=n_obs, ddof=0)
        
        default_model_spec_kws = dict(extension_kws=dict(fix_lv_var=False))
        model_spec_kws = handle_default_kws(model_spec_kws, default_model_spec_kws)
        
        var_order = dict(zip(data.sample_cov_df.columns,
                             np.arange(len(data.sample_cov_df.columns))))
        model_spec = ModelSpecification(formula, var_order=var_order, **model_spec_kws)
        lv_ov = set(model_spec.names["lv_extended"]).difference(
            set.union(model_spec.names["lv"], model_spec.names["y"], model_spec.names["v"]))
        lv_ov = sorted(lv_ov, key=lambda x: model_spec.lv_order[x])
        C = data.sample_cov_df.loc[lv_ov, lv_ov] #data[lv_ov].cov(ddof=0)
        model_spec.fixed_mats[2].loc[lv_ov, lv_ov] = C
        matrix_names = ["L", "B", "F", "P"]
        matrix_order = dict(L=0, B=1, F=2, P=3)
        init_kws = {}
        for name in matrix_names:
            i = matrix_order[name]
            init_kws[f"{name}_free"]  = model_spec.free_mats[i].values
            init_kws[f"{name}_fixed"] = model_spec.fixed_mats[i].values
            init_kws[f"{name}_fixed_loc"] = model_spec.fixed_mats[i].values!=0
        super().__init__(**init_kws)
        self.model_spec = model_spec
        self.sample_cov = data.sample_cov
        self.data = data
        self.means = data.sample_mean
        self.n_obs = data.n_obs
        self.lndetS = np.linalg.slogdet(data.sample_cov)[1]
        self.dA, self.matrix_type = self.cov_der.dA, self.cov_der.matrix_type
        self.r, self.c = self.cov_der.r, self.cov_der.c
        self.d2_kind =self.cov_der.d2_kind
        self.dS = self.cov_der.dS
        self.d2S = self.cov_der.d2S
        self.nf=self.nf1
        self._vech_inds = self.cov_der._vech_inds
        self.parameter_indices = self.cov_der.parameter_indices
        self.matrix_dims = self.cov_der.matrix_dims

        self.J_theta = self.cov_der.J_theta

    
    def func(self, theta):
        Sigma = self.implied_cov(theta)
        n_obs, C = self.data.n_obs, self.data.sample_cov
        lndS = np.linalg.slogdet(Sigma)[1]
        SinvC = np.linalg.solve(Sigma, C)
        trSinvC = np.trace(SinvC)
        f = (lndS + trSinvC - self.lndetS - len(C)) * n_obs / 2.0
        return f
    
    def gradient(self, theta):
        par = self.free_to_par(self.theta_to_free(theta))
        L, B, F, P = self.par_to_mat(par)
        B = np.linalg.inv(np.eye(B.shape[0]) - B)
        LB = np.dot(L, B)
        Sigma = LB.dot(F).dot(LB.T) + P
        dA, r, c, deriv_type = self.dA, self.r, self.c, self.matrix_type
        n_obs, C = self.data.n_obs, self.data.sample_cov
        R = C - Sigma
        Sinv = np.linalg.inv(Sigma)
        VRV = Sinv.dot(R).dot(Sinv)
        vecVRV = _vec(VRV)
        g = np.zeros(self.nf)
        g =_dloglike(g, L, B, F, vecVRV, dA, r, c, deriv_type, self.nf, self._vech_inds)
        g = n_obs  / 2 * g
        g = self.J_theta.T.dot(g)
        return g

    def hessian(self, theta):
        par = self.free_to_par(self.theta_to_free(theta))
        L, B, F, P = self.par_to_mat(par)
        B = np.linalg.inv(np.eye(B.shape[0]) - B)
        LB = np.dot(L, B)
        Sigma = LB.dot(F).dot(LB.T) + P
        dA, r, c, deriv_type = self.dA, self.r, self.c, self.matrix_type
        n_obs, C = self.data.n_obs, self.data.sample_cov
        R = C - Sigma
        Sinv = np.linalg.inv(Sigma)
        VRV = Sinv.dot(R).dot(Sinv)
        vecVRV = _vec(VRV)
        vecV = _vec(Sinv)
        H = np.zeros((self.nf,)*2)
        H = _d2loglike(H, L, B, F, Sinv, C, vecVRV, vecV, dA, r, c, 
                       deriv_type, self.d2_kind, self.nf, self._vech_inds)
        J = self.J_theta
        H = J.T.dot(H)
        H = _sparse_post_mult(H, J)
        H = n_obs / 2 * H
        return H
    
    def func_free(self, free):
        Sigma = self.implied_cov_free(free)
        n_obs, C = self.data.n_obs, self.data.sample_cov
        lndS = np.linalg.slogdet(Sigma)[1]
        SinvC = np.linalg.solve(Sigma, C)
        trSinvC = np.trace(SinvC)
        f = (lndS + trSinvC - self.lndetS - len(C)) * n_obs / 2.0
        return f
    
    def gradient_free(self, free):
        par = self.free_to_par(free)
        L, B, F, P = self.par_to_mat(par)
        B = np.linalg.inv(np.eye(B.shape[0]) - B)
        LB = np.dot(L, B)
        Sigma = LB.dot(F).dot(LB.T) + P
        dA, r, c, deriv_type = self.dA, self.r, self.c, self.matrix_type
        n_obs, C = self.data.n_obs, self.data.sample_cov
        R = C - Sigma
        Sinv = np.linalg.inv(Sigma)
        VRV = Sinv.dot(R).dot(Sinv)
        vecVRV = _vec(VRV)
        g = np.zeros(self.nf)
        g =_dloglike(g, L, B, F, vecVRV, dA, r, c, deriv_type, self.nf, self._vech_inds)
        g = n_obs  / 2 * g
        return g

    def hessian_free(self, free):
        par = self.free_to_par(free)
        L, B, F, P = self.par_to_mat(par)
        B = np.linalg.inv(np.eye(B.shape[0]) - B)
        LB = np.dot(L, B)
        Sigma = LB.dot(F).dot(LB.T) + P
        dA, r, c, deriv_type = self.dA, self.r, self.c, self.matrix_type
        n_obs, C = self.data.n_obs, self.data.sample_cov
        R = C - Sigma
        Sinv = np.linalg.inv(Sigma)
        VRV = Sinv.dot(R).dot(Sinv)
        vecVRV = _vec(VRV)
        vecV = _vec(Sinv)
        H = np.zeros((self.nf,)*2)
        H = _d2loglike(H, L, B, F, Sinv, C, vecVRV, vecV, dA, r, c, 
                       deriv_type, self.d2_kind, self.nf, self._vech_inds)
        H = n_obs / 2 * H
        return H
    
    def fit(self,  minimize_kws=None, minimize_options=None, constrain=False, use_hess=False):
        x = self.theta.copy()
        bounds = self.make_bounds()
        if constrain:
            constraints = self.make_constraints()
        else:
            constraints = None
        fun = self.func
        jac = self.gradient
        if use_hess:
            hess = self.hessian
        else:
            hess=None
        
        default_minimize_options = dict(initial_tr_radius=1.0, verbose=3)
        minimize_options = handle_default_kws(minimize_options, default_minimize_options)
        
        default_minimize_kws = dict(method="trust-constr", options=minimize_options)
        minimize_kws = handle_default_kws(minimize_kws, default_minimize_kws)
        
        res = sp.optimize.minimize(fun,x0=x, jac=jac, hess=hess, bounds=bounds,
                                   constraints=constraints, **minimize_kws)
        self.opt_res = res
        self.theta = res.x
        self.free = self.theta_to_free(self.theta)
        self.n_params = len(res.x)
        self.theta_hess = self.hessian(self.theta)
        self.theta_cov = np.linalg.pinv(self.theta_hess)
        self.theta_se = np.sqrt(np.diag(self.theta_cov))
        self.free_hess = self.hessian_free(self.free)
        self.free_cov = np.linalg.pinv(self.free_hess)
        self.free_se = self.theta_to_free(self.theta_se)
        self.L, self.B, self.F, self.P = self.to_model_mats(self.theta)
        ov_names, lv_names = self._row_col_names["L"]
        self.param_names = (self.model_spec.ptable.loc[self.model_spec.ptable["free"]!=0,
                                                  ["lhs", "rel","rhs"]].astype(str).agg(' '.join, axis=1))
        self.res = get_param_table(self.free, self.free_se, self.data.n_obs-self.n_params,
                                   index=self.param_names)
        self.Ldf = pd.DataFrame(self.L, index=ov_names, columns=lv_names)
        self.Bdf = pd.DataFrame(self.B, index=lv_names, columns=lv_names)
        self.Fdf = pd.DataFrame(self.F, index=lv_names, columns=lv_names)
        self.Pdf = pd.DataFrame(self.P, index=ov_names, columns=ov_names)
        
    def free_to_par(self, free):
        par = self.p_template.copy()
        par[self.par_to_free_ind] = free
        return par
    
    def theta_to_free(self, theta):
        free = theta[self.theta_to_free_ind]
        return free

    def par_to_mat(self, par):
        minds = self.parameter_indices
        mdims = self.matrix_dims
        L, B = _invec(par[minds[0]], *mdims[0]), _invec(par[minds[1]], *mdims[1])
        F, P = _invech(par[minds[2]]),  _invech(par[minds[3]])
        return L, B, F, P
    
    def to_model_mats(self, theta):
        par = self.free_to_par(self.theta_to_free(theta))
        L, B, F, P = self.par_to_mat(par)
        return L, B, F, P
    
    def dsigma(self, theta):
        par = self.free_to_par(self.theta_to_free(theta))
        L, B, F, P = self.par_to_mat(par)
        B = np.linalg.inv(np.eye(B.shape[0]) - B)
        dS = self.dS.copy()
        kws = self.cov_der.cov_first_deriv_kws
        dS = _dsigma(dS, L, B, F, **kws)
        return dS
 
    def dsigma_free(self, free):
        par = self.free_to_par(free)
        L, B, F, P = self.par_to_mat(par)
        B = np.linalg.inv(np.eye(B.shape[0]) - B)
        dS = self.dS.copy()
        kws = self.cov_der.cov_first_deriv_kws
        dS = _dsigma(dS, L, B, F, **kws)
        return dS
    
    def implied_cov(self, theta):
        par = self.free_to_par(self.theta_to_free(theta))
        L, B, F, P = self.par_to_mat(par)
        B = np.linalg.inv(np.eye(B.shape[0]) - B)
        LB = np.dot(L, B)
        S = LB.dot(F).dot(LB.T) + P
        return S
    
    def implied_cov_free(self, free):
        par = self.free_to_par(free)
        L, B, F, P = self.par_to_mat(par)
        B = np.linalg.inv(np.eye(B.shape[0]) - B)
        LB = np.dot(L, B)
        S = LB.dot(F).dot(LB.T) + P
        return S
        
    def d2sigma(self, theta):
        par = self.free_to_par(self.theta_to_free(theta))
        L, B, F, P = self.par_to_mat(par)
        B = np.linalg.inv(np.eye(B.shape[0]) - B)
        d2S = self.d2S.copy()
        kws = self.cov_der.cov_second_deriv_kws
        d2S = _d2sigma(d2S, L, B, F, **kws)
        return d2S
    
    def d2sigma_free(self, free):
        par = self.free_to_par(free)
        L, B, F, P = self.par_to_mat(par)
        B = np.linalg.inv(np.eye(B.shape[0]) - B)
        d2S = self.d2S.copy()
        kws = self.cov_der.cov_second_deriv_kws
        d2S = _d2sigma(d2S, L, B, F, **kws)
        return d2S
    
        
            
            
    
    
    
    