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
from .cov_derivatives import _d2sigma, _dsigma
from ..utilities.func_utils import handle_default_kws
from ..utilities.data_utils import cov
from ..utilities.linalg_operations import _vech, _vec, _invec, _invech

def _sparse_post_mult(A, S):
    prod = S.T.dot(A.T)
    prod = prod.T
    return prod


class SEM(CovarianceStructure):
    
    def __init__(self, formula, data=None, sample_cov=None, sample_mean=None, n_obs=None,
                 model_spec_kws=None, fit_function=LikelihoodObjective):
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
        self.fit_function = LikelihoodObjective(data)
        self.sample_cov = data.sample_cov
        self.data = data
        self.means = data.sample_mean
        self.n_obs = data.n_obs
    
    def func(self, theta):
        Sigma = self.implied_cov(theta)
        f = self.fit_function.function(Sigma)
        return f
    
    def gradient(self, theta):
        Sigma = self.implied_cov(theta)
        dSigma = self.dsigma(theta)
        g = self.fit_function.gradient(Sigma, dSigma)
        g = self.J_theta.T.dot(g)
        return g

    def hessian(self, theta):
        Sigma = self.implied_cov(theta)
        free = self.theta_to_free(theta)
        dSigma = self.dsigma_free(free)
        d2Sigma = self.d2sigma_free(free)
        H = self.fit_function.hessian(Sigma, dSigma, d2Sigma)
        J = self.J_theta
        H = J.T.dot(H)
        H = _sparse_post_mult(H, J)
        return H
    
    def func_free(self, free):
        Sigma = self.implied_cov_free(free)
        f = self.fit_function.function(Sigma)
        return f
    
    def gradient_free(self, free):
        Sigma = self.implied_cov_free(free)
        dSigma = self.dsigma_free(free)
        g = self.fit_function.gradient(Sigma, dSigma)
        return g

    def hessian_free(self, free):
        Sigma = self.implied_cov_free(free)
        dSigma = self.dsigma_free(free)
        d2Sigma = self.d2sigma_free(free)
        H = self.fit_function.hessian(Sigma, dSigma, d2Sigma)
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
        self.n_params = len(res.x)
        self.theta_hess = self.hessian(self.theta)
        self.theta_cov = np.linalg.pinv(self.theta_hess*self.n_obs / 2)
        self.theta_se = np.sqrt(np.diag(self.theta_cov))
        self.L, self.B, self.F, self.P = self.to_model_mats(self.theta)
        ov_names, lv_names = self._row_col_names["L"]
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
        minds, mdims = self.minds, self.mdims
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
        dS, dA, r, c = self.dSf.copy(), self.dAf, self.rf, self.cf
        mtype, ns = self.mtypef, self.nf
        dS = _dsigma(dS, L, B, F, dA, r, c, mtype, ns)
        return dS
 
    def dsigma_free(self, free):
        par = self.free_to_par(free)
        L, B, F, P = self.par_to_mat(par)
        B = np.linalg.inv(np.eye(B.shape[0]) - B)
        dS, dA, r, c = self.dSf.copy(), self.dAf, self.rf, self.cf
        mtype, ns = self.mtypef, self.nf
        dS = _dsigma(dS, L, B, F, dA, r, c, mtype, ns)
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
        d2S, dA, r, c = self.d2Sf.copy(), self.dAf, self.rf, self.cf
        ltr_inds, htype, ns2 = self.d2_indsf, self.d2_kindf, self.nf2
        d2S = _d2sigma(d2S, L, B, F, dA, r, c, ltr_inds, htype, ns2)
        return d2S
    
    def d2sigma_free(self, free):
        par = self.free_to_par(free)
        L, B, F, P = self.par_to_mat(par)
        B = np.linalg.inv(np.eye(B.shape[0]) - B)
        d2S, dA, r, c = self.d2Sf.copy(), self.dAf, self.rf, self.cf
        ltr_inds, htype, ns2 = self.d2_indsf, self.d2_kindf, self.nf2
        d2S = _d2sigma(d2S, L, B, F, dA, r, c, ltr_inds, htype, ns2)
        return d2S
    
        
            
            
    
    
    
    