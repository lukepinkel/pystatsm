#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 20:06:29 2023

@author: lukepinkel
"""

import numpy as np
import scipy as sp
from .cov_model import CovarianceStructure
from .fitfunctions import LikelihoodObjective
from .formula import ModelSpecification

from ..utilities.func_utils import handle_default_kws
from ..utilities.data_utils import cov
class SEM(CovarianceStructure):
    
    def __init__(self, formula, data, model_spec_kws=None, 
                 fit_function=LikelihoodObjective):
        default_model_spec_kws = dict(extension_kws=dict(fix_lv_var=False))
        model_spec_kws = handle_default_kws(model_spec_kws, default_model_spec_kws)
        
        var_order = dict(zip(data.columns, np.arange(len(data.columns))))
        model_spec = ModelSpecification(formula, var_order=var_order, **model_spec_kws)
        lv_ov = set(model_spec.names["lv_extended"]).difference(set(model_spec.names["lv"]).union(model_spec.names["y"]))
        lv_ov = sorted(lv_ov, key=lambda x: model_spec.lv_order[x])
        C = data[lv_ov].cov()
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
        self.fit_function = LikelihoodObjective(cov(data))
    
    def func(self, theta):
        Sigma = self.implied_cov(theta)
        f = self.fit_function.function(Sigma)
        return f
    
    def gradient(self, theta):
        Sigma = self.implied_cov(theta)
        dSigma = self.dsigma(theta, free=False)
        g = self.fit_function.gradient(Sigma, dSigma)
        return g

    def hessian(self, theta):
        Sigma = self.implied_cov(theta)
        dSigma = self.dsigma(theta, free=False)
        d2Sigma = self.d2sigma(theta, free=False)
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
        
            
            
    
    
    
    