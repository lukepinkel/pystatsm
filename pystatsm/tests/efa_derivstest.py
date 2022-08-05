#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 21:21:09 2022

@author: lukepinkel
"""

import tqdm
import numpy as np
import pandas as pd
from pystatsm.pyfa.efa import FactorAnalysis 
from pystatsm.pyfa.simulate_factor_model import FactorModelSim 
from pystatsm.utilities.linalg_operations import vec
from pystatsm.utilities.func_utils import handle_default_kws
from pystatsm.utilities.numerical_derivs import jac_approx, fo_fc_cd, so_gc_cd

rng = np.random.default_rng(123)


class FactorModelTest(object):
    
    def __init__(self, model_sim_kws=None, cov_sim_kws=None, data_sim_kws=None,
                 factor_model_kws=None):
        
        model_sim_kws = handle_default_kws(model_sim_kws, dict(n_vars=12, n_facs=3))
        cov_sim_kws = handle_default_kws(cov_sim_kws, {})
        data_sim_kws = handle_default_kws(data_sim_kws, dict(exact=False))
        factor_model_kws = handle_default_kws(factor_model_kws, dict(n_factors=3))

        model_sim = FactorModelSim(**model_sim_kws)
        model_sim.simulate_cov(**cov_sim_kws)
        Z, X = model_sim.simulate_data(**data_sim_kws)
        
        model = FactorAnalysis(X, **factor_model_kws)
        model.fit()
        
        self.model_sim, self.model = model_sim, model
        self.Z, self.X = Z, X
        self.tests = {}
        self._test_data = {}
     
    def _reshape_dsigma(self, D):
        p, k = self.model.p, np.prod(D.shape[:2])
        D = D.reshape(k, -1, order='F')
        J = np.tril(np.ones((p, p)))
        D = D[vec(J)==1]
        return D
    
    def _test_d2sigma(self):
        model = self.model
        x2 = model.params.copy() + 0.01
        #global counter
        #counter = 0
        pbar = tqdm.tqdm(total=len(x2)**2*model.p**2, smoothing=1e-6)
        def func(params):
            #global counter
            #counter = counter + 1
            pbar.update(1)
            return model.dsigma_params(params)
        self._test_deriv(jac_approx, func, model.d2sigma_params, x2, "d2sigma")
        pbar.close()

        
    def _test_deriv(self, fdiff, func, deriv, x, name, diff_reshape=None):
        D_analytic = deriv(x)
        D_numerical = fdiff(func, x)
        if diff_reshape is not None:
            D_numerical = diff_reshape(D_numerical)
        all_close =  np.allclose(D_analytic, D_numerical, atol=1e-5, rtol=1e-5)
        max_diff = np.abs(D_analytic-D_numerical).max()
        self.tests[name] = all_close, max_diff
        self._test_data[name] = dict(analytic=D_analytic, numerical=D_numerical)
        
    
    def test_derivs(self, test_d2sigma=False):
        model = self.model
        x1 = np.log(model.psi.copy()) + 0.05
        x2 = model.params.copy() + 0.01
        
        self._test_deriv(fo_fc_cd, model.loglike_exp, model.gradient_exp, x1, "grad")
        self._test_deriv(fo_fc_cd, model.loglike_params, model.gradient_params, x2, "grad_par")
        self._test_deriv(so_gc_cd, model.gradient_exp, model.hessian_exp, x1, "hess")
        self._test_deriv(so_gc_cd, model.gradient_params, model.hessian_params, x2, "hess_par")
        self._test_deriv(jac_approx, model.implied_cov_params, model.dsigma_params, x2, "dsigma", self._reshape_dsigma)
        if test_d2sigma:
            self._test_d2sigma()
           
def test_efaderivs():
    rotation_methods = ["varimax", "quartimax", "equamax", "parsimax"]
    oblique_tests = {}
    ortho_tests = {}
    oblique_tests_summary = {}
    ortho_tests_summary = {}
    
    for method in rotation_methods:
        factor_model_kws = dict(rotation_type="oblique", rotation_method=method)
        oblique_tests[method] = FactorModelTest(factor_model_kws=factor_model_kws)
        oblique_tests[method].test_derivs()
        
        factor_model_kws = dict(rotation_type="ortho", rotation_method=method)
        ortho_tests[method] = FactorModelTest(factor_model_kws=factor_model_kws)    
        ortho_tests[method].test_derivs()
    
        oblique_tests_summary[method] = oblique_tests[method].tests    
        ortho_tests_summary[method] = ortho_tests[method].tests   
        print(method)
    
    oblique_tests_summary = pd.DataFrame(oblique_tests_summary)
    ortho_tests_summary = pd.DataFrame(ortho_tests_summary)
    
    assert(oblique_tests_summary.all().all())
    assert(ortho_tests_summary.all().all())