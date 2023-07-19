#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 14:03:53 2022

@author: lukepinkel
"""

import numpy as np
import pandas as pd
from pystatsm.pyfa import rotation 
from pystatsm.pyfa.efa import FactorAnalysis 
from pystatsm.utilities.data_utils import _csd
from pystatsm.utilities import cov_utils
from pystatsm.utilities.numerical_derivs import jac_approx
rng = np.random.default_rng(123)


class RotationTest(object):
    
    def __init__(self, p, m, rotation_type, rotation_method, Phi=None):
        Phi = np.eye(m) if Phi is None else Phi
        A = self._make_loadings(p, m)
        psi = rng.uniform(low=0.2, high=0.8, size=(p))
        S = cov_utils.get_factor_cov(A, psi, Phi)
        X = _csd(rng.multivariate_normal(np.zeros(p), S, size=1000))
        model = FactorAnalysis(X, n_factors=m, rotation_type=rotation_type, 
                               rotation_method=rotation_method)
        model.fit()
        
        self.A_gen, self.Phi_gen, self.psi_gen = A, Phi, psi
        self.X, self.S = X, S
        self.r, self.model = model._rotate, model
        self.rotation_type, self.rotation_method = rotation_type, rotation_method
        self.A, self.T, self.Phi = model.A, model.T, model.Phi
        self.L = self.r.rotate(self.T)
        self.p, self.m = p, m
        self.tests = {}

    def _make_loadings(self, p, m):
        s = p//m
        A = np.zeros((p, m))
        A[np.arange(s*m), np.repeat(np.arange(m), s)] = rng.binomial(n=1, p=0.5, size=s*m)*2.0 - 1.0
        return A
    
    def _make_test_results(self, arr, arr_approx, name, rtol=1e-6, atol=1e-6):
        res = {f"{name}":arr,
               f"{name}_approx":arr_approx, 
               "Success":np.allclose(arr, arr_approx, rtol=rtol, atol=atol)
               }
        self.tests[name] = res
    
    def test_dQ(self):
        dQ = self.r.dQ(self.A)
        dQ_approx = jac_approx(self.r.Q, self.A)
        self._make_test_results(dQ, dQ_approx, "dQ")
    
    def test_dF(self):
        dF = self.r.df(np.eye(self.m))
        dF_approx = jac_approx(self.r.f, np.eye(self.m))
        self._make_test_results(dF, dF_approx, "dF")

    def test_drotate(self):
        dRotate = self.r.d_rotate(self.T)
        dRotate_approx = jac_approx(self.r.rotate, self.T)
        dRotate_approx = dRotate_approx.reshape(self.p*self.m, self.m*self.m, order='F')
        self._make_test_results(dRotate, dRotate_approx, "dRotate")
    
    def test(self):
        self._test()
        self.test_summary = {key:val["Success"] for key, val in self.tests.items()}
                
                

class ObliqueRotationTest(RotationTest):
    
    def __init__(self, p, m, rotation_method="varimax"):
        Phi = cov_utils.get_mdependent_corr([[0.2 * (-1)**i for i in range(m-1)]])
        super().__init__(p, m, rotation_type="oblique", rotation_method=rotation_method,
                         Phi=Phi)
    
    
    def Constraints_Phi(self, Phi):
        C = self.r.oblique_constraints(self.L, Phi)
        return C
    
    def Constraints_L(self, L):
        C = self.r.oblique_constraints(L, self.Phi)
        return C
    
    def test_dC_dL(self):
        dC_dL = self.r.dC_dL_Obl(self.L, self.Phi)
        dC_dL_approx = jac_approx(self.Constraints_L, self.L)
        self._make_test_results(dC_dL, dC_dL_approx, "dC_dL")
    
    def test_dC_dP(self):
        dC_dP = self.r.dC_dP_Obl(self.L, self.Phi)
        dC_dP_approx = jac_approx(self.Constraints_Phi, self.Phi)
        dC_dP_approx  = dC_dP_approx+np.swapaxes(dC_dP_approx, 2, 3)
        for j, k in np.ndindex(self.m, self.m):
            dC_dP_approx[j, k, np.arange(self.m), np.arange(self.m)] = 0.0
        self._make_test_results(dC_dP, dC_dP_approx, "dC_dP")
    
    def _test(self):
        self.test_dC_dP()
        self.test_dC_dL()
        self.test_dQ()
        self.test_dF()
        self.test_drotate()
        
class OrthoRotationTest(RotationTest):
    
    def __init__(self, p, m, rotation_method="varimax"):
        super().__init__(p, m, rotation_type="ortho", rotation_method=rotation_method)
      
    def Constraints_Phi(self, Phi):
        C = self.r.ortho_constraints(self.L, Phi)
        return C
    
    def Constraints_L(self, L):
        C = self.r.ortho_constraints(L, self.Phi)
        return C
        
    def L_A(self, A):
        consts = rotation.get_gcf_constants(self.rotation_method, self.p, self.m)
        r = rotation.GeneralizedCrawfordFerguson(A=A, rotation_method=self.rotation_method,
                                                 rotation_type="ortho")
        r.fit()
        L = r.rotate(r.T)
        return L
    
    def test_dC_dL(self):
        dC_dL = self.r.dC_dL_Ortho(self.L, self.Phi)
        dC_dL_approx = jac_approx(self.Constraints_L, self.L).reshape(self.m*self.m, self.m*self.p, order="F")
        self._make_test_results(dC_dL, dC_dL_approx, "dC_dL")
     
    def test_dL_dA(self):
        dL_dA = self.r.dL_dA_Ortho(self.L, self.Phi)
        dL_dA_approx = jac_approx(self.L_A, self.A)
        dL_dA_approx = dL_dA_approx.reshape(self.p*self.m, self.p*self.m, order='F')
        self._make_test_results(dL_dA, dL_dA_approx, "dL_dA", rtol=1e-2, atol=1e-2)
    
    def _test(self):
        self.test_dC_dL()
        self.test_dL_dA()
        self.test_dQ()
        self.test_dF()
        self.test_drotate()
        
  
def test_rotations():
    rotation_methods = ["varimax", "quartimax", "equamax", "parsimax"]
    oblique_tests = {}
    ortho_tests = {}
    oblique_tests_summary = {}
    ortho_tests_summary = {}
    
    for method in rotation_methods:
        oblique_tests[method] = ObliqueRotationTest(16, 4, rotation_method=method)
        oblique_tests[method].test()
        ortho_tests[method] = OrthoRotationTest(12, 3, rotation_method=method)    
        ortho_tests[method].test()
    
        oblique_tests_summary[method] = oblique_tests[method].test_summary    
        ortho_tests_summary[method] = ortho_tests[method].test_summary   
        
    
    oblique_tests_summary = pd.DataFrame(oblique_tests_summary)
    ortho_tests_summary = pd.DataFrame(ortho_tests_summary)
    
    assert(oblique_tests_summary.all().all())
    assert(ortho_tests_summary.all().all())
    
    
    
        
  
