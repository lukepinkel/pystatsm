#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 14:09:39 2022

@author: lukepinkel
"""


import numpy as np
from pystatsm.pylvcorr.sim_lvcorr import LVCorrSim
from pystatsm.pylvcorr.lvcorr import Polychoric, Polyserial
from pystatsm.pystatsm.utilities.numerical_derivs import fo_fc_cd, so_gc_cd

def test_polychoric():
    rng = np.random.default_rng(1234)
    
    R = np.array([[1.0, 0.5],
                  [0.5, 1.0]])
    
    lv_sim = LVCorrSim(corr_mat=R, x_bins=5, y_bins=3, rng=rng)
    x, y = lv_sim.simulate(1000)
    
    polychor_model = Polychoric(x=x, y=y)
    polychor_model.fit()
    
    grad = lambda r: np.atleast_1d(polychor_model.gradient(r))
    hess = lambda r: np.atleast_2d(polychor_model.hessian(r))
    x0 = np.atleast_1d(polychor_model.rho_hat)
    x1 = np.array([0.2])
    assert(np.allclose(fo_fc_cd(polychor_model.loglike, x0), grad(x0), atol=1e-6, rtol=1e-4))
    assert(np.allclose(fo_fc_cd(polychor_model.loglike, x1), grad(x1), atol=1e-6, rtol=1e-4))
    assert(np.allclose(so_gc_cd(grad, x0), hess(x0), atol=1e-6, rtol=1e-4))
    assert(np.allclose(so_gc_cd(grad, x1), hess(x1), atol=1e-6, rtol=1e-4))
    assert(polychor_model.optimizer.success)

def test_polyserial():
    rng = np.random.default_rng(1234)
    
    R = np.array([[1.0, 0.5],
                  [0.5, 1.0]])
    
    lv_sim = LVCorrSim(corr_mat=R, x_bins=5, y_bins=False, rng=rng)
    x, y = lv_sim.simulate(1000)
    
    polyser_model = Polyserial(x=x, y=y)
    polyser_model.fit()
    
    grad = lambda r: np.atleast_1d(polyser_model.gradient(r))
    hess = lambda r: np.atleast_2d(polyser_model.hessian(r))
    x0 = np.atleast_1d(polyser_model.rho_hat)
    x1 = np.array([0.2])
    assert(np.allclose(fo_fc_cd(polyser_model.loglike, x0), grad(x0), atol=1e-6, rtol=1e-4))
    assert(np.allclose(fo_fc_cd(polyser_model.loglike, x1), grad(x1), atol=1e-6, rtol=1e-4))
    assert(np.allclose(so_gc_cd(grad, x0), hess(x0), atol=1e-6, rtol=1e-4))
    assert(np.allclose(so_gc_cd(grad, x1), hess(x1), atol=1e-6, rtol=1e-4))
    assert(polyser_model.optimizer.success)







