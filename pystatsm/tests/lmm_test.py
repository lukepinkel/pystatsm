# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 20:59:40 2021

@author: lukepinkel
"""

import numpy as np
from pystatsm.pylmm.lmm import LMM
from pystatsm.pylmm.sim_lmm import MixedModelSim
from pystatsm.utilities.linalg_operations import invech
from pystatsm.utilities.numerical_derivs import fo_fc_cd, so_gc_cd



def test_lmm():
    rng = np.random.default_rng(123)
    n_grp, n_per = 100, 100
    formula = "y~1+x1+x2+x3+(1+x3|id1)"
    
    model_dict = {}
    model_dict["ginfo"] = dict(id1=dict(n_grp=n_grp, n_per=n_per))
    model_dict["beta"] = np.array([0.0, 0.5, -1.0, 1.0])
    model_dict["vcov"] = np.eye(3)
    model_dict["mu"] = np.zeros(3)
    model_dict["gcov"] = {"id1":invech(np.array([ 2.0, -1.0,  2.0]))}
    model_dict["n_obs"] = n_grp * n_per
    group_dict = dict(id1=np.repeat(np.arange(n_grp), n_per))
    rfe, rre = 0.4, 0.4
    msim = MixedModelSim(formula=formula, model_dict=model_dict, rng=rng, 
                         group_dict=group_dict, var_ratios=np.array([rfe, rre]))
    var_ratios = np.array([msim.v_fe, msim.v_re, msim.v_rs])
    var_ratios = var_ratios / np.sum(var_ratios)
    assert(np.allclose(var_ratios, np.array([0.4, 0.4, 0.2])))
    
    df = msim.df.copy()
    df["y"], u = msim.simulate_response()
    model = LMM(formula, data=df)
    model.fit(opt_kws=dict(xtol=1e-16, gtol=1e-12))
    assert(model.optimizer.success==True)
    assert((np.abs(model.optimizer.grad)<1e-2).all())
    
    theta = np.array([ 0.95418349, -0.36657794,  0.84136618,  1.12001821])
    assert(np.allclose(model.theta, theta))
    
    eps = np.finfo(float).eps**(1/4)
    grad_close = np.allclose(model.gradient(theta), fo_fc_cd(model.loglike, theta), atol=eps, rtol=eps)
    hess_close = np.allclose(model.hessian(theta), so_gc_cd(model.gradient, theta), atol=eps, rtol=eps)
    
    assert(grad_close)
    assert(hess_close)
    

