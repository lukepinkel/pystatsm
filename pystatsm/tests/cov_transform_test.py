#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  5 23:51:27 2022

@author: lukepinkel
"""


import numpy as np
from pystatsm.utilities.numerical_derivs import jac_approx
from pystatsm.utilities.cov_transform import CholeskyCov, _vech
from pystatsm.utilities.random import r_lkj

def test_cov_transform():
    rng = np.random.default_rng(123)
    mat_size = 5
    R = r_lkj(eta=1.0, n=1, dim=mat_size, rng=rng)[0, 0]
    V = np.diag(rng.uniform(low=2.0, high=6.0, size=mat_size)**0.5)
    S = V.dot(R).dot(V)
    
    x1 = _vech(S)
    
    t = CholeskyCov(mat_size)
    x = x1.copy()
    u = t._fwd(x)
    
    assert(np.allclose(t._rvs(t._fwd(x)), x))
    assert(np.allclose(t._fwd(t._rvs(u)), u))
    
    du_dx_nm = jac_approx(t._fwd, x)
    du_dx_an = t._jac_fwd(x)
    assert(np.allclose(du_dx_nm, du_dx_an))
    
    
    dx_du_nm = jac_approx(t._rvs, u)
    dx_du_an = t._jac_rvs(u)
    
    assert(np.allclose(dx_du_nm, dx_du_an))
    
    
    d2u_dx2_nm = jac_approx(t._jac_fwd, x)
    d2u_dx2_an = t._hess_fwd(x)
    
    assert(np.allclose(d2u_dx2_nm, d2u_dx2_an))


