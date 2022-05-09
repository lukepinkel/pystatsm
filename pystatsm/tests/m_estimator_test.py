# -*- coding: utf-8 -*-
"""
Created on Mon May  9 15:25:35 2022

@author: lukepinkel
"""

import numpy as np
from pystatsm.pyrobust import m_estimators
from pystatsm.utilities import numerical_derivs
from pystatsm.utilities.func_utils import norm_pdf


def test_m_estimators():
    rng = np.random.default_rng(321)
    
    x = rng.standard_t(df=1, size=1000)
    
    
    
    M = m_estimators.Bisquare(c=1.54764)
    assert(np.allclose(M.E_rho(), 0.5))
    
    psi_approx = numerical_derivs.fd_derivative(M.chi, x, order=1)
    psi = M.chi(x, deriv=1)
    
    assert(np.allclose(psi_approx, psi))
    
    phi_approx = numerical_derivs.fd_derivative(lambda x: M.chi(x, deriv=1), x)
    phi = M.chi(x, deriv=2)
    
    assert(np.allclose(phi_approx, phi))
    
    
    M = m_estimators.Huber()
    
    f = lambda x: M.rho(x) * norm_pdf(x)
    I_approx = M._normal_ev(f, -np.inf, np.inf)
    assert(np.allclose(I_approx, M.E_rho()))
    
    f = lambda x: M.psi(x) * norm_pdf(x)
    I_approx = M._normal_ev(f, -np.inf, np.inf)
    assert(np.allclose(I_approx, M.E_psi()))
    
    
    f = lambda x: M.phi(x) * norm_pdf(x)
    I_approx = M._normal_ev(f, -np.inf, np.inf)
    assert(np.allclose(I_approx, M.E_phi()))
    
    
    f = lambda x: M.psi(x)**2 * norm_pdf(x)
    I_approx = M._normal_ev(f, -np.inf, np.inf)
    assert(np.allclose(I_approx, M.E_psi2()))
