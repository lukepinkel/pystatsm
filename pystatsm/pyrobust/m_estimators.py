#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 00:50:23 2020

@author: lukepinkel
"""

import numpy as np
import scipy as sp
from .dispersion_estimators import mad, gmd, sn_estimator, qn_estimator # analysis:ignore
from ..utilities.func_utils import norm_pdf
from ..utilities.linalg_operations import nwls

QTF75 = sp.stats.norm(0, 1).ppf(0.75)

class MEstimator(object):
    
    
    def _normal_ev(self, f, a=-np.inf, b=np.inf, *args, **kwargs):
        I, _ =sp.integrate.quad(f, a, b, *args, **kwargs)
        return I
        
    def E_rho(self, c=None):
        f = lambda x: self.rho(x, c) * norm_pdf(x)
        return self._normal_ev(f)
    
    def E_psi(self, c=None):
        f = lambda x: self.rho(x, c) * norm_pdf(x)
        return self._normal_ev(f)
        
    def E_phi(self, c=None):
        f = lambda x: self.phi(x, c) * norm_pdf(x)
        return self._normal_ev(f)
    
    def E_psi2(self, c=None):
        f = lambda x: self.psi(x, c)**2 * norm_pdf(x)
        return self._normal_ev(f)
    
    def chi(self, x, deriv=0, *args):
        if deriv == 0:
            y =self.rho(x, *args)
        elif deriv == 1:
            y = self.psi(x, *args)
        elif deriv == 2:
            y = self.phi(x, *args)
        
        if deriv > 0:
            I = self.rho_norm(*args)
            y /= I
        return y
    
    def m_estimate_scale(self, r, scale=None, bd=0.5, n_iters=1, tol=1e-12):
        scale = mad(r) if scale is None else scale
        for i in range(n_iters):
            ss = np.mean(self.rho(r / scale)) / bd
            s_new = np.sqrt(scale**2 * ss)
            conv = np.abs(s_new / scale - 1.0) < tol
            scale = s_new
            if conv:
                break
        return scale
    
    def irls_step(self, X, y,  beta, scale=None, bd=0.5, n_iters=50, tol=1e-12):
        r = y - X.dot(beta)
        scale = mad(r) if scale is None else scale
        for i in range(n_iters):
            ss = np.mean(self.rho(r / scale)) / bd
            scale = np.sqrt(scale**2 * ss)
            w = self.weights(r / scale)
            beta_new = nwls(X, y, w)
            conv = np.linalg.norm(beta - beta_new)**2 < tol * np.linalg.norm(beta)+tol
            r = y - X.dot(beta_new)
            beta = beta_new
            if conv:
                break
        return r, beta, scale
            
            
        

        
    
    

class Huber(MEstimator):
    
    def __init__(self, c=1.345):
        self.c = 1.345

        
    def rho(self, x, c=None):
        c = self.c if c is None else c
        x = np.asarray(x)
        abs_x = np.abs(x)
        ix = abs_x <= c
        y = np.zeros_like(x)
        y[ix] = x[ix]**2 / 2.0
        y[~ix] = c * (abs_x[~ix] - c / 2.0)
        return y
    
    def psi(self, x, c=None):
        c = self.c if c is None else c
        x = np.asarray(x)
        abs_x = np.abs(x)
        ix = abs_x <= c
        y = np.zeros_like(x)
        y[ix] = x[ix]
        y[~ix] = c *np.sign(x[~ix])
        return y
    
    def phi(self, x, c=None):
        c = self.c if c is None else c
        x = np.asarray(x)
        abs_x = np.abs(x)
        ix = abs_x <= c
        y = np.zeros_like(x)
        y[ix] = 1.0
        return y
    
    def weights(self, x, c=None):
        c = self.c if c is None else c
        x = np.asarray(x)
        abs_x = np.abs(x)
        ix = abs_x <= c
        y = np.zeros_like(x)
        y[ix] = 1.0
        y[~ix] = c / abs_x[~ix]
        return y
    
    def E_rho(self, c=None):
        c = self.c if c is None else c
        p = 1.0 - sp.special.ndtr(c)
        d = norm_pdf(c)
        return 0.5 - p + c * (d - c * p)
    
    def E_psi(self, c=None):
        c = self.c if c is None else c
        return 0.0
    
    def E_psi2(self, c=None):
        c = self.c if c is None else c
        p = sp.special.ndtr(c)
        d = norm_pdf(c)
        return 2.0 * (c**2 * (1.0 - p) + p - 0.5 - c * d)
    
    def E_phi(self, c=None):
        c = self.c if c is None else c
        return 2.0 * sp.special.ndtr(c) -  1.0
    
    def rho_norm(self, c=None):
        return 1.0
    
    
    
        

        
class Bisquare(MEstimator):
    
    def __init__(self, c=4.685):
        #1.548 for S
        self.c = c
    
    def rho(self, x, c=None):
        x = np.atleast_1d(x)
        c = self.c if c is None else c
        abs_x = np.abs(x)
        ix = abs_x <= c
        y = np.zeros_like(x)
        y[~ix] = 1.0
        t = x[ix] / c
        t = t**2
        y[ix] = t * (3.0 + t * (t - 3.0))
        y = y
        return y
        
    def psi(self, x, c=None):
        x = np.asarray(x)
        c = self.c if c is None else c
        abs_x = np.abs(x)
        ix = abs_x <= c
        y = np.zeros_like(x)
        a = x[ix] / c
        u = 1.0 - a**2
        y[ix] = x[ix] * u**2
        return y

    def phi(self, x, c=None):
        x = np.asarray(x)
        c = self.c if c is None else c
        abs_x = np.abs(x)
        ix = abs_x <= c
        y = np.zeros_like(x)
        u = x[ix] / c
        u *= u
        y[ix] = (1.0 - u) * (1.0 - 5.0 * u)
        return y
        
    def weights(self, x, c=None):
        x = np.asarray(x)
        c = self.c if c is None else c
        abs_x = np.abs(x)
        ix = abs_x <= c
        y = np.zeros_like(x)
        u = x[ix] / c
        y[ix] = ((1.0 - u) * (1.0 + u))**2
        return y
    
    
    def rho_norm(self, c=None):
        c = self.c if c is None else c
        return c**2 / 6.0




   
 
def estimate_simultaneous(x, func=Huber(), d=0.5, n_iters=200, tol=1e-6, 
                          rethist=False, dispersion_est=mad):   
    n = len(x)
   
    mu, sd = np.median(x, axis=0), dispersion_est(x)
    r = (x - mu) / sd
    f_prev = func.rho_func(r).sum()
    
    mu_vec = np.zeros(n_iters)
    sd_vec = np.ones(n_iters)
    fvals = np.zeros(n_iters)
    
    mu_vec[0] = mu
    sd_vec[0] = sd
    fvals[0] = f_prev
    
    for i in range(n_iters):
        w_mu = func.weights(r)
        w_sd =func.scale_weights(r)
        
        mu = np.sum((w_mu * x)) / np.sum(w_mu)
        sd = np.sqrt((sd**2) / (d * n) * np.sum(w_sd * r**2))
        mu_vec[i] = mu
        sd_vec[i] = sd
        r = (x - mu) / sd
        
        f_new = func.rho_func(r).sum()
        fvals[i] = f_new
        if np.abs(f_new - f_prev)<tol:
            break
    if rethist:
        return fvals[:i], mu_vec[:i], sd_vec[:i]
    else:
        return fvals[i], mu_vec[i], sd_vec[i]
            
        
        
        
'''
kws = dict(func=Huber(), n_iters=500, tol=1e-16, rethist=False)
dist = sp.stats.cauchy(loc=200, scale=200).rvs
cauchy_sim_huber = np.vstack([np.array(estimate_simultaneous(dist(100),**kws)) 
                              for i in range(1000)])
    
kws = dict(func=Bisquare(), n_iters=500, tol=1e-16, rethist=False)

cauchy_sim_bisquare = np.vstack([np.array(estimate_simultaneous(dist(100),**kws)) 
                              for i in range(1000)])

    
    
kws = dict(func=Hampel(), n_iters=500, tol=1e-16, rethist=False)

cauchy_sim_hamepl = np.vstack([np.array(estimate_simultaneous(dist(100),**kws)) 
                               for i in range(1000)])
'''






