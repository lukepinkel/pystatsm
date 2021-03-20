#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 00:50:23 2020

@author: lukepinkel
"""

import numpy as np
import scipy as sp
from .dispersion_estimators import mad, gmd, sn_estimator, qn_estimator # analysis:ignore

QTF75 = sp.stats.norm(0, 1).ppf(0.75)


    
    

class Huber:
    
    def __init__(self, scale_estimator=mad):
        self.c0 = 1.345
        self.c1 = QTF75
        self._scale_estimator = scale_estimator
        
    def rho_func(self, u):
        '''
        Function to be minimized
        '''
        v = u.copy()
        ixa = np.abs(u) < self.c0
        ixb = ~ixa
        v[ixa] = u[ixa]**2
        v[ixb] = np.abs(2*u[ixb])*self.c0 - self.c0**2
        return v
    
    def psi_func(self, u):
        '''
        Derivative of rho
        '''
        v = u.copy()
        ixa = np.abs(u) < self.c0
        ixb = ~ixa
        v[ixb] = self.c0 * np.sign(u[ixb])
        return v
    
    def phi_func(self, u):
        '''
        Second derivative of rho
        '''
        v = u.copy()
        ixa = np.abs(u) <= self.c0
        ixb = ~ixa
        v[ixa] = 1
        v[ixb] = 0
        return v
    
    def weights(self, u):
        '''
        Equivelant to psi(u)/u
        '''
        v = u.copy()
        ixa = np.abs(u) < self.c0
        ixb = ~ixa
        v[ixa] = 1
        v[ixb] = self.c0 / np.abs(u[ixb])
        return v
    
    def estimate_scale(self, r, *args):
        return self._scale_estimator(r, *args)
    
    def scale_weights(self, u):
        w = np.ones_like(u)
        ixa = u!=0
        ixb = u==0
        w[ixa] = self.rho_func(u[ixa]) / u**2
        w[ixb] = self.phi_func(u[ixb])
        return w
        
class Bisquare:
    
    def __init__(self, scale_estimator=mad):
        self.c0 = 4.685
        self.c1 = 0.6745
        self._scale_estimator = scale_estimator
    
    def rho_func(self, u):
        '''
        Function to be minimized
        '''
        v = u.copy()
        c = self.c0
        ixa = np.abs(u) < c
        ixb = ~ixa
        v[ixa] = c**2 / 3 * (1 - ((1 - (u[ixa] / c)**2)**3))
        v[ixb] = 2 * c
        return v
    
    def psi_func(self, u):
        '''
        Derivative of rho
        '''
        v = u.copy()
        c = self.c0
        ixa = np.abs(u) <= c
        ixb = ~ixa
        v[ixa] = u[ixa] * (1 - (u[ixa] / c)**2)**2
        v[ixb] = 0
        return v
    
    def phi_func(self, u):
        '''
        Second derivative of rho
        '''
        v = u.copy()
        c = self.c0
        ixa = np.abs(u) <= self.c0
        ixb = ~ixa
        u2c2 = (u**2 / c**2)
        v[ixa] = (1 -u2c2[ixa]) * (1 - 5 * u2c2[ixa])
        v[ixb] = 0
        return v
    
    def weights(self, u):
        '''
        Equivelant to psi(u)/u
        '''
        v = u.copy()
        c = self.c0
        ixa = np.abs(u) < c
        ixb = ~ixa
        v[ixa] = (1 - (u[ixa] / c)**2)**2
        v[ixb] = 0
        return v
    
    def scale_weights(self, u):
        w = np.ones_like(u)
        ixa = u!=0
        ixb = u==0
        w[ixa] = self.rho_func(u[ixa]) / u**2
        w[ixb] = self.phi_func(u[ixb])
        return w
     
    def estimate_scale(self, r, *args):
        return self._scale_estimator(r, *args)
        
class Hampel:
    
    def __init__(self, k=0.9016085, scale_estimator=mad):
        self.a = 1.5 * k
        self.b = 3.5 * k
        self.r = 8.0 * k
        self.k = k
        self.c = self.a / 2.0 * (self.b - self.a + self.r)
        self.a2 = self.a**2
        self._scale_estimator = scale_estimator
    
    def rho_func(self, u):
        '''
        Function to be minimized
        '''
        a, a2, b, c, r = self.a, self.a2, self.b, self.c, self.r
        v = u.copy()
        au = np.abs(u)
        ixa = au <= a
        ixb = (au>a) * (au<=b)
        ixc = (au>b) * (au<=r)
        ixd = au>r
        v[ixa] = 0.5 * u[ixa]**2 / c
        v[ixb] = (0.5 * a2 + a*(au[ixb] - a)) / c
        v[ixc] = 0.5 * (2*b-a+(au[ixc]-b)*(1+(r-au[ixc])/(r-b))) / c
        v[ixd] = 1.0
        return v
    
    def psi_func(self, u):
        '''
        Derivative of rho
        '''
        v = u.copy()
        a, b, r = self.a, self.b, self.r
        au = np.abs(u)
        sgnu = np.sign(u)
        ixa = au <= self.a
        ixb = (au>a) * (au<=b)
        ixc = (au>b) * (au<=r)
        ixd = au>r
        v[ixa] = u[ixa]
        v[ixb] = a * sgnu[ixb]
        v[ixc] = a * sgnu[ixc] * (r - au[ixc]) / (r - b)
        v[ixd] = 0
        return v
    
    def phi_func(self, u):
        '''
        Second derivative of rho
        '''
        v = np.zeros(u.shape[0])
        a, b, r = self.a, self.b, self.r
        au = np.abs(u)
        ixa = au <= self.a
        ixc = (au>b) * (au<=r)
        v[ixa] = 1.0
        v[ixc] = (a * np.sign(u)[ixc] * u[ixc]) / (au[ixc] * (r - b))
        return v
    
    def scale_weights(self, u):
        w = np.ones_like(u)
        ixa = u!=0
        ixb = u==0
        w[ixa] = self.rho_func(u[ixa]) / u**2
        w[ixb] = self.phi_func(u[ixb])
        return w
    
    def weights(self, u):
        '''
        Equivelant to psi(u)/u
        '''
        v = np.zeros(u.shape[0])
        a, b, r = self.a, self.b, self.r
        au = np.abs(u)
        ixa = au <= self.a
        ixb = (au>a) * (au<=b)
        ixc = (au>b) * (au<=r)
        v[ixa] = 1.0
        v[ixb] = a / au[ixb]
        v[ixc] = a * (r - au[ixc]) / (au[ixc] * (r - b))
        return v
      
    def estimate_scale(self, r, *args):
        return self._scale_estimator(r, *args)
    




class Laplace:
    
    def __init__(self, scale_estimator=mad):
        self.a = 1.0
        self._scale_estimator = scale_estimator
     
    def rho_func(self, u):
        rho = np.abs(u)
        return rho

    def psi_func(self, u):
        psi = np.sign(u)
        return psi

    def phi_func(self, u):
        phi = np.ones_like(u)
        return phi

    def weights(self, u):
        w = self.psi_func(u) / u
        return w
       
    def estimate_scale(self, r):
        return self._scale_estimator(r)   
    
    def scale_weights(self, u):
        w = np.ones_like(u)
        ixa = u!=0
        ixb = u==0
        w[ixa] = self.rho_func(u[ixa]) / u**2
        w[ixb] = self.phi_func(u[ixb])
        return w
    
    
class Lpnorm:
    
    def __init__(self, p=1.5, scale_estimator=mad):
        self.p = p
        self.a = p - 1.0
        self.b = p / 2.0
        self.c = self.a * self.b
        self.d = p - 2.0
        self._scale_estimator = scale_estimator
     
    def rho_func(self, u):
        rho = 0.5 * np.abs(u)**self.p
        return rho

    def psi_func(self, u):
        psi = self.b * np.abs(u)**self.a
        psi*= np.sign(u)
        return psi

    def phi_func(self, u):
        phi = -np.abs(u)**self.d * self.c * np.sign(u)
        return phi

    def weights(self, u):
        w = self.psi_func(u) / u
        return w
          
    def estimate_scale(self, r, *args):
        return self._scale_estimator(r, *args)

    def scale_weights(self, u):
        w = np.ones_like(u)
        ixa = u!=0
        ixb = u==0
        w[ixa] = self.rho_func(u[ixa]) / u**2
        w[ixb] = self.phi_func(u[ixb])
        return w
    
    
   
 
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






