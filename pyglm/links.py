#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 21:21:01 2020

@author: lukepinkel
"""
import numpy as np
import scipy as sp
        
class Link(object):
    def inv_link(self, eta):
        raise NotImplementedError
    
    def dinv_link(self, eta):
        raise NotImplementedError
        
    def d2inv_link(self, eta):
        raise NotImplementedError
    
    def link(self, mu):
        raise NotImplementedError
    
    def dlink(self, mu):
        raise NotImplementedError
        

class IdentityLink(Link):

    def __init__(self):
        self.fnc='identity'
        
    def inv_link(self, eta):
        mu = eta
        return mu
    
    def dinv_link(self, eta):
        dmu = 0.0*eta+1.0
        return dmu
        
    def d2inv_link(self, eta):
        d2mu = 0.0*eta
        return d2mu
    
    def link(self, mu):
        return mu
    
    def dlink(self, mu):
        return 1


class LogitLink(Link):

    def __init__(self):
        self.fnc='logit'
        
    def inv_link(self, eta):
        u = np.exp(eta)
        mu = u / (u + 1)
        return mu
    
    def dinv_link(self, eta):
        u = np.exp(eta)
        dmu = u / ((1 + u)**2)
        return dmu
        
    def d2inv_link(self, eta):
        u = np.exp(eta)
        d2mu = -(u * (u - 1.0)) / ((1.0 + u)**3)
        return d2mu
    
    def link(self, mu):
        eta = np.log(mu / (1 - mu))
        return eta
    
    def dlink(self, mu):
        dmu = 1 / (mu * (1 - mu))
        return dmu
        
        

class ProbitLink(Link):
    
    def __init__(self):
        self.fnc='probit'
        
    def inv_link(self, eta):
        mu = sp.stats.norm.cdf(eta, loc=0, scale=1)
        mu[mu==1.0]-=1e-16
        return mu
    
    def dinv_link(self, eta):
        dmu = sp.stats.norm.pdf(eta, loc=0, scale=1)
        return dmu
        
    def d2inv_link(self, eta):
        d2mu = -eta * sp.stats.norm.pdf(eta, loc=0, scale=1)
        return d2mu
    
    def link(self, mu):
        eta = sp.stats.norm.ppf(mu, loc=0, scale=1)
        return eta
    
    def dlink(self, mu):
        dmu = 1.0 / (self.dinv_link(self.link(mu)))
        return dmu
        
        

class LogLink(Link):
    def __init__(self):
        self.fnc='log'
        
    def inv_link(self, eta):
        mu = np.exp(eta)
        return mu
    
    def dinv_link(self, eta):
        dmu = np.exp(eta)
        return dmu
        
    def d2inv_link(self, eta):
        d2mu = np.exp(eta)
        return d2mu
    
    def link(self, mu):
        eta = np.log(mu)
        return eta
    
    def dlink(self, mu):
        dmu = 1.0 / (self.dinv_link(self.link(mu)))
        return dmu
    
    
class ReciprocalLink(Link):
    
    
    def __init__(self):
        self.fnc='reciprocal'
    
    def inv_link(self, eta):
        mu = 1 / (eta)
        return mu
    
    def dinv_link(self, eta):
        dmu = -1 / (eta**2)
        return dmu
    
    def d2inv_link(self, eta):
        d2mu = 2 / (eta**3)
        return d2mu
    
    def link(self, mu):
        eta  = 1 / mu
        return eta
    
    def dlink(self, mu):
        dmu = 1.0 / (self.dinv_link(self.link(mu)))
        return dmu


class CloglogLink(Link):
    
    def __init__(self):
        self.fnc='cloglog'
        
    def inv_link(self, eta):
        mu = 1.0-np.exp(-np.exp(eta))
        return mu
    
    def dinv_link(self, eta):
        dmu = np.exp(eta-np.exp(eta))
        return dmu
    def d2inv_link(self, eta):
        d2mu = -np.exp(eta - np.exp(eta)) * (np.exp(eta) - 1.0)
        return d2mu
    
    def link(self, mu):
        eta = np.log(np.log(1 / (1 - mu)))
        return eta
    
    def dlink(self, mu):
        dmu = 1.0 / (self.dinv_link(self.link(mu)))
        return dmu
    
    

class PowerLink(Link):
    
    def __init__(self, alpha):
        self.fnc = 'power'
        self.alpha=alpha
        
    def inv_link(self, eta):
        if self.alpha==0:
            mu = np.exp(eta)
        else:
            mu = eta**(1/self.alpha)
        return mu
    
    def dinv_link(self, eta):
        if self.alpha==0:
            dmu = np.exp(eta)
        else:
            dmu = eta**(1/self.alpha - 1.0) / self.alpha
        return dmu
    
    def d2inv_link(self, eta):
        alpha=self.alpha
        if alpha==0:
            d2mu = np.exp(eta)
        else:
            d2mu = (alpha - 1.0) * eta**(1/alpha - 2.0) / alpha**2
        return d2mu
    
    def link(self, mu):
        if self.alpha==0:
            eta = np.log(mu)
        else:
            eta = mu**(self.alpha)
        return eta
    
    def dlink(self, mu):
        dmu = 1.0 / (self.dinv_link(self.link(mu)))
        return dmu
    
 
class LogComplementLink(Link):
    
    def __init__(self):
        self.fnc = 'logcomp'
        
    def inv_link(self, eta):
        mu = 1.0 - np.exp(eta)
        return mu
    
    def dinv_link(self, eta):
        dmu = -np.exp(eta)
        return dmu
    
    def d2inv_link(self, eta):
        d2mu = -np.exp(eta)
        return d2mu
    
    def link(self, mu):
        eta = np.log(1 - mu)
        return eta
    
    def dlink(self, mu):
        dmu = -1.0 / (1.0 - mu)
        return dmu
    
    
class NegativeBinomialLink(Link):
    
    def __init__(self, scale=1.0):
        self.fnc = 'negbin'
        self.k = scale
        self.v = 1.0 / scale
        
    def inv_link(self, eta):
        u = np.exp(eta)
        mu = u / (self.k * (1 - u))
        return mu
    
    def dinv_link(self, eta):
        u = np.exp(eta)
        dmu = u / (self.k * (1 - u)**2)
        return dmu
    
    def d2inv_link(self, eta):
        u = np.exp(eta)
        num = u * (u + 1)
        den = self.k *(u - 1)**3
        d2mu = -num / den
        return d2mu
    
    def link(self, mu):
        eta = np.log(mu / (mu + self.v))
        return eta
    
    def dlink(self, mu):
        dmu = 1.0 / (mu + self.k * mu**2)
        return dmu
    
    
    