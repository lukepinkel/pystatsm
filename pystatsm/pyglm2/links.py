#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 21:21:01 2020

@author: lukepinkel
"""
import numpy as np
import scipy as sp
from abc import ABCMeta, abstractmethod

npdf_const = np.sqrt(2*np.pi)

class Link(metaclass=ABCMeta):
    
    @staticmethod
    @abstractmethod
    def inv_link(eta):
        pass
    
    @staticmethod
    @abstractmethod
    def dinv_link(eta):
        pass
        
    @staticmethod
    @abstractmethod
    def d2inv_link(eta):
        pass
    
    @staticmethod
    @abstractmethod
    def link(mu):
        pass
    
    @staticmethod
    @abstractmethod
    def dlink(mu):
        pass
    
    
    @staticmethod
    @abstractmethod
    def d2link(mu):
        pass
    
    
    @staticmethod
    @abstractmethod    
    def d3link(mu):
        pass
    
    @staticmethod
    @abstractmethod
    def d4link(self, mu):
        pass
    
    

class IdentityLink(Link):

    def __init__(self):
        self.fnc='identity'
        
    @staticmethod
    def inv_link(eta):
        mu = eta
        return mu
    
    @staticmethod
    def dinv_link(self, eta):
        dmu = 0.0*eta+1.0
        return dmu
       
    @staticmethod
    def d2inv_link(eta):
        d2mu = 0.0*eta
        return d2mu
    
    @staticmethod
    def link(mu):
        return mu
    
    
    @staticmethod
    def dlink(mu):
        return np.ones_like(mu)
    
    @staticmethod
    def d2link(mu):
        return np.zeros_like(mu)
    
    @staticmethod
    def d3link(mu):
        return np.zeros_like(mu)
    
    
    @staticmethod
    def d4link(mu):
        return np.zeros_like(mu)


class LogitLink(Link):

    def __init__(self):
        self.fnc='logit'
    
    @staticmethod
    def inv_link(eta):
        u = np.exp(eta)
        mu = u / (u + 1)
        return mu
    
    @staticmethod
    def dinv_link(eta):
        u = np.exp(eta)
        dmu = u / ((1 + u)**2)
        return dmu
       
    @staticmethod
    def d2inv_link(eta):
        u = np.exp(eta)
        d2mu = -(u * (u - 1.0)) / ((1.0 + u)**3)
        return d2mu
    
    @staticmethod
    def link(mu):
        eta = np.log(mu / (1 - mu))
        return eta
    
    @staticmethod
    def dlink(mu):
        dmu = 1 / (mu * (1 - mu))
        return dmu
    
    
    @staticmethod
    def d2link(mu):
        d2mu = (2.0 * mu - 1.0) / (mu**2 * (1 - mu)**2)
        return d2mu
    
    
    @staticmethod
    def d3link(mu):
        a = -6.0 * mu**2 + 6.0 * mu - 2.0
        b = (mu - 1.0)**3 * mu**3
        return a / b
    
    @staticmethod
    def d4link(mu):
        a = 6.0 * (4.0 * mu**3 - 6.0 * mu**2 + 4.0 * mu - 1.0)
        b = (1.0 - mu)**4 * mu**4
        return a / b
        

class ProbitLink(Link):
    
    def __init__(self):
        self.fnc='probit'
        
        
    @staticmethod
    def inv_link(eta):
        mu = sp.special.ndtr(eta)
        mu[mu==1.0]-=1e-16
        return mu
    
    @staticmethod
    def dinv_link(eta):
        dmu = np.exp(-eta**2 / 2.0) / npdf_const
        return dmu
        
    @staticmethod
    def d2inv_link(eta):
        d2mu = -eta *  np.exp(-eta**2 / 2.0) / npdf_const
        return d2mu
    
    @staticmethod
    def link(mu):
        eta = sp.special.ndtri(mu)
        return eta
    
    @staticmethod
    def dlink(mu):
        dmu = np.exp(sp.special.erfinv(2.0 * mu - 1)**2) * npdf_const
        return dmu
    
    @staticmethod
    def d2link(mu):
        c = 2.0 * np.sqrt(2.0) * np.pi
        u = sp.special.erfinv(2.0 * mu - 1.0)
        g = c * np.exp(2*u**2) * u
        return g
    
    @staticmethod
    def d3link(mu):
        u = sp.special.erfinv(2.0 * mu - 1.0)**2
        c = 2.0 * np.sqrt(2.0) * np.pi**(3.0/2.0)
        g = c * np.exp(3.0 * u) * (4.0 * u + 1.0)
        return g
    
    @staticmethod
    def d4link(mu):
        u = sp.special.erfinc(2.0 * mu - 1.0)
        c = 4.0 * np.sqrt(2.0) * np.pi**2
        g = c * np.exp(4.0 * u**2) * u * (12 * u**2 + 7)
        return g
        
        

class LogLink(Link):
    def __init__(self):
        self.fnc='log'
      
    @staticmethod
    def inv_link(eta):
        mu = np.exp(eta)
        return mu
    
    @staticmethod
    def dinv_link(eta):
        dmu = np.exp(eta)
        return dmu
       
    @staticmethod
    def d2inv_link(eta):
        d2mu = np.exp(eta)
        return d2mu
    
    @staticmethod
    def link(mu):
        eta = np.log(mu)
        return eta
    
    @staticmethod
    def dlink(mu):
        dmu = 1.0 / mu
        return dmu
    
    @staticmethod
    def d2link(mu):
        return -1.0 / mu**2
    
    @staticmethod
    def d3link(mu):
        return 2.0 / (mu**3)
    
    @staticmethod
    def d4link(mu):
        return -6.0 / mu**4
    
    
class ReciprocalLink(Link):
    
    
    def __init__(self):
        self.fnc='reciprocal'
    
    @staticmethod
    def inv_link(eta):
        mu = 1 / eta
        return mu
    
    @staticmethod
    def dinv_link(eta):
        dmu = -1 / (eta**2)
        return dmu
    
    @staticmethod
    def d2inv_link(eta):
        d2mu = 2 / (eta**3)
        return d2mu
    
    @staticmethod
    def link(mu):
        eta  = 1 / mu
        return eta
    
    @staticmethod
    def dlink(mu):
        dmu = -1.0 / mu**2
        return dmu
    
    @staticmethod
    def d2link(mu):
        return 2.0 / mu**3
    
    @staticmethod
    def d3link(mu):
        return -6.0 / (mu**4)
    
    @staticmethod
    def d4link(mu):
        return 24 / mu**5


class CloglogLink(Link):
    
    def __init__(self):
        self.fnc='cloglog'
      
    @staticmethod
    def inv_link(eta):
        mu = 1.0-np.exp(-np.exp(eta))
        return mu
    
    @staticmethod
    def dinv_link(eta):
        dmu = np.exp(eta-np.exp(eta))
        return dmu
    
    @staticmethod
    def d2inv_link(eta):
        d2mu = -np.exp(eta - np.exp(eta)) * (np.exp(eta) - 1.0)
        return d2mu
    
    @staticmethod
    def link(mu):
        eta = np.log(np.log(1 / (1 - mu)))
        return eta
    
    @staticmethod
    def dlink(mu):
        u = 1.0 - mu
        dmu = 1.0 / (u * np.log(1.0 / u))
        return dmu
    
    @staticmethod
    def d2link(mu):
        u = np.log(1.0 / (1.0 - mu))
        return (u - 1.0) / ((1.0 - mu)**2 * u**2)
    
    @staticmethod
    def d3link(mu):
        u = np.log(1 / (1.0 - mu))
        a = -2.0 * u**2 + 3 * u - 2.0
        b = (mu - 1.0)**3 * u**3
        return a / b
    
    @staticmethod
    def d4link(mu):
        u = np.log(1.0 / (1.0 - mu))
        a = 6.0 * u**3 - 11.0 * u**2 + 12.0 * u - 6.0
        b = (1.0 - mu)**4 * u**4
        return a / b
        
    
    

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
    
    def d2link(self, mu):
        if self.alpha==0:
            g = -1.0 / (mu**2)
        else:
            g = (self.alpha - 1.0) * self.alpha * mu**(self.alpha - 2.0)
        return g
    
    def d3link(self, mu):
        if self.alpha==0:
            g = 2.0 / (mu**3)
        else:
            a = self.alpha
            g = (a - 2.0) * (a - 1.0) * a * mu**(a - 3.0)
        return g
    
    def d4link(self, mu):
        if self.alpha==0:
            g = -6.0 / (mu**4)
        else:
            a = self.alpha
            g = (a - 3.0) * (a - 2.0) * (a - 1.0) * a * mu**(a - 4.0)
        return g
            
 
class LogComplementLink(Link):
    
    def __init__(self):
        self.fnc = 'logcomp'
    
    @staticmethod
    def inv_link(eta):
        mu = 1.0 - np.exp(eta)
        return mu
    
    @staticmethod
    def dinv_link(eta):
        dmu = -np.exp(eta)
        return dmu
    
    @staticmethod
    def d2inv_link(eta):
        d2mu = -np.exp(eta)
        return d2mu
    
    @staticmethod
    def link(mu):
        eta = np.log(1 - mu)
        return eta
    
    @staticmethod
    def dlink(mu):
        dmu = -1.0 / (1.0 - mu)
        return dmu
    
    @staticmethod
    def d2link(mu):
        return -1.0 / (1.0 - mu)**2
    
    @staticmethod
    def d3link(mu):
        return 2.0 / (mu - 1.0)**3
    
    @staticmethod
    def d4link(mu):
        return -6.0 / (1.0 - mu)**4
    
    
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
    
    def d2link(self, mu):
        a = self.v * (self.v + 2.0 * mu)
        b = mu**2 * (self.v + mu)**2
        return a / b
        
    def d3link(self, mu):
        v = self.v
        a = 2.0 * v * (v**2 + 3.0 * v * mu + 3.0 * mu**2)
        b = mu**3  * (v + mu)**3
        return a / b
    
    def d4link(self, mu):
        v = self.v
        a = 6.0 * v * (v**3 + 4.0 * v**2 * mu + 6.0 * v * mu**2 + 4.0 * mu**3)
        b = mu**4 * (mu + v)**4
        return a / b
        
    
    
    