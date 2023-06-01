#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 23:30:43 2023

@author: lukepinkel
"""

import numpy as np
from abc import ABCMeta, abstractmethod

from ..utilities import data_utils
from ..utilities.linalg_operations import ( _vec, _invec, _vech, _invech) #analysis:ignore


class CovarianceFitFunction(metaclass=ABCMeta):
    """
    Abstract base class for fit/object functions.

    Subclasses must implement the _function, _gradient, and _hessian methods.
    """
    def __init__(self, data=None):
        self.data = data


    def function(self, Sigma, data=None):
        data = self.data if data is None else data
        return self._function(Sigma, data)
    
    def gradient(self, Sigma, dSigma, data=None):
        data = self.data if data is None else data
        return self._gradient(Sigma, dSigma, data)


    def hessian(self, Sigma, dSigma, d2Sigma, data=None):
        data = self.data if data is None else data
        return self._hessian(Sigma, dSigma, d2Sigma, data)
    
    @staticmethod
    @abstractmethod
    def _function(self, Sigma, data):
        pass

    @staticmethod
    @abstractmethod
    def _gradient(self, Sigma, dSigma, data):
        pass
    
        
    @staticmethod
    @abstractmethod
    def _hessian(self, Sigma, dSigma, d2Sigma, data):
        pass
    
    
class LikelihoodObjective(CovarianceFitFunction):

    
    @staticmethod 
    def _function(Sigma, data):
        C = data
        lndS = np.linalg.slogdet(Sigma)[1]
        SinvC = np.linalg.solve(Sigma, C)
        trSinvC = np.trace(SinvC)
        f = lndS + trSinvC
        return f
    
    @staticmethod
    def _gradient(Sigma, dSigma, data):
        C = data
        R = C - Sigma
        Sinv = np.linalg.inv(Sigma)
        A = Sinv.dot(R).dot(Sinv)
        D1 = _invech(dSigma.T).T
        #Needs to be p(p+1)/2 x t 
        g = -np.einsum("ji,ijk->k", A, D1)
        return g
        
     
    @staticmethod
    def _hessian(Sigma, dSigma, d2Sigma, data):
        C = data
        R = C - Sigma
        Sinv = np.linalg.inv(Sigma)
        D1 = _invech(dSigma.T).T
        D2 = _invech(d2Sigma.T).T
        R = C - Sigma
        A = Sinv.dot(R).dot(Sinv)
        A1 = np.einsum("hi,ijk->hjk", A + Sinv/2, D1)
        A2 = np.einsum("hi,ijk->hjk", Sinv, D1)
        H_t1 = np.einsum("ji,ijkl->kl", A, D2) 
        H_t2 = np.einsum("ijk,jim->km", A1, A2, optimize=True)
        H = 2*H_t2 - H_t1
        return H  
    
    
        
    
    
    
    
    
    
    
