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
    This class serves as the base class for classes that provide specific implementations of fit/object functions.
    Subclasses must implement the _function, _gradient, and _hessian methods.

    Parameters
    ----------
    data : (n, n) array_like, optional
        The data that the fit function operates on. If None, it should be supplied when calling the function, gradient or hessian methods.
    """

    def __init__(self, data=None, level=None, mean=False):
        self.data = data
        self.level = level
        self.mean = mean


    def function(self, Sigma, data=None, level=None, mean=None):
        """
        Calls the specific function implementation, using the stored data if not supplied.

        Parameters
        ----------
        Sigma : (n, n) array_like
            The covariance matrix.
        data : (n, n) array_like, optional
            The data to apply the function on. If None, the stored data is used.

        Returns
        -------
        float
            The result of the function.
        """
        data = self.data if data is None else data
        level = self.level if level is None else level
        mean = self.mean if mean is None else mean
        f = self._function(Sigma, data, level, mean)
        return f
    
    def gradient(self, Sigma, dSigma, data=None, level=None, mean=None):
        """
        Calls the specific gradient implementation, using the stored data if not supplied.

        Parameters
        ----------
        Sigma : (n, n) array_like
            The covariance matrix.
        dSigma : (n, n) array_like
            The change in the covariance matrix.
        data : (n, n) array_like, optional
            The data to apply the gradient on. If None, the stored data is used.

        Returns
        -------
        float
            The result of the gradient function.
        """
        data = self.data if data is None else data
        level = self.level if level is None else level
        mean = self.mean if mean is None else mean
        g = self._gradient(Sigma, dSigma, data, level, mean)
        return g


    def hessian(self, Sigma, dSigma, d2Sigma, data=None, level=None, mean=None):
        """
        Calls the specific hessian implementation, using the stored data if not supplied.

        Parameters
        ----------
        Sigma : (n, n) array_like
            The covariance matrix.
        dSigma : (n, n) array_like
            The change in the covariance matrix.
        d2Sigma : (n, n) array_like
            The second order change in the covariance matrix.
        data : (n, n) array_like, optional
            The data to apply the hessian on. If None, the stored data is used.

        Returns
        -------
        float
            The result of the hessian function.
        """
        data = self.data if data is None else data
        level = self.level if level is None else level
        mean = self.mean if mean is None else mean
        H = self._hessian(Sigma, dSigma, d2Sigma, data, level, mean)
        return H
    
    @staticmethod
    @abstractmethod
    def _function(self, Sigma, data, level, mean):
        pass

    @staticmethod
    @abstractmethod
    def _gradient(self, Sigma, dSigma, data, level, mean):
        pass
    
        
    @staticmethod
    @abstractmethod
    def _hessian(self, Sigma, dSigma, d2Sigma, data, level, mean):
        pass
    
    
class LikelihoodObjective(CovarianceFitFunction):
    """
    Implementation of CovarianceFitFunction providing likelihood objective specific function, gradient, and hessian computations.

    This class makes use of the logarithm of the determinant of the covariance matrix, the inverse of the covariance matrix, 
    and the trace of the product of the inverse covariance matrix and data matrix for its computations.
    """
    
    @staticmethod 
    def _function(Sigma, data, level, mean):
        """
        Function computation for the likelihood objective.

        Parameters
        ----------
        Sigma : (p, p) array_like
            The covariance matrix.
        data : (p, p) array_like
            The data matrix.

        Returns
        -------
        f: float
            The result of the function computation.
        """
        C = data.sample_cov
        lndS = np.linalg.slogdet(Sigma)[1]
        SinvC = np.linalg.solve(Sigma, C)
        trSinvC = np.trace(SinvC)
        f = lndS + trSinvC
        return f
    
    @staticmethod
    def _gradient(Sigma, dSigma, data, level, mean):
        """
        Gradient computation for the likelihood objective for covariance and
        Sigma of size (p, p) and dSigma of size (p*(p+1)/2, t) where t
        is the number of parameters Sigma is being differentiated wrt

        Parameters
        ----------
        Sigma : (p, p) array_like
            The covariance matrix.
        dSigma : (p*(p+1)/2, t) array_like
            d vech(Sigma)
        data : (p, p) array_like
            The data matrix.

        Returns
        -------
        g: (t,) array_like
            The result of the gradient computation of shape (t,).
        """
        C = data.sample_cov
        R = C - Sigma
        Sinv = np.linalg.inv(Sigma)
        A = Sinv.dot(R).dot(Sinv)
        D1 = _invech(dSigma.T).T
        #Needs to be p(p+1)/2 x t 
        g = -np.einsum("ji,ijk->k", A, D1)
        return g
        
     
    @staticmethod
    def _hessian(Sigma, dSigma, d2Sigma, data, level, mean):
        """
        Hessian computation for the likelihood objective.

        Parameters
        ----------
        Sigma : (p, p) array_like
            The covariance matrix.
        dSigma : (p*(p+1)/2, t) array_like
            d vech(Sigma)
        d2Sigma : (p*(p+1)/2, t, t) array_like
            d^2 vech(Sigma)
        data : (p, p) array_like
            The data matrix.

        Returns
        -------
        H: (t, t) array_like
            The (t,t) hessian
        """
        C = data.sample_cov
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
    
    
        
    
    
    
    
    
    
    
