#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 18:55:34 2019

@author: lukepinkel
"""

import numpy as np
import pandas as pd
import scipy as sp
import scipy.optimize

from ..utilities.data_utils import _check_np, corr
from .statfuncs import (norm_qtf, polychor_thresh,  binorm_pdf, 
                        binorm_cdf, polyex, norm_cdf, norm_pdf)

 

class Polychoric:
    """
    Class that computes the MLE of a polychoric correlation
    
    Attributes
    ----------
    a : array
        Estimates of the thresholds for the first variable
        
        
    b : array
        Estimates of the threshold for the second variable
    
    ixi1 : array
        Indices used to construct a1
    
    ixi2 : array
        Indices used to construct a2
    
    ixj1 : array
        Indices used to construct b1
    
    ixj2 : array
        Indices used to construct b2
    
    a1 : array
        Index of the (i-1)th threshold for i in the number of categories in x
        and for j in the number of categories in y
    
    a2 : array
        Index of the i-th threshold for i in the number of categories in x
        and for j in the number of categories in y
    
    b1 : array
        Index of the (j-1)th threshold for i in the number of categories in x
        and for j in the number of categories in y    
     
    b2 : array
        Index of the j-th threshold for i in the number of categories in x
        and for j in the number of categories in y   
    
    x : array
        Ordinal variable
    
    y : array
        Ordinal variable
    
    p : int
        Number of categories in x
        
    q : int
        Number of categories in y
        
    """
    def __init__(self, x=None, y=None, df=None):
        if (x is None) and (y is None):
            df = _check_np(df)
            x, y= df[:, 0], df[:, 1]
        else:
            if type(x) in [float, int, str]:
                x = _check_np(df[x])
            else:
                x = _check_np(x)
                
                
            if type(y) in [float, int, str]:
                y = _check_np(df[y])
            else:
                y = _check_np(y)
        
        xtab = pd.crosstab(x, y).values
        p, q = xtab.shape
        vecx = xtab.flatten()
        a, b = polychor_thresh(xtab)
        
        ixi, ixj = np.meshgrid(np.arange(1, q+1), np.arange(1, p+1))
        ixi1, ixj1 = ixi.flatten(), ixj.flatten()
        ixi2, ixj2 = ixi1 - 1, ixj1 - 1
        
        self.a, self.b = a, b
        self.ixi1, self.ixi2 = ixi1, ixi2
        self.ixj1, self.ixj2 = ixj1, ixj2
        self.xtab, self.vecx = xtab, vecx
        self.a1, self.a2 = a[ixi1], a[ixi2]
        self.b1, self.b2 = b[ixj1], b[ixj2]
        self.x, self.p, self.y, self.q = x, p, y, q
    
    def prob(self, r):
        """
        Calculates P(n_{ij})
        
        Parameters
        ----------
        r : float
             Correlation
        
        """
        p = binorm_cdf(self.a1, self.b1, r) \
            - binorm_cdf(self.a2, self.b1, r)\
            - binorm_cdf(self.a1, self.b2, r)\
            + binorm_cdf(self.a2, self.b2, r)
        return p
    
    def dprob(self, r):
        """
        Calculates the derivative of P(n_{ij}) with respect to r
        
        Parameters
        ----------
        r : float
             Correlation
        
        """
        p = binorm_pdf(self.a1, self.b1, r) \
            - binorm_pdf(self.a2, self.b1, r)\
            - binorm_pdf(self.a1, self.b2, r)\
            + binorm_pdf(self.a2, self.b2, r)
        return p
    
    def _dphi(self, a, b, r):
        """
        Calculates the derivative of p(n_{ij}) with respect to r
        
        Parameters
        ----------
        r : float
             Correlation
        
        """
        xy, x2, y2 = a * b, a**2, b**2
        r2 = r**2
        s = (1 - r2)
        
        u1 = x2 / (2 * s)
        u2 = r*xy / s
        u3 = y2 / (2 * s)
        
        num1 = np.exp(-u1 + u2 - u3)
        num2 = r**3 - r2*xy + r*x2 + r*y2 - r - xy
        num = num1 * num2
        den = 2*np.pi*(r-1)*(r+1)*np.sqrt(s**3)
        g = num / den
        return g
     
    def gfunc(self, r):
        """
        Calculates the derivative of p(n_{ij}) with respect to r for all ij
        
        Parameters
        ----------
        r : float
             Correlation
        
        """
        g = self._dphi(self.a1, self.b1, r)\
            -self._dphi(self.a2, self.b1, r)\
            -self._dphi(self.a1, self.b2, r)\
            +self._dphi(self.a2, self.b2, r)
        return g
  
    def loglike(self, r):
        """
        Calculates negative log likelihood
        
        Parameters
        ----------
        r : float
             Correlation
        
        """
        p = self.prob(r)
        p = np.maximum(p, 1e-16)
        return -np.sum(self.vecx * np.log(p))
    
    def gradient(self, r):
        p = self.prob(r)
        dp = self.dprob(r)
        
        p = np.maximum(p, 1e-16)
        
        ll = np.sum(self.vecx / p * dp)
        return -ll
    
    def hessian(self, r):
        prb = np.maximum(self.prob(r), 1e-16)
        phi = self.dprob(r)
        gfn = self.gfunc(r)
        
        u = self.vecx / prb
        v = self.vecx / np.maximum(prb**2, 1e-16)
        
        H = u * gfn - v * phi**2
        return -np.sum(H)
    
    def fit(self, verbose=0, xtol=1e-12, gtol=1e-12):
        bounds =[(-1.0+1e-16, 1.0-1e-16)]
        x0 = np.atleast_1d(np.corrcoef(self.x, self.y)[0, 1])
        opt = sp.optimize.minimize(self.loglike, x0, jac=self.gradient,
                                   hess=self.hessian, bounds = bounds,
                                   options=dict(verbose=verbose),
                                   method='trust-constr')
        self.optimizer = opt
        self.rho_hat = opt.x[0]
        self.observed_information = self.hessian(self.rho_hat)
        self.se_rho = np.sqrt(1.0 / self.observed_information)




                

def dcrep(arr, dic):
    keys = np.array(list(dic.keys()))
    dicv = np.array(list(dic.values()))
    indx = keys.argsort()
    yv = dicv[indx[np.searchsorted(keys,arr.copy(),sorter=indx)]]
    return yv



class Polyserial:
    """
    Class that computes the MLE of a polyserial correlation
    
    Attributes
    ----------
    
    order : dict
        Dictionary specifying the order of the ordinal variable
        
    marginal_counts : array
        The marginal counts of the ordinal variable
        
    tau_arr : array
        Array of the estimated thresholds used to make the latent variable
        assumed to underly the ordinal variable, ordinal
        
    y_ordered : array
        Version of y converted into order integers
        
    tau1 : array
        Upper threshold
        
    tau2 : array
        Lower threshold
    
    x : array
        Continuous variable
        
    y : array
        Ordinal variable
    """
    def __init__(self, x=None, y=None, df=None):
        if (x is None) and (y is None):
            df = _check_np(df)
            x, y= df[:, 0], df[:, 1]
        else:
            if type(x) in [float, int, str]:
                x = _check_np(df[x])
            else:
                x = _check_np(x)
                
                
            if type(y) in [float, int, str]:
                y = _check_np(df[y])
            else:
                y = _check_np(y)
        
        order = dict(zip(np.unique(y), np.unique(y).argsort()))
        marginal_counts = np.array([np.sum(y==z) for z in np.unique(y)]).astype(float)
        tau_arr = norm_qtf(marginal_counts.cumsum()/marginal_counts.sum())
        tau_arr = np.concatenate([[-np.inf], tau_arr])
        tau_dict = dict(zip(list(order.values())+[list(order.values())[-1]+1], 
                            tau_arr.tolist()))
        y_ordered = dcrep(y, order)
        tau1, tau2 = dcrep(y_ordered, tau_dict),  dcrep(y_ordered+1, tau_dict)
        self.order, self.marginal_counts = order, marginal_counts
        self.tau_arr, self.y_ordered = tau_arr, y_ordered
        self.tau1, self.tau2 = tau1, tau2
        self.x, self.y = x, y
        
       
        
    def prob(self, r):
        """
        Calculates P(x|y)
        
        Parameters
        ----------
        r : float
             Correlation
        
        """
        tau1, tau2 = self.tau1, self.tau2
        th1 = polyex(self.x, tau1, r)
        th2 = polyex(self.x, tau2, r)
        p = norm_cdf(th2) - norm_cdf(th1)
        return p
    
    def loglike(self, r):
        """
        Returns the (negative) log likelihood
        
        Parameters
        ----------
        r : float
             Correlation
             
        Returns
        -------
        ll : float
            The log likelihood
        """
        ll = -np.sum(np.log(self.prob(r)))
        return ll
    
    def gradient(self, r):
        """
        Returns the derivative of the (negative) log likelihood with respect
        to the correlation
        
        Parameters
        ----------
        r : float
             Correlation
             
        Returns
        -------
        g : float
            The derivative of the negative log likelihood
        """
        tau1, tau2, x = self.tau1, self.tau2, self.x
        th1 = polyex(self.x, tau1, r)
        th2 = polyex(self.x, tau2, r)
    
        tmp1 = tau1.copy()
        tmp1[tmp1<-1e12] = 0.0
        
        tmp2 = tau2.copy()
        tmp2[tmp2>1e12] = 0.0
        u = norm_pdf(th2) * (tmp2 * r  - x)
        v = norm_pdf(th1) * (tmp1 * r  - x)
        
        
        p = self.prob(r)
        
        g = -1.0 /  (p * np.sqrt((1 - r**2)**3)) * (u - v)
        return g.sum()
    
    def hessian(self, r):
        """
        Returns an approximation of the second derivative of the (negative) 
        log likelihood with respectto the correlation.  Too lazy to 
        do the math at the moment, but a correct analytical derivative will
        be implemented in the future
        
        Parameters
        ----------
        r : float
             Correlation
             
        Returns
        -------
        H : float
            second derivative of the (negative) log likelihood
        """
        H = sp.optimize.approx_fprime(np.atleast_1d(r), 
                                      self.gradient, 
                                      np.finfo(1.0).eps**(1/3))
        return H
    
    
    def fit(self, verbose=0):
        """
        Fits the model
        
        Parameters
        ----------
        verbose : int
             The verbosity of the otpimizer
        """
        bounds =[(-1.0+1e-16, 1.0-1e-16)]
        x0 = np.atleast_1d(np.corrcoef(self.x, self.y)[0, 1])
        opt = sp.optimize.minimize(self.loglike, x0, jac=self.gradient,
                                   options=dict(verbose=verbose),
                                   method='trust-constr',
                                   bounds = bounds)
        self.optimizer = opt
        self.rho_hat = opt.x[0]
        self.observed_information = self.hessian(self.rho_hat)
        self.se_rho = np.sqrt(1.0 / self.observed_information)


        
        
        
        
    
                
        
    





