#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 18:55:34 2019

@author: lukepinkel
"""
import tqdm
import numpy as np
import pandas as pd
import scipy as sp
import scipy.optimize

from ..utilities.data_utils import _check_np, corr
from ..utilities.func_utils import handle_default_kws
from ..utilities.indexing_utils import tril_indices, inv_tril_indices
from .statfuncs import (norm_qtf, polychor_thresh,  binorm_pdf, 
                        binorm_cdf, polyex, norm_cdf, norm_pdf,
                        binorm_cdf2, binorm_pdf2, dbinorm_pdf2)

 

class _Polychor(object):
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
        self.rho_init = np.atleast_1d(np.corrcoef(self.x, self.y)[0, 1])
    
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
    
    def fit(self, opt_kws=None):
        default_opt_kws = dict(method='trust-constr')
        opt_kws = {} if opt_kws is None else opt_kws
        opt_kws = {**default_opt_kws, **opt_kws}
        bounds =[(-1.0+1e-16, 1.0-1e-16)]
        x0 = self.rho_init
        opt = sp.optimize.minimize(self.loglike, x0, jac=self.gradient,
                                   hess=self.hessian, bounds = bounds,
                                   **opt_kws)
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




class Polychoric(object):

    def __init__(self, data):
        X = data.values
        n, p = X.shape
        N_cats = np.array([len(np.unique(X[:, i])) for i in range(p)]).astype(int)
        Thresholds = {}
        for i in range(p):
            Thresholds[i] = self._get_threshold(X[:, i], outer=True)
        
        p_star = int(p * (p - 1) // 2)
        IJ_ind = np.array(tril_indices(p, -1)).T
        Xtab ={}
        XThresholds = {}
        rho_inits = {}
        for ii in range(p_star):
            i, j = IJ_ind[ii]
            XThresholds[ii], Xtab[ii] = self._init_crosstab(X[:,i],
                                                            X[:,j],
                                                            Thresholds[i],
                                                            Thresholds[j]
                                                            )
            rho_inits[ii] = corr(X[:,i], X[:, j])
            
            
        
        self.rho_inits = rho_inits
        self.rhos = np.zeros(p_star)
        self.rhos_se = np.zeros(p_star)
        self.opts = {}
        self.data, self.X, self.n, self.p = data, X, n, p
        self.N_cats, self.Thresholds = N_cats, Thresholds
        self.p_star = p_star
        self.IJ_ind = IJ_ind
        self.II_ind = inv_tril_indices(p, -1)
        self.XThresholds, self.Xtab = XThresholds, Xtab
        
    def _get_threshold(self, x, outer=True):
        vals, counts = np.unique(x, return_counts=True)
        t_inner = sp.special.ndtri(np.cumsum(counts)[:-1] / np.sum(counts))
        if outer:
            t = np.r_[-1e6, t_inner, 1e6]
        else:
            t = t
        return t
    
    def _init_crosstab(self, xi, xj, ti=None, tj=None):
        not_nan = ~(np.isnan(xi) | np.isnan(xj))
        xi, xj = xi[not_nan], xj[not_nan]
        ti = self._get_threshold(xi) if ti is None else ti
        tj = self._get_threshold(xj) if tj is None else tj
        _, xtab = sp.stats.contingency.crosstab(xi, xj)
        mi, mj = xtab.shape

        ui_ind, uj_ind = np.meshgrid(np.arange(1, mi+1), np.arange(1, mj+1), indexing="ij")
        li_ind, lj_ind = ui_ind-1, uj_ind-1
        
        tiu, tju = ti[ui_ind], tj[uj_ind]
        til, tjl = ti[li_ind], tj[lj_ind]
        
        Tau_ij = np.zeros((2, 2, mi, mj))
        Tau_ij[0, 0] = tiu
        Tau_ij[0, 1] = til
        Tau_ij[1, 0] = tju
        Tau_ij[1, 1] = tjl
        
        return Tau_ij, xtab
        
    def prob(self, r, i, j):
        Thresh = self.XThresholds[self.II_ind[i, j]]
        upper, lower = Thresh[:, 0], Thresh[:, 1]
        pr = binorm_cdf2(lower, upper, r)
        return pr
    
    def dprob(self, r, i, j):
        Thresh = self.XThresholds[self.II_ind[i, j]]
        upper, lower = Thresh[:, 0], Thresh[:, 1]
        p = binorm_pdf2(lower, upper, r)
        return p
     
    def d2prob(self, r, i, j):
        Thresh = self.XThresholds[self.II_ind[i, j]]
        upper, lower = Thresh[:, 0], Thresh[:, 1]
        dp = dbinorm_pdf2(lower, upper, r)
        return dp
    
    def loglike(self, r, i, j):
        pr = self.prob(r, i, j)
        pr = np.maximum(pr, 1e-16)
        Xcount = self.Xtab[self.II_ind[i, j]]
        ll = - np.sum(Xcount * np.log(pr))
        return ll
  
    def loglike_tanh(self, atanh_r, i, j):
        r = np.tanh(atanh_r)
        ll = self.loglike(r, i, j)
        return ll
    
    def gradient(self, r, i, j):
        pr = np.maximum(self.prob(r, i, j), 1e-16)
        dp = self.dprob(r, i, j)
        Xcount = self.Xtab[self.II_ind[i, j]]
        g = -np.sum(Xcount / pr * dp)
        return g
        
    def gradient_tanh(self, atanh_r, i, j):
        r = np.tanh(atanh_r)
        g = self.gradient(r, i, j)
        g = g * (1.0 - r**2)
        return g
    
    def hessian(self, r, i, j):
        pr = np.maximum(self.prob(r, i, j), 1e-16)
        dp = self.dprob(r, i, j)
        d2p = self.d2prob(r, i, j)
        Xcount = self.Xtab[self.II_ind[i, j]]
        
        u = Xcount / pr
        v = Xcount / np.maximum(pr**2, 1e-16)
        H = u * d2p - v * dp**2
        H = -np.sum(H)
        return H
    
    def hessian_tanh(self, atanh_r, i, j):
        r = np.tanh(atanh_r)
        g = self.gradient_tanh(atanh_r, i, j)
        H = self.hessian(r, i, j)
        H = 1 / np.cosh(atanh_r)**4 * H - 2.0 * r * g
        return H
    

    def _fit(self, i, j, opt_kws=None):
        opt_kws = handle_default_kws(opt_kws, dict(method='trust-constr'))
        func = lambda x: self.loglike_tanh(x, i, j)
        grad = lambda x: self.gradient_tanh(x, i, j)
        hess = lambda x: self.hessian_tanh(x, i, j)
        
        x0 = np.arctanh(self.rho_inits[self.II_ind[i, j]])
        opt = sp.optimize.minimize(func, x0, jac=grad, hess=hess, **opt_kws)
        
        r = np.tanh(opt.x)
        r_se = np.sqrt(1.0 / self.hessian(r, i, j))
        return r, r_se, opt

    
    def fit(self, verbose=False, opt_kws=None):
        if verbose:
            pbar = tqdm.tqdm(total=self.p_star, smoothing=1e-3)
        R = np.eye(self.p)
        R_se = np.zeros((self.p, self.p))
        for ii in range(self.p_star):
            i, j = self.IJ_ind[ii]
            self.rhos[ii], self.rhos_se[ii], self.opts[ii] = self._fit(i, j, opt_kws)
            R[i, j] = R[j, i] = self.rhos[ii]
            R_se[i, j] = R_se[j, i] = self.rhos_se[ii]
            if verbose:
                pbar.update(1)
        if verbose:
            pbar.close()
        self.R, self.R_se = R, R_se
        

class PolychoricCorr(object):

    def __init__(self, data):
        X = data.values
        n, p = X.shape
        xthresh= {i:{} for i in range(p)}
        xcounts = {i:{} for i in range(p)}
        rho_inits = {i:{} for i in range(p)}
        for i, j in np.ndindex(p, p):
            Xij = X[:, [i, j]]
            ind = ~np.isnan(Xij).any(axis=1)
            Xij = Xij[ind]
            xthresh[i][j], xcounts[i][j] = self._init_crosstab(Xij[:, 0], Xij[:, 1])
            rho_inits[i][j] = np.atleast_1d(np.corrcoef(Xij[:, 0], Xij[:, 1])[0, 1])
        
        self.data = data
        self.X = X
        self.nobs = self.n = n
        self.nvar = self.p = p
        self.xthresh = xthresh
        self.xcounts = xcounts
        self.rho_inits =rho_inits
        self.cats = {i:np.unique(X[:, i]) for i in range(self.p)}
        self.ncats = {i:len(self.cats[i]) for i in range(self.p)}
        self.taus = {i:self._get_threshold(X[:, i]) for i in range(self.p)}
        self.opts = {i:{} for i in range(self.p)}
        self.R = np.eye(self.p)
        self.R_se = np.zeros((self.p, self.p))
    
    def _get_threshold(self, x):
        vals, counts = np.unique(x, return_counts=True)
        t_inner = sp.special.ndtri(np.cumsum(counts)[:-1] / np.sum(counts))
        t = np.r_[-1e6, t_inner, 1e6]
        return t
    
    def _init_crosstab(self, x, y):
        _, xtab = sp.stats.contingency.crosstab(x, y)
        p, q = xtab.shape
        vecx = xtab.flatten()
        a = self._get_threshold(y)
        b = self._get_threshold(x)
        ixi, ixj = np.meshgrid(np.arange(1, q+1), np.arange(1, p+1))
        ixi1, ixj1 = ixi.flatten(), ixj.flatten()
        ixi2, ixj2 = ixi1 - 1, ixj1 - 1
        a1, a2 = a[ixi1], a[ixi2]
        b1, b2 = b[ixj1], b[ixj2]
        t = dict(a1=a1, a2=a2, b1=b1, b2=b2)
        return t, vecx
        
    
    def prob(self, r, i, j):
        xthresh = self.xthresh[i][j]
        a1, a2, b1, b2 = xthresh["a1"], xthresh["a2"], xthresh["b1"], xthresh["b2"]
        p =  binorm_cdf(a1, b1, r) \
            - binorm_cdf(a2, b1, r)\
            - binorm_cdf(a1, b2, r)\
            + binorm_cdf(a2, b2, r)
        return p
    
    def dprob(self, r, i, j):
        xthresh = self.xthresh[i][j]
        a1, a2, b1, b2 = xthresh["a1"], xthresh["a2"], xthresh["b1"], xthresh["b2"]
        p =   binorm_pdf(a1, b1, r) \
            - binorm_pdf(a2, b1, r)\
            - binorm_pdf(a1, b2, r)\
            + binorm_pdf(a2, b2, r)
        return p
    
    def _dphi(self, a, b, r):
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
     
    def gfunc(self, r, i, j):
        xthresh = self.xthresh[i][j]
        a1, a2, b1, b2 = xthresh["a1"], xthresh["a2"], xthresh["b1"], xthresh["b2"]
        g =  self._dphi(a1, b1, r)\
            -self._dphi(a2, b1, r)\
            -self._dphi(a1, b2, r)\
            +self._dphi(a2, b2, r)
        return g
    
    def loglike(self, r, i, j):
        p = self.prob(r, i, j)
        p = np.maximum(p, 1e-16)
        return -np.sum(self.xcounts[i][j] * np.log(p))
  
    def loglike_tanh(self, atanh_r, i, j):
        r = np.tanh(atanh_r)
        ll = self.loglike(r, i, j)
        return ll
    
    def gradient(self, r, i, j):
        p = self.prob(r, i, j)
        dp = self.dprob(r, i, j)
        p = np.maximum(p, 1e-16)
        ll = -np.sum(self.xcounts[i][j] / p * dp)
        return ll
    
        
    def gradient_tanh(self, atanh_r, i, j):
        r = np.tanh(atanh_r)
        g = self.gradient(r, i, j)
        g = g * (1.0 - r**2)
        return g
    
    def hessian(self, r, i, j):
        prb = np.maximum(self.prob(r, i, j), 1e-16)
        phi = self.dprob(r, i, j)
        gfn = self.gfunc(r, i, j)
        
        u = self.xcounts[i][j] / prb
        v = self.xcounts[i][j] / np.maximum(prb**2, 1e-16)
        H = u * gfn - v * phi**2
        H = -np.sum(H)
        return H
    
    def hessian_tanh(self, atanh_r, i, j):
        r = np.tanh(atanh_r)
        g = self.gradient_tanh(atanh_r, i, j)
        H = self.hessian(r, i, j)
        H = 1 / np.cosh(atanh_r)**4 * H - 2.0 * r * g
        return H
    

    def _fit(self, i, j, opt_kws=None):
        opt_kws = handle_default_kws(opt_kws, dict(method='trust-constr'))
        func = lambda x: self.loglike_tanh(x, i, j)
        grad = lambda x: self.gradient_tanh(x, i, j)
        hess = lambda x: self.hessian_tanh(x, i, j)
        
        x0 = np.arctanh(self.rho_inits[i][j])
        opt = sp.optimize.minimize(func, x0, jac=grad, hess=hess, **opt_kws)
        
        r = np.tanh(opt.x)
        r_se = np.sqrt(1.0 / self.hessian(r, i, j))
        self.R[i][j] = self.R[j][i] = r
        self.R_se[i][j] = self.R_se[j][i] = r_se
        self.opts[i][j] = self.opts[j][i] = opt
    
    def fit(self, verbose=False, opt_kws=None):
        inds = list(zip(*tril_indices(self.p, -1)))
        if verbose:
            pbar = tqdm.tqdm(total=len(inds), smoothing=1e-3)
        for i, j in inds:
            self._fit(i, j, opt_kws)
            if verbose:
                pbar.update(1)
        if verbose:
            pbar.close()
        
        

    
                

