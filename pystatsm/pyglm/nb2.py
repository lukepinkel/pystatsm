#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 23:09:53 2020

@author: lukepinkel
"""


import tqdm
import patsy
import pandas as pd
import numpy as np
import scipy as sp
import scipy.special
import scipy.stats
import scipy.optimize
from scipy.special import loggamma, digamma
from ..utilities.data_utils import _check_shape, _check_type

def trigamma(x):
    return sp.special.polygamma(1, x)      
    
    
    
class NegativeBinomial(object):
    
    def __init__(self, formula, data):
        Y, X = patsy.dmatrices(formula, data, return_type='dataframe')
        self.X, self.xcols, self.xix, self.x_is_pd = _check_type(X)
        self.Y, self.ycols, self.yix, self.y_is_pd = _check_type(Y)
        self.y = _check_shape(self.Y, 1)
        self.n_obs, self.n_feats = X.shape
        self.beta = np.zeros(self.n_feats)
        self.varp = np.log(np.ones(1)/2.0)
        self.params = np.concatenate([self.beta, self.varp])
        self.cnst = [(None, None) for i in range(self.n_feats)]+[(None, None)]
    
    def loglike(self, params, X=None):
        if X is None:
            X = self.X
        beta, kappa = params[:-1], np.exp(params[-1])
        mu = np.exp(X.dot(beta))
        a = 1.0 / kappa
        u = self.y + a
        v = kappa * mu
        ll = self.y * np.log(v) - u * np.log(1 + v) + loggamma(u) - loggamma(a) 
        ll = np.sum(ll)
        ll = ll - np.sum(loggamma(self.y + 1.0))
        return -ll

    def gradient(self, params, X=None):
        if X is None:
            X = self.X
        beta, kappa = params[:-1], np.exp(params[-1])
        mu = np.exp(X.dot(beta))
        w = 1.0 / ((mu + kappa * mu**2) * (1.0 / mu))
        gb = (X * w[:, np.newaxis]).T.dot(self.y - mu)
        u = kappa * (self.y - mu) / (1.0 + kappa * mu)
        gt = (u + np.log(1.0 + kappa * mu) - digamma(self.y + 1.0 / kappa) \
              +digamma(1.0 / kappa)) / kappa
        gt = np.sum(gt)
        g = np.concatenate([np.atleast_1d(gb), np.atleast_1d(gt)])
        return -g
    
    def hessian(self, params, X=None):
        if X is None:
            X = self.X
        beta, kappa = params[:-1], np.exp(params[-1])
        mu = np.exp(X.dot(beta))
        a = 1.0 / kappa
        u = 1 + kappa * mu
        r = self.y - mu
        
        wbb = mu * (1.0 + kappa * self.y) / (u**2)
        H11 = -(X.T * wbb).dot(X)
        H21 = -X.T.dot(kappa * r / (u**2 * 1.0 / mu))
        
        denom = -self.y * kappa * mu + mu + 2 * kappa * mu**2
        numer = u**2
        v = denom / numer
        
        H22 = v - a * np.log(u) + a * (digamma(self.y + a)-digamma(a)) +\
              a**2 * (trigamma(self.y+a) - trigamma(a))
        H22 = np.atleast_2d(np.sum(H22))
        H21 = np.atleast_2d(H21)
        H11 = np.atleast_2d(H11)
        H = np.block([[H11, H21.T], [H21, H22]])
        return -H  

    def deviance(self, params):
        X, y = self.X, _check_shape(self.Y, 1)
        b, a = params[:-1], np.exp(params[-1])
        mu = np.exp(X.dot(b))
        v = 1.0 / a
        ix = y>0
        dev1 = y[ix] * np.log(y[ix] / mu[ix])
        dev1 -= (y[ix]+v) * np.log((y[ix] + v) / (mu[ix] + v))
        
        dev2 = np.log(1 + a * mu[~ix]) / a
        dev = 2.0 * (np.sum(dev1) + np.sum(dev2))
        return dev
    
    def fit(self, opt_kws={}):
        default_opt_kws = dict(options=dict(verbose=0), 
                                        method='trust-constr')
                                        
        for key, val in default_opt_kws.items():
            if key not in opt_kws.keys():
                opt_kws[key] = val
        
        intercept = np.ones((self.X.shape[0], 1))
        b0 = np.zeros(2)
        self.opt_mean = sp.optimize.minimize(self.loglike, b0, jac=self.gradient,
                                        hess=self.hessian, args=(intercept,),
                                        **opt_kws)
        self.opt_full = sp.optimize.minimize(self.loglike, self.params, jac=self.gradient,
                                        hess=self.hessian, **opt_kws)
        self.params = self.opt_full.x
        self.se_params = np.sqrt(np.diag(np.linalg.inv(self.hessian(self.params))))
        self.ll_null = -self.opt_mean.fun
        self.ll_full = -self.opt_full.fun
        self.res = pd.DataFrame(np.vstack((self.params, self.se_params,
                                           self.params/self.se_params)).T,
                                columns=['param', 'SE', 't'])
        self.res.index = self.xcols.tolist() + ['variance']
        nu = self.X.shape[0]-2
        self.res['p'] = sp.stats.t(nu).sf(np.abs(self.res['t'])) * 2.0
        self.LLR = 2.0*(self.ll_full - self.ll_null)
        self.BIC = (np.log(self.n_obs) * len(self.params) - 2 * self.ll_full)
        self.AIC = (2 * len(self.params) - 2 * self.ll_full)
        self.dev = self.deviance(self.params)
        self.yhat = self.predict(params=self.params)
        chi2 = (_check_shape(self.Y, 1) - self.yhat)**2
        chi2/= self.yhat + self.yhat**2 * np.exp(self.params[-1])
        self.chi2 = np.sum(chi2)
        self.scchi2 = self.chi2 / (self.n_obs - self.n_feats)
        self.chi2_p = sp.stats.chi2.sf(self.chi2,  (self.n_obs - self.n_feats))
        self.dev_p = sp.stats.chi2.sf(self.dev,  (self.n_obs - self.n_feats))
        self.LLRp = sp.stats.chi2.sf(self.LLR,  (self.n_feats-1))
        n, p = self.X.shape
        yhat = self.predict(params=self.params)
        self.ssr =np.sum((yhat - yhat.mean())**2)
        rmax =  (1 - np.exp(2.0/n * (self.ll_null)))
        rcs = 1 - np.exp(2.0/n*-(self.ll_full-self.ll_null))
        rna = rcs / rmax
        rmf = 1 - self.ll_full/self.ll_null
        rma = 1 - (self.ll_full-p)/self.ll_null
        rmc = self.ssr/(self.ssr+3.29*n)
        rad = self.LLR/(self.LLR+n)
        
        ss = [[self.AIC, self.BIC, self.chi2, self.dev, self.LLR, self.scchi2,
               rcs, rna, rmf, rma, rmc, rad],
              ['-', '-', self.chi2_p, self.dev_p, self.LLRp, '-', '-', '-',
               '-', '-', '-', '-']]
        ss = pd.DataFrame(ss).T
        ss.index = ['AIC', 'BIC', 'chi2', 'deviance', 'LLR', 'scaled_chi2', 
                    'R2_Cox_Snell', 'R2_Nagelkerke', 'R2_McFadden',
                    'R2_McFaddenAdj', 'R2_McKelvey', 'R2_Aldrich']
        ss.columns = ['Test_stat', 'P-value']
        self.sumstats = ss
                     
    def predict(self, X=None, params=None, b=None):
        if X is None:
            X = self.X
        if b is None:
            b = params[:-1]
        mu_hat = np.exp(X.dot(b))
        return mu_hat
    
    def bootstrap(self, n_boot=2000, opt_kws={}):
        default_opt_kws = dict(gtol=1e-4, xtol=1e-4, verbose=0)
        for key, val in default_opt_kws.items():
            if key not in opt_kws.keys():
                opt_kws[key] = val
        self.fit()
        t_init = self.params.copy()
        beta_samples = np.zeros((n_boot, len(t_init)))
        X, Y = self.X.copy(), self.Y.copy()
        n = self.X.shape[0]
        ix = np.random.choice(n, n)
        
        Xb, Yb = X.copy(), Y.copy()
        pbar = tqdm.tqdm(total=n_boot)
        for i in range(n_boot):
            ix = np.random.choice(n, n)
            self.X, self.y = Xb[ix], Yb[ix][:, 0]
            optf = sp.optimize.minimize(self.loglike, 
                                        t_init, method='trust-constr',
                                        jac=self.gradient, 
                                        hess=self.hessian,
                                        options=opt_kws)
            beta_samples[i] = optf.x
            pbar.update(1)
        pbar.close()
        self.beta_samples = beta_samples
        self.X, self.Y = X, Y
        samples_df = pd.DataFrame(self.beta_samples, columns=self.res.index)
        boot_res = samples_df.agg(["mean", "std", "min"]).T
        boot_res["pct1.0%"] = sp.stats.scoreatpercentile(beta_samples, 1.0, axis=0)
        boot_res["pct2.5%"] = sp.stats.scoreatpercentile(beta_samples, 2.5, axis=0)
        boot_res["pct97.5%"] = sp.stats.scoreatpercentile(beta_samples, 97.5, axis=0)
        boot_res["pct99.0%"] = sp.stats.scoreatpercentile(beta_samples, 99.0, axis=0)
        boot_res["max"] = np.max(beta_samples, axis=0)
        boot_res["t"] = boot_res["mean"] / boot_res["std"]
        boot_res["p"] = sp.stats.t(self.X.shape[0]).sf(np.abs(boot_res["t"] ))*2.0
        self.boot_res =  boot_res
                    
    
        
        
        
                                
    