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
from ..utilities.linalg_operations import _check_shape
from ..utilities.data_utils import _check_type

def trigamma(x):
    return sp.special.polygamma(1, x)      
    
class MinimalNB2:
    
    def __init__(self, X, Y):
        self.X, self.Y = X, Y
        self.n_obs, self.n_feats = X.shape
        self.beta = np.zeros(self.n_feats)
        self.varp = np.ones(1)/2.0
        self.params = np.concatenate([self.beta, np.log(self.varp)])
        self.cnst = [(None, None) for i in range(self.n_feats)]+[(None, None)]
    
    def loglike(self, params, X=None):
        if X is None:
            X = self.X
        y = _check_shape(self.Y, 1)
        b, a = params[:-1], np.exp(params[-1])
        v = 1.0 / a
        mu = np.exp(X.dot(b))
        u = 1.0 + a * mu
        lg = sp.special.gammaln(y + v) - sp.special.gammaln(v) - sp.special.gammaln(y + 1)
        ln = y * np.log(mu) + y * np.log(a) - (y + v) * np.log(u)
        ll = np.sum(lg + ln)
        return -ll
    
    def jac(self, params):
        params = _check_shape(params, 1)
        X, y = self.X, _check_shape(self.Y, 1)
        b, a = params[:-1], np.exp(params[-1])
        v = 1.0 / a
        mu = np.exp(X.dot(b))
        u = 1 + a * mu
        r = y-mu
        gb = X.T.dot(r / u)
        ga = np.sum(np.log(u)+(a*r)/u + sp.special.digamma(v) - sp.special.digamma(y+v))
        ga /= a**2
        g = np.concatenate([gb, np.array([ga])])
        return -g
    
    def gradient(self, params):
        params = _check_shape(params, 1)
        X, y = self.X, _check_shape(self.Y, 1)
        b, a = params[:-1], np.exp(params[-1])
        v = 1.0 / a
        mu = np.exp(X.dot(b))
        u = 1 + a * mu
        r = y-mu
        gb = X.T.dot(r / u)
        ga = np.sum(np.log(u)+(a*r)/u + sp.special.digamma(v) - sp.special.digamma(y+v))
        ga /= a**2
        g = np.concatenate([gb, np.array([ga])])
        g[-1] = g[-1] * a
        return -g
        
    def var_deriv(self, a, mu, y):
        v = 1/a
        u = 1+a*mu
        r = y-mu
        p = 1/u
        vm = v + mu
        vy = v+y
        a2, a3 = a**-2, a**-3
        a4 = (-a2)**2
        dig = sp.special.digamma(v+y) - sp.special.digamma(v)
        z = (dig + np.log(p) - (a * r) / u)
        trg = a4*(trigamma(vy)-trigamma(v) + a - 1/vm + r/(vm**2))
        res = 2*a3*z + trg
        return -res.sum()
    
    def hessian(self, params):
        X, y = self.X, _check_shape(self.Y, 1)
        b, a = params[:-1], np.exp(params[-1])

        mu = np.exp(X.dot(b))
        u = 1 + a * mu
        r = y-mu
        
        wbb = mu * (1.0 + a * y) / (u**2)
        Hb = -(X.T * wbb).dot(X)
        
        Hab = -X.T.dot((mu * r) / (u**2))
        
        Ha = np.array([self.var_deriv(a, mu, y)]) 
        Hq = -np.block([[Hb, Hab[:, None]], [Hab[:, None].T, Ha]])
                
        Jf = np.ones_like(params)
        Jf[-1] = a
        Jq = self.jac(params)
        
        Hf = np.zeros((len(params), len(params)))
        Hf[-1, -1] = Jq[-1]
        Jf = np.diag(Jf)
        H = Jf.dot(Hq).dot(Jf)+Hf.dot(Jf)
        return H
    
    def fit(self, verbose=0):
        params = self.params
        optimizer = sp.optimize.minimize(self.loglike, params, jac=self.gradient, 
                             hess=self.hessian, method='trust-constr',
                             bounds=self.cnst,
                             options={'verbose':verbose})
        self.optimize = optimizer
        self.params = optimizer.x
        self.LLA = self.loglike(self.params)
        self.vcov = np.linalg.inv(self.hessian(self.params))
        self.vcov[-1, -1]=np.abs(self.vcov[-1, -1])
        self.params_se = np.sqrt(np.diag(self.vcov))

    def predict(self, X=None, params=None, b=None):
        if X is None:
            X = self.X
        if b is None:
            b = params[:-1]
        mu_hat = np.exp(X.dot(b))
        return mu_hat
    
        
                                

class NegativeBinomial:
    
    def __init__(self, formula, data):
        Y, X = patsy.dmatrices(formula, data, return_type='dataframe')
        self.X, self.xcols, self.xix, self.x_is_pd = _check_type(X)
        self.Y, self.ycols, self.yix, self.y_is_pd = _check_type(Y)
        self.n_obs, self.n_feats = X.shape
        self.beta = np.zeros(self.n_feats)
        self.varp = np.log(np.ones(1)/2.0)
        self.params = np.concatenate([self.beta, self.varp])
        self.cnst = [(None, None) for i in range(self.n_feats)]+[(None, None)]
    
    def loglike(self, params, X=None):
        if X is None:
            X = self.X
        y = _check_shape(self.Y, 1)
        b, a = params[:-1], np.exp(params[-1])
        v = 1.0 / a
        mu = np.exp(X.dot(b))
        u = 1.0 + a * mu
        lg = sp.special.gammaln(y + v) - sp.special.gammaln(v) - sp.special.gammaln(y + 1)
        ln = y * np.log(mu) + y * np.log(a) - (y + v) * np.log(u)
        ll = np.sum(lg + ln)
        return -ll
    
    def jac(self, params):
        params = _check_shape(params, 1)
        X, y = self.X, _check_shape(self.Y, 1)
        b, a = params[:-1], np.exp(params[-1])
        v = 1.0 / a
        mu = np.exp(X.dot(b))
        u = 1 + a * mu
        r = y-mu
        gb = X.T.dot(r / u)
        ga = np.sum(np.log(u)+(a*r)/u + sp.special.digamma(v) - sp.special.digamma(y+v))
        ga /= a**2
        g = np.concatenate([gb, np.array([ga])])
        return -g
    
    def gradient(self, params):
        params = _check_shape(params, 1)
        X, y = self.X, _check_shape(self.Y, 1)
        b, a = params[:-1], np.exp(params[-1])
        v = 1.0 / a
        mu = np.exp(X.dot(b))
        u = 1 + a * mu
        r = y-mu
        gb = X.T.dot(r / u)
        ga = np.sum(np.log(u)+(a*r)/u + sp.special.digamma(v) - sp.special.digamma(y+v))
        ga /= a**2
        g = np.concatenate([gb, np.array([ga])])
        g[-1] = g[-1] * a
        return -g
        
    def var_deriv(self, a, mu, y):
        v = 1/a
        u = 1+a*mu
        r = y-mu
        p = 1/u
        vm = v + mu
        vy = v+y
        a2, a3 = a**-2, a**-3
        a4 = (-a2)**2
        dig = sp.special.digamma(v+y) - sp.special.digamma(v)
        z = (dig + np.log(p) - (a * r) / u)
        trg = a4*(trigamma(vy)-trigamma(v) + a - 1/vm + r/(vm**2))
        res = 2*a3*z + trg
        return -res.sum()
    
    def hessian(self, params):
        X, y = self.X, _check_shape(self.Y, 1)
        b, a = params[:-1], np.exp(params[-1])

        mu = np.exp(X.dot(b))
        u = 1 + a * mu
        r = y-mu
        
        wbb = mu * (1.0 + a * y) / (u**2)
        Hb = -(X.T * wbb).dot(X)
        
        Hab = -X.T.dot((mu * r) / (u**2))
        
        Ha = np.array([self.var_deriv(a, mu, y)]) 
        Hq = -np.block([[Hb, Hab[:, None]], [Hab[:, None].T, Ha]])
                
        Jf = np.ones_like(params)
        Jf[-1] = a
        Jq = self.jac(params)
        
        Hf = np.zeros((len(params), len(params)))
        Hf[-1, -1] = Jq[-1]
        Jf = np.diag(Jf)
        H = Jf.dot(Hq).dot(Jf)+Hf.dot(Jf)
        return H
    
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
    
    def var_mu(self, params):
        X = self.X
        b, a = params[:-1], params[-1]
        mu = np.exp(X.dot(b))
        v = mu + a * mu**2
        return v
        
    
    def fit(self, optimizer_kwargs=None):
        if optimizer_kwargs is None:
            optimizer_kwargs = {'method':'trust-constr', 
                                'options':{'verbose':0}}
        intercept_model = MinimalNB2(np.ones((self.n_obs ,1)), self.Y)
        intercept_model.fit()
        self.LL0 = intercept_model.LLA
        params = self.params
        optimizer = sp.optimize.minimize(self.loglike, params,
                                         jac=self.gradient, 
                                         hess=self.hessian,
                                         bounds=self.cnst,
                                         **optimizer_kwargs)
        
        self.optimizer = optimizer
        self.params = optimizer.x
        self.LLA = self.loglike(self.params)
        self.vcov = np.linalg.inv(self.hessian(self.params))
        self.vcov[-1, -1]=np.abs(self.vcov[-1, -1])
        self.params_se = np.sqrt(np.diag(self.vcov))
        self.res = pd.DataFrame(np.vstack([self.params, self.params_se]),
                                columns=self.xcols.tolist()+['variance']).T
        self.res.columns = ['param', 'SE']
        self.res['t'] = self.res['param']/self.res['SE']
        self.res['p'] = sp.stats.t.sf(np.abs(self.res['t']),
                        self.n_obs-self.n_feats)*2.0
        self.LLR = -(self.LLA - self.LL0)
        self.BIC = -(np.log(self.n_obs) * len(self.params) - 2 * self.LLA)
        self.AIC = -(2 * len(self.params) - 2 * self.LLA)
        self.dev = self.deviance(self.params)
        self.yhat = self.predict(params=self.params)
        chi2 = (_check_shape(self.Y, 1) - self.yhat)**2
        chi2/= self.var_mu(self.params)
        self.chi2 = np.sum(chi2)
        self.scchi2 = self.chi2 / (self.n_obs - self.n_feats)
        self.chi2_p = sp.stats.chi2.sf(self.chi2,  (self.n_obs - self.n_feats))
        self.dev_p = sp.stats.chi2.sf(self.dev,  (self.n_obs - self.n_feats))
        self.LLRp = sp.stats.chi2.sf(self.LLR,  (self.n_obs - self.n_feats))
        n, p = self.X.shape
        yhat = self.predict(params=self.params)
        self.ssr =np.sum((yhat - yhat.mean())**2)
        rmax =  (1 - np.exp(-2.0/n * (self.LL0)))
        rcs = 1 - np.exp(2.0/n*(self.LLA-self.LL0))
        rna = rcs / rmax
        rmf = 1 - self.LLA/self.LL0
        rma = 1 - (self.LLA-p)/self.LL0
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
            self.X, self.Y = Xb[ix], Yb[ix]
            optf = sp.optimize.minimize(self.loglike, 
                                        t_init, method='trust-constr',
                                        jac=self.gradient, 
                                        hess=self.hessian,
                                        bounds=self.cnst,
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
                    
    
        
        
        
                                
    