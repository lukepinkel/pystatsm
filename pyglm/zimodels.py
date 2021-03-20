# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 22:33:08 2020

@author: lukepinkel
"""

import numpy as np
import scipy as sp
import scipy.stats
import pandas as pd

def process_data(X, default_varname='x'):
    if type(X) not in [pd.DataFrame, pd.Series]:
        if X.ndim==1:
            xcols = [f'{default_varname}']
        else:
            xcols = [f'{default_varname}{i}' for i in range(1, X.shape[1]+1)]
        xix = np.arange(X.shape[0])
    else:
        xcols, xix = X.columns, X.index
        X = X.values
    return X, xcols, xix


class ZIP:
    
    def __init__(self, X, y, Z=None):
        self.X, self.xcols, self.xix = process_data(X)
        self.y, self.ycols, self.yix = process_data(y, "y")
        if self.y.ndim==2:
            self.y = self.y.reshape(-1)
        if self.X.ndim==1:
            self.X = self.X.reshape(-1, 1)
            
        self.n_obs, self.n_xvars = self.X.shape
                
        if Z is None:
            self.Z = np.ones((self.n_obs, 1))
            self.zcols = ['z']
            self.zix = self.xix
        else:
            self.Z, self.zcols, self.zix = process_data(Z, "z")
        
        self.n_zvars = self.Z.shape[1]
        self.n_params = self.n_xvars+self.n_zvars
        self.params = np.zeros(self.n_params)
        self.ix0 = self.y==0
        self.ix1 = ~self.ix0
        self.X0, self.X1 = self.X[self.ix0], self.X[self.ix1]
        self.y0, self.y1 = self.y[self.ix0], self.y[self.ix1]
        self.Z0, self.Z1 = self.Z[self.ix0], self.Z[self.ix1]
        self._llc = sp.special.gammaln(self.y1)
    
    def _check_params(self, params, X):
        b, a = params[:X.shape[1]], params[X.shape[1]:]
        b, a = np.atleast_1d(b), np.atleast_1d(a)
        return b, a
    
    def _check_mats(self, X, Z):
        if X is None:
            X = self.X
        if Z is None:
            Z = self.Z
        X0, Z0 = X[self.ix0], Z[self.ix0]
        X1, Z1 = X[self.ix1], Z[self.ix1]
        return X, Z, X0, Z0, X1, Z1
    
    def _loglikez(self, b, a, X0, Z0):
        etax, etaz = X0.dot(b), Z0.dot(a)
        mux, muz = np.exp(etax), np.exp(etaz)
        ll = np.log(muz + np.exp(-mux))
        return ll
    
    def _logliken(self, b, X1):
        mu = np.exp(X1.dot(b))
        ll = self.y1 * np.log(mu) - mu - self._llc
        return ll
    
    def loglike(self, params, X=None, Z=None):
        X, Z, X0, Z0, X1, _ = self._check_mats(X, Z)
        b, a = self._check_params(params, X)
        llz = self._loglikez(b, a, X0, Z0)
        lln = self._logliken(b, X1)
        llm = np.log(1.0 + np.exp(Z.dot(a)))
        ll = np.sum(llz) + np.sum(lln) - np.sum(llm)
        return -ll
    
    def _gradientb(self, mu0, u0, mu1, X0, X1):
        w = mu0 / (np.exp(mu0) * u0 + 1.0)
        g = - X0.T.dot(w)
        g = g + X1.T.dot(self.y1 - mu1)
        return g
    
    def _gradienta(self, mu0, u0, u, Z0, Z):
        v = u0 * np.exp(mu0)
        w = v / (v + 1.0)
        g = Z0.T.dot(w)
        g = g - Z.T.dot(u / (u + 1.0))
        return g
        
    def gradient(self, params, X=None, Z=None):
        X, Z, X0, Z0, X1, Z1 = self._check_mats(X, Z)
        b, a = self._check_params(params, X)
        u, mu = np.exp(Z.dot(a)), np.exp(X.dot(b))
        u0 = u[self.ix0]
        mu0, mu1 = mu[self.ix0], mu[self.ix1]
        db = np.atleast_1d(self._gradientb(mu0, u0, mu1, X0, X1))
        da = np.atleast_1d(self._gradienta(mu0, u0, u, Z0, Z))
        g = np.concatenate((db, da))
        return -g
    
    def _hessbb(self, mu0, u0, mu1, X0, X1):
        w = mu0 * ((mu0 - 1.0) * u0 * np.exp(mu0) - 1.0)
        w = w / (u0 * np.exp(mu0) + 1.0)**2
        w = w.reshape(-1, 1)
        H = (X0 * w).T.dot(X0)
        H = H - (X1 * mu1.reshape(-1, 1)).T.dot(X1)
        return H
    
    def _hessaa(self, mu0, u0, u, Z0, Z):
        v = u0 * np.exp(mu0)
        w = v /(v + 1)**2
        w = w.reshape(-1, 1)
        H = (Z0 * w).T.dot(Z0)
        w = u / (u + 1.0)**2
        w = w.reshape(-1, 1)
        H = H - (Z * w).T.dot(Z)
        return H
    
    def _hessba(self, mu0, u0, X0, Z0):
        v = u0 * np.exp(mu0)
        w = ((v * mu0) / (v + 1.0)**2).reshape(-1, 1)
        H = (X0 * w).T.dot(Z0)
        return H
    
    def hessian(self, params, X=None, Z=None):
        X, Z, X0, Z0, X1, Z1 = self._check_mats(X, Z)
        b, a = self._check_params(params, X)
        u = np.exp(Z.dot(a))
        u0 = u[self.ix0]
        mu = np.exp(X.dot(b))
        mu0 = mu[self.ix0]
        mu1 = mu[self.ix1]
        
        Hbb = np.atleast_2d(self._hessbb(mu0, u0, mu1, X0, X1))
        Haa = np.atleast_2d(self._hessaa(mu0, u0, u, Z0, Z))
        Hba = np.atleast_2d(self._hessba(mu0, u0, X0, Z0))
        H = np.block([[Hbb, Hba], [Hba.T, Haa]])
        return -H
    
    def predict(self, params, X=None, Z=None):
        X, Z, _, _, _, _ = self._check_mats(X, Z)
        b, a = self._check_params(params, X)
        u = np.exp(Z.dot(a))
        mu = np.exp(X.dot(b))
        prob = u / (1.0 + u)
        yhat = mu * (1.0 - prob)
        var = yhat * (1.0 + mu * prob)
        return yhat, var
    
    def fit(self, opt_kws={}):
        theta = self.params
        null_args = np.ones((self.n_obs, 1)), np.ones((self.n_obs, 1))
        self.opt_null = sp.optimize.minimize(self.loglike, np.ones(2), args=null_args, 
                                             jac=self.gradient, hess=self.hessian, 
                                             method='trust-constr', options=opt_kws)
        self.ll_null = -self.loglike(self.opt_null.x, *null_args)
        self.opt = sp.optimize.minimize(self.loglike, theta, jac=self.gradient, 
                                        hess=self.hessian, method='trust-constr',
                                        options=opt_kws)
        self.ll_model = -self.loglike(self.opt.x)
        self.ll_ratio = -2.0 * (self.ll_null - self.ll_model)
        self.ll_rpval = sp.stats.chi2(self.n_params - 2).sf(self.ll_ratio)
        self.params = self.opt.x
        self.se_params = np.diag(np.linalg.inv(self.hessian(self.params)))**0.5
        t = self.params / self.se_params
        p = sp.stats.t(self.n_params).sf(np.abs(t))*2.0
        res = pd.DataFrame(np.vstack((self.params, self.se_params, 
                                      t, p)).T)
        res.columns = ['param', 'SE', 't', 'p']
        res.index = self.xcols+self.zcols
        self.res = res
        self.aic = -2.0 * (self.ll_model - self.n_params)
        self.bic = self.n_params * np.log(self.n_obs) - 2.0 * self.ll_model
        self.aicc = self.aic + (self.n_params**2 + self.n_params)/(self.n_obs-self.n_params-1.0)
        self.yhat, self.yhvar = self.predict(self.params)
        self.resids = self.y - self.yhat
        self.pearson_resids = self.resids / np.sqrt(self.yhvar)
        sumstats = dict(AIC=self.aic, AICc=self.aicc, BIC=self.bic, LLR=self.ll_ratio,
                        LLRp=self.ll_rpval)
        sumstats['PseudoR2_CS'] = 1 - np.exp(1.0/ self.n_obs * (self.ll_null - self.ll_model))
        rmax = 1-np.exp(1.0 / self.n_obs *(self.ll_null))
        sumstats['PseudoR2_N'] = sumstats['PseudoR2_CS'] / rmax
        sumstats['PseudoR2_MCF'] =  1.0 - self.ll_model / self.ll_null
        sumstats['PseudoR2_MCFA'] = 1.0 - (-self.ll_model - self.n_params) / -self.ll_null
        sumstats['PseudoR2_MFA'] = (self.ll_ratio) / (self.ll_ratio + self.n_obs)
        self.sumstats = pd.DataFrame(sumstats, index=['Fit Statistic']).T

        
        