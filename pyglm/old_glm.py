#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 22:50:32 2020

@author: lukepinkel
"""

import patsy  # analysis:ignore
import numpy as np # analysis:ignore
import scipy as sp # analysis:ignore
import scipy.stats # analysis:ignore
import pandas as pd # analysis:ignore
from ..utilities.linalg_operations import _check_shape
from ..utilities.data_utils import _check_type
from .links import LogitLink, ProbitLink, Link, LogLink # analysis:ignore
from .families import (Binomial, ExponentialFamily, Gamma, Gaussian,  # analysis:ignore
                      IdentityLink, InverseGaussian, NegativeBinomial,  # analysis:ignore
                      Poisson) # analysis:ignore


class GLM:
    
    def __init__(self, frm=None, data=None, fam=None, scale_estimator='M'):
        if isinstance(fam, ExponentialFamily) is False:
            fam = fam()
        self.f = fam
        Y, X = patsy.dmatrices(frm, data, return_type='dataframe')
        self.X, self.xcols, self.xix, self.x_is_pd = _check_type(X)
        self.Y, self.ycols, self.yix, self.y_is_pd = _check_type(Y)
        self.n_obs, self.n_feats = self.X.shape
        self.dfe = self.n_obs - self.n_feats
        self.jn = np.ones((self.n_obs, 1))
        self.YtX = self.Y.T.dot(self.X)
        self.theta_init = np.zeros(self.X.shape[1])
        
        if isinstance(fam, Gamma):
            self.theta_init = np.linalg.lstsq(self.X, self.f.link(self.Y))[0]
            self.theta_init = _check_shape(self.theta_init, 1)
        
        if isinstance(fam, (Binomial, Poisson)):
            self.scale_handling = 'fixed'
        else:
            self.scale_handling = scale_estimator   
        
        if self.scale_handling == 'NR':
            if isinstance(fam, Gamma):
                mu_hat_init = self.f.inv_link(self.X.dot(self.theta_init))
                phi_init = self._est_scale(self.Y, mu_hat_init)
            else:
                phi_init = np.ones(1)
            self.theta_init = np.concatenate([self.theta_init, np.atleast_1d(phi_init)])
     
    def _est_scale(self, y, mu):
    
        y, mu = self.f.cshape(y, mu)
        r = (y - mu)**2
        v = self.f.var_func(mu=mu)
        s = np.sum(r / v)
        s/= self.dfe
        return s
    
    def predict(self, params):

        if self.scale_handling == 'NR':
            beta, _ = params[:-1], params[-1]
            eta = self.X.dot(beta)
            mu = self.f.inv_link(eta)
        else:
            eta = self.X.dot(params)
            mu = self.f.inv_link(eta)
        return mu
    
    def loglike(self, params):
        params = _check_shape(params, 1)
        if self.scale_handling == 'NR':
            beta, tau = params[:-1], params[-1]
            eta = self.X.dot(beta)
            mu = self.f.inv_link(eta)
            phi = np.exp(tau)
        else:
            eta = self.X.dot(params)
            mu = self.f.inv_link(eta)
            if self.scale_handling == 'M':
                phi = self._est_scale(self.Y, mu)
            else:
                phi = 1.0
        ll = self.f.loglike(self.Y, mu=mu, scale=phi)
        return ll

    def gradient(self, params):
        params = _check_shape(params, 1)
        if self.scale_handling == 'NR':
            beta, tau = params[:-1], params[-1]
            eta = self.X.dot(beta)
            mu = self.f.inv_link(eta)
            phi = np.exp(tau)
            dt = np.atleast_1d(np.sum(self.f.dtau(tau, self.Y, mu)))
        else:
            eta = self.X.dot(params)
            mu = self.f.inv_link(eta)
            if self.scale_handling == 'M':
                phi = self._est_scale(self.Y, mu)
            else:
                phi = 1.0
        w = self.f.gw(self.Y, mu=mu, phi=phi)
        g = np.dot(self.X.T, w)
        if self.scale_handling == 'NR':
            g = np.concatenate([g, dt])
        return g
    
    def hessian(self, params):
        if self.scale_handling == 'NR':
            beta, tau = params[:-1], params[-1]
            eta = self.X.dot(beta)
            mu = self.f.inv_link(eta)
            phi = np.exp(tau)
            d2t = np.atleast_2d(self.f.d2tau(tau, self.Y, mu))
            dbdt = -np.atleast_2d(self.gradient(params)[:-1])
        else:
            eta = self.X.dot(params)
            mu = self.f.inv_link(eta)
            if self.scale_handling == 'M':
                phi = self._est_scale(self.Y, mu)
            else:
                phi = 1.0
        w = self.f.hw(self.Y, mu=mu, phi=phi)
        H = (self.X.T * w).dot(self.X)
        if self.scale_handling == 'NR':
            H = np.block([[H, dbdt.T], [dbdt, d2t]])
        return H 
    
    
    def _fit_optim(self):

        opts = {'verbose':3}
        optimizer = sp.optimize.minimize(self.loglike, self.theta_init,
                                         jac=self.gradient,
                                         hess=self.hessian, options=opts,
                                         method='trust-constr')
        return optimizer
    
    def _fit_manual(self, theta=None):

        if theta is None:
            theta = self.theta_init
    
        fit_hist = {'|g|':[], 'theta':[], 'i':[], 'll':[]}
        ll_k = self.loglike(theta)
        sh = 1.0
        for i in range(100): 
            H = self.hessian(theta)
            g =  self.gradient(theta)
            gnorm = np.linalg.norm(g)
            fit_hist['|g|'].append(gnorm)
            fit_hist['i'].append(i)
            fit_hist['theta'].append(theta.copy())
            fit_hist['ll'].append(self.loglike(theta))
            if gnorm/len(g)<1e-9:
                 break
            dx = np.atleast_1d(np.linalg.solve(H, g))
            if self.loglike(theta - dx)>ll_k:
                for j in range(100):
                    sh*=2
                    if self.loglike(theta - dx/sh)<ll_k:
                        break
            theta -= dx/sh
            sh = 1.0
        return theta, fit_hist
            
    def fit(self, method=None):

        self.theta0 = self.theta_init.copy()
        if method is None:
            if isinstance(self.f, Gamma):
                method = 'mn'
            else:
                method = 'sp'
        if method == 'sp':
            res = self._fit_optim()
            params = res.x
        else:
            params, res = self._fit_manual()
        self.optimizer = res
        self.params = params
        mu = self.predict(params)
        y, mu = self.f.cshape(self.Y, mu)
        presid = (y - mu) / np.sqrt(self.f.var_func(mu=mu))
        self.pearson_resid = presid
        self.mu = mu
        self.y = y
        if self.scale_handling == 'NR':
            beta, tau = params[:-1], params[-1]
            eta = self.X.dot(beta)
            mu = self.f.inv_link(eta)
            phi = np.exp(tau)
        else:
            beta = params
            eta = self.X.dot(beta)
            mu = self.f.inv_link(eta)
            if self.scale_handling == 'M':
                phi = self._est_scale(self.Y, mu)
            else:
                phi = 1.0
        
        self.beta = beta
        self.phi = phi
        
        llf = self.f.full_loglike(y, mu=mu, scale=phi)
        lln = self.f.full_loglike(y, mu=np.ones(mu.shape[0])*y.mean(), 
                                  scale=phi)
        self.LLA = llf*2.0
        self.LL0 = lln*2.0
        k = len(params)
        N = self.X.shape[0]
        sumstats = {}
        sumstats['aic'] = 2*llf + k
        sumstats['aicc'] = 2*llf + (2 * k * N) / (N - k - 1)
        sumstats['bic'] = 2*llf + np.log(N)*k
        sumstats['caic'] = 2*llf + k*(np.log(N) + 1.0)
        sumstats['LLR'] = 2*(lln - llf)
        sumstats['pearson_chi2'] = self._est_scale(self.Y, 
                                    self.predict(self.params))*self.dfe
        sumstats['deviance'] = self.f.deviance(y=self.Y, mu=mu, scale=phi).sum()
        if isinstance(self.f, Binomial):
            sumstats['PseudoR2_CS'] = 1-np.exp(1.0/N * (self.LLA - self.LL0))
            rmax = 1-np.exp(1.0/N *(-self.LL0))
            LLR = 2*(lln - llf)
            sumstats['PseudoR2_N'] = sumstats['PseudoR2_CS'] / rmax
            sumstats['PseudoR2_MCF'] = 1 - self.LLA / self.LL0
            sumstats['PseudoR2_MCFA'] = 1 - (self.LLA - k) / self.LL0
            sumstats['PseudoR2_MFA'] = 1 - (LLR) / (LLR + N)
            
        self.sumstats = pd.DataFrame(sumstats, index=['Fit Statistic']).T

        self.vcov = np.linalg.pinv(self.hessian(self.params))
        V = self.vcov
        W = (self.X.T * self.f.gw(self.y, self.mu, phi=self.phi)).dot(self.X)
        self.vcov_robust = V.dot(W).dot(V)
        
        self.se_theta = np.diag(self.vcov)**0.5
        self.res = np.vstack([self.params, self.se_theta]).T
        self.res = pd.DataFrame(self.res, columns=['params', 'SE'])
        self.res['t'] = self.res['params'] / self.res['SE']
        self.res['p'] = sp.stats.t.sf(np.abs(self.res['t']), self.dfe)*2.0
        
        
        
        