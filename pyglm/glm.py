#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 19:23:32 2020

@author: lukepinkel
"""

import tqdm
import patsy  # analysis:ignore
import numpy as np # analysis:ignore
import scipy as sp # analysis:ignore
import scipy.stats # analysis:ignore
import pandas as pd # analysis:ignore
from ..utilities.linalg_operations import _check_shape
from ..utilities.data_utils import _check_type
from .links import LogitLink, ProbitLink, Link, LogLink, ReciprocalLink, PowerLink # analysis:ignore
from .families import (Binomial, ExponentialFamily, Gamma, Gaussian,  # analysis:ignore
                      IdentityLink, InverseGaussian, NegativeBinomial,  # analysis:ignore
                      Poisson) # analysis:ignore


class GLM:
    
    def __init__(self, X=None, Y=None, formula=None, data=None, fam=None, 
                 scale_estimator='M'):
        if isinstance(fam, ExponentialFamily) is False:
            fam = fam()
        
        self.f = fam
        
        if formula is not None and data is not None:
            Y, X = patsy.dmatrices(formula, data, return_type='dataframe')
            X, xcols, xix = X.values, X.columns, X.index
            Y, ycols, yix = Y.values, Y.columns, Y.index
        elif X is not None and Y is not None:
            if type(X) not in [pd.DataFrame, pd.Series]:
                xcols = [f'x{i}' for i in range(1, X.shape[1]+1)]
                xix = np.arange(X.shape[0])
            else:
                xcols, xix = X.columns, X.index
                X = X.values
            if type(Y) not in [pd.DataFrame, pd.Series]:
                ycols = ['y']
                yix = np.arange(Y.shape[0])
            else:
                 ycols, yix = Y.columns, Y.index
                 Y = Y.values
                 
        self.X, self.xcols, self.xix, self.x_is_pd = X, xcols, xix, True
        self.Y, self.ycols, self.yix, self.y_is_pd = Y, ycols, yix, True
        self.n_obs, self.n_feats = self.X.shape
        self.dfe = self.n_obs - self.n_feats
        self.jn = np.ones((self.n_obs, 1))
        self.YtX = self.Y.T.dot(self.X)
        self.theta_init = np.zeros(self.X.shape[1])
        self.param_labels = list(self.xcols) 
        if isinstance(fam, Gamma) or isinstance(fam, InverseGaussian):
            if isinstance(fam, InverseGaussian):
                mu0 = (self.Y + self.Y.mean()) / 2.0
            else:
                mu0 = self.Y
            nu, gp, vmu = self.f.link(mu0), self.f.dlink(mu0), self.f.var_func(mu=mu0)
            w = 1 / (vmu[:, None] * gp**2)
            b0 = np.linalg.solve((self.X * w).T.dot(self.X), (self.X * w).T.dot(nu))
            self.theta_init = _check_shape(b0, 1)
        
        if isinstance(fam, (Binomial, Poisson)):
            self.scale_handling = 'fixed'
        else:
            self.scale_handling = scale_estimator   
        
        if self.scale_handling == 'NR':
            if isinstance(fam, Gamma) or isinstance(fam, InverseGaussian):
                mu_hat_init = self.f.inv_link(self.X.dot(self.theta_init))
                phi_init = self._est_scale(self.Y, mu_hat_init)
            else:
                phi_init = np.ones(1)
            self.theta_init = np.concatenate([self.theta_init, np.atleast_1d(phi_init)])
            self.param_labels += ['log_scale']
            
    def _est_scale(self, y, mu):
    
        y, mu = self.f.cshape(y, mu)
        r = self.f.weights * (y - mu)**2
        v = self.f.var_func(mu=mu)
        s = np.sum(r / v)
        s/= self.dfe
        return s
    
    def predict(self, params, X=None, Y=None):
        if X is None:
            X = self.X
        if self.scale_handling == 'NR':
            beta, _ = params[:-1], params[-1]
            eta = X.dot(beta)
            mu = self.f.inv_link(eta)
        else:
            eta = X.dot(params)
            mu = self.f.inv_link(eta)
        return mu
    
    def _check_mats(self, params, X, Y):
        if X is None:
            X = self.X
        if Y is None:
            Y = self.Y
        params = _check_shape(params, 1)
        return params, X, Y 
    
    def _handle_scale(self, params, X, Y):
        if self.scale_handling == 'NR':
            beta, tau = params[:-1], params[-1]
            eta = X.dot(beta)
            mu = self.f.inv_link(eta)
            phi = np.exp(tau)
        else:
            eta = X.dot(params)
            mu = self.f.inv_link(eta)
            if self.scale_handling == 'M':
                phi = self._est_scale(Y, mu)
            else:
                phi = 1.0
            tau = np.log(phi)
        return mu, phi, tau
    
    def loglike(self, params, X=None, Y=None):
        params, X, Y = self._check_mats(params, X, Y)
        mu, phi, _ = self._handle_scale(params, X, Y)
        ll = self.f.loglike(Y, mu=mu, scale=phi)
        return ll

    def gradient(self, params, X=None, Y=None):
        params, X, Y = self._check_mats(params, X, Y)
        mu, phi, tau = self._handle_scale(params, X, Y)
        w = self.f.gw(Y, mu=mu, phi=phi)
        g = np.dot(X.T, w)
        if self.scale_handling == 'NR':
            dt = np.atleast_1d(np.sum(self.f.dtau(tau, Y, mu)))
            g = np.concatenate([g, dt])
        return g
    
    def hessian(self, params, X=None, Y=None):
        params, X, Y = self._check_mats(params, X, Y)
        mu, phi, tau = self._handle_scale(params, X, Y)
        w = self.f.hw(Y, mu=mu, phi=phi)
        H = (X.T * w).dot(X)
        if self.scale_handling == 'NR':
            d2t = np.atleast_2d(self.f.d2tau(tau, Y, mu))
            dbdt = -np.atleast_2d(self.gradient(params)[:-1])
            H = np.block([[H, dbdt.T], [dbdt, d2t]])
        return H 
    
    
    def _fit_optim(self, opt_kws={}, t_init=None, X=None, Y=None):
        if t_init is None:
            t_init = self.theta_init

        default_args = dict(verbose=0, gtol=1e-6, xtol=1e-6)
        for key, val in default_args.items():
            if key not in opt_kws.keys():
                opt_kws[key] = val
        optimizer = sp.optimize.minimize(self.loglike, t_init, args=(X, Y),
                                         jac=self.gradient, hess=self.hessian, 
                                         options=opt_kws, method='trust-constr')
        return optimizer
    
    def _fit_manual(self, theta=None):

        if theta is None:
            theta = self.theta_init
    
        fit_hist = {'|g|':[], 'theta':[], 'i':[], 'll':[],
                    'step_half_success':[], 'n_step_halves':[]}
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
            step_half_success, n_step_halves = None, None
            if self.loglike(theta - dx)>ll_k:
                step_half_success = False
                for j in range(100):
                    sh*=2
                    if self.loglike(theta - dx/sh)<ll_k:
                        step_half_success = True
                        n_step_halves = j+1
                        break
            fit_hist['step_half_success'].append(step_half_success)
            fit_hist['n_step_halves'].append(n_step_halves)
            if step_half_success==False:
                break
            theta -= dx/sh
            ll_k = self.loglike(theta)
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
        presid = self.f.weights * (y - mu) / np.sqrt(self.f.var_func(mu=mu))
        self.pearson_resid = presid
        self.mu = mu
        self.y = y
        mu, phi, tau = self._handle_scale(params, self.X, self.Y)
        
        if self.scale_handling == 'NR':
            self.beta = params[:-1]
        else:
            self.beta = params
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

        sumstats['PseudoR2_CS'] = 1-np.exp(1.0/N * (self.LLA - self.LL0))
        rmax = 1-np.exp(1.0/N *(-self.LL0))
        LLR = 2*(lln - llf)
        sumstats['PseudoR2_N'] = sumstats['PseudoR2_CS'] / rmax
        sumstats['PseudoR2_MCF'] = 1 - self.LLA / self.LL0
        sumstats['PseudoR2_MCFA'] = 1 - (self.LLA - k) / self.LL0
        sumstats['PseudoR2_MFA'] = 1 - (LLR) / (LLR + N)
        
        self.sumstats = pd.DataFrame(sumstats, index=['Fit Statistic']).T

        self.vcov = np.linalg.pinv(self.hessian(self.params))
        #V = self.vcov
        #W = (self.X.T * self.f.gw(self.y, self.mu, phi=self.phi)).dot(self.X)
        #self.vcov_robust = V.dot(W).dot(V)
        
        self.se_theta = np.diag(self.vcov)**0.5
        self.res = np.vstack([self.params, self.se_theta]).T
        self.res = pd.DataFrame(self.res, columns=['params', 'SE'], index=self.param_labels)
        self.res['t'] = self.res['params'] / self.res['SE']
        self.res['p'] = sp.stats.t.sf(np.abs(self.res['t']), self.dfe)*2.0
    
    
    def _bootfit(self, params, X=None, Y=None, n_iters=500, tol=1e-6):
        params, X, Y = self._check_mats(params, X, Y)
        for i in range(n_iters):
            H = self.hessian(params, X=X, Y=Y)
            g = self.gradient(params, X=X, Y=Y)
            d = -np.linalg.solve(H, g)
            if np.abs(d).mean() < tol:
                break
            params = params + d
        return params, i
    
    def bootstrap(self, n_boot=5000, opt_kws={}, method='sp'):
        if hasattr(self, 'res')==False:
            self.fit()
        t_init = self.params
        theta_samples = np.zeros((n_boot, len(t_init)))
        pbar = tqdm.tqdm(total=n_boot)
        for i in range(n_boot):
            ix = np.random.choice(self.X.shape[0], self.X.shape[0])
            if method == 'sp':
                theta_samples[i] = self._fit_optim(opt_kws=opt_kws,
                                               t_init=t_init, 
                                               X=self.X[ix], 
                                               Y=self.Y[ix]).x
            else:
                theta_samples[i], _ = self._bootfit(params=t_init, X=self.X[ix],
                                                   Y=self.Y[ix])
            pbar.update(1)
        pbar.close()
        k = self.n_obs-self.n_feats
        self.res.insert(2, "SE_boot", theta_samples.std(axis=0))
        self.res.insert(4, "t_boot", self.res['params']/self.res['SE_boot'])
        abst = np.abs(self.res['t_boot'])
        self.res.insert(6, "p_boot", sp.stats.t(k).sf(abst)*2.0)
        self.theta_samples = theta_samples
        

            
                
                
        
        