#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 21:33:29 2020

@author: lukepinkel
"""
import tqdm
import patsy
import numpy as np
import scipy as sp
import scipy.stats
import pandas as pd
from ..utilities.linalg_operations import _check_shape
from ..utilities.data_utils import _check_type
from ..utilities.optimizer_utils import process_optimizer_kwargs
from .links import LogitLink, ProbitLink, Link # analysis:ignore

class CLM:
    
    def __init__(self, frm, data, X=None, Y=None, link=LogitLink):
        Y, X = patsy.dmatrices(frm, data, return_type='dataframe')
        self.X, self.xcols, self.xix, self.x_is_pd = _check_type(X)
        self.Y, self.ycols, self.yix, self.y_is_p = _check_type(Y)
        self.Y = _check_shape(self.Y, 1)
        self.n_cats = len(np.unique(self.Y[~np.isnan(self.Y)]))
        self.resps = np.unique(self.Y[~np.isnan(self.Y)])
        self.resps = np.sort(self.resps)
        self.Y = np.concatenate([(self.Y==x)[:, None] for x in self.resps],
                                 axis=1).astype(float)
        #self.W = self.Y.dot(np.arange(self.n_cats))+1.0
        self.W = np.ones(self.X.shape[0])
        self.constraints = [dict(zip(
                ['type', 'fun'], 
                ['ineq', lambda params: params[i+1]-params[i]])) 
                for i in range(self.n_cats-2)]
        self.A1, self.A2 = self.Y[:, :-1], self.Y[:, 1:]
        self.o1, self.o2 = self.Y[:, -1]*30e1, self.Y[:, 0]*-10e5
        self.ix = self.o1!=0
        self.B1 = np.block([self.A1, -self.X])
        self.B2 = np.block([self.A2, -self.X])
       
        intercept = np.ones((self.X.shape[0], 1))
        self.B1_intercept = np.block([self.A1, -intercept])
        self.B2_intercept = np.block([self.A2, -intercept])
        if isinstance(link, Link) is False:
            link = link()
        self.f = link
        Yprops = np.sum(self.Y, axis=0).cumsum()[:-1]/np.sum(self.Y)
        self.theta_init = sp.stats.norm(0, 1).ppf(Yprops)
        self.beta_init = np.ones(self.X.shape[1])
        self.params_init = np.concatenate([self.theta_init, self.beta_init], axis=0)
    
    def loglike(self, params, B1=None, B2=None):
        W = self.W
        params = _check_shape(params, 1)
        o1, o2 = self.o1, self.o2
        if B1 is None:
            B1 = self.B1
        if B2 is None:
            B2 = self.B2
        Nu_1, Nu_2 = B1.dot(params)+o1, B2.dot(params)+o2
        Gamma_1, Gamma_2 = self.f.inv_link(Nu_1), self.f.inv_link(Nu_2)
        Pi = Gamma_1 - Gamma_2
        LL = np.sum(W * np.log(Pi))
        return -LL
    
    def gradient(self, params, B1=None, B2=None):
        W = self.W
        o2 = self.o2
        
        if B1 is None:
            B1 = self.B1
        if B2 is None:
            B2 = self.B2
            
        Nu_1, Nu_2 = B1.dot(params), B2.dot(params)+o2
        Phi_11, Phi_12 = self.f.dinv_link(Nu_1), self.f.dinv_link(Nu_2)
        Phi_11[self.ix] = 0.0
        Phi_11, Phi_12 = _check_shape(Phi_11, 2), _check_shape(Phi_12, 2)
        
        dPi = (B1 * Phi_11).T - (B2*Phi_12).T
        Gamma_1, Gamma_2 = self.f.inv_link(Nu_1), self.f.inv_link(Nu_2)
        Gamma_1[self.ix] = 1.0
        Pi = Gamma_1 - Gamma_2
        g = -np.dot(dPi, W / Pi)
        return g
    
    def hessian(self, params, B1=None, B2=None):
        
        W = self.W
        o2 = self.o2
        
        if B1 is None:
            B1 = self.B1
        if B2 is None:
            B2 = self.B2
        
        Nu_1, Nu_2 = B1.dot(params), B2.dot(params)+o2
        
        Phi_21, Phi_22 = self.f.d2inv_link(Nu_1),  self.f.d2inv_link(Nu_2)
        Phi_11, Phi_12 = self.f.dinv_link(Nu_1), self.f.dinv_link(Nu_2)
        
        Phi_11[self.ix] = 0.0
        Phi_21[self.ix] = 0.0
        
        Phi_11, Phi_12 = _check_shape(Phi_11, 2), _check_shape(Phi_12, 2)
        Phi_21, Phi_22 = _check_shape(Phi_21, 2), _check_shape(Phi_22, 2)
        
        Gamma_1, Gamma_2 = self.f.inv_link(Nu_1), self.f.inv_link(Nu_2)
        Gamma_1[self.ix] = 1.0
        Pi = Gamma_1 - Gamma_2
        Phi3 =_check_shape(W / Pi**2, 2)
        dPi = (B1 * Phi_11).T - (B2*Phi_12).T
        Pi = _check_shape(Pi, 2)
        T0 = (B1 * Phi_21 / Pi).T.dot(B1)
        T1 = (B2 * Phi_22 / Pi).T.dot(B2)
        T2 = dPi.dot(dPi.T*Phi3)
        H = T0 - T1 - T2
        return -H
    
    def _optimize(self, params=None, model_args=(None, None), 
                  optimizer_kwargs={}):
        params = self.params_init.copy() if params is None else params
        
        optimizer_kwargs = process_optimizer_kwargs(optimizer_kwargs)
        
        opt = sp.optimize.minimize(self.loglike, params, args=model_args,
                                   jac=self.gradient, hess=self.hessian, 
                                   constraints=self.constraints, 
                                   **optimizer_kwargs)
        return opt
    
    def _postprocess_params(self, params, SE):
        res = np.concatenate([_check_shape(params, 2), _check_shape(SE, 2)],
                              axis=1)
        res = pd.DataFrame(res, columns=['coef', 'SE'])
        res['t'] = res['coef']/res['SE']
        n_th = len(self.theta_init)
        idx = ["threshold %i|%i"%(i, i+1) for i in range(1, n_th+1)]
        if self.xcols is not None:
            idx = idx+self.xcols.tolist()
        else:
            idx = idx+["beta%i"%i for i in range(self.X.shape[1])]
        res.index  = idx
        t_df = self.Y.shape[0]-len(params)
        tdist = sp.stats.t(t_df)
        res['p'] = tdist.sf(np.abs(res['t']))*2.0
        theta = self.params[:len(self.theta_init)]
        beta = self.params[len(self.theta_init):]
        return res, theta, beta
    
    def _summarystats(self, paramsf, paramsr):
        LLf = self.loglike(paramsf)
        LL0 = self.loglike(paramsr, self.B1_intercept, self.B2_intercept)
        degfree_f = len(paramsf)
        degfree_r = len(paramsr)
        degfree_llr = degfree_f - degfree_r
        chi2_dist = sp.stats.chi2(degfree_llr)
        LLR = LL0 - LLf
        LLR_pvalue = chi2_dist.sf(LLR)
        n, p = self.X.shape
        rmax =  (1 - np.exp(-2.0/n * (LL0)))
        r2_coxsnell = 1 - np.exp(2.0/n*(LLf - LL0))
        r2_mcfadden = 1 - LLf / LL0
        r2_mcfadden_adj = 1 - (LLf-p) / LL0
        r2_nagelkerke = r2_coxsnell / rmax
        
        statistic = dict(LLR=LLR, 
                         r2_coxsnell=r2_coxsnell,
                         r2_mcfadden=r2_mcfadden, 
                         r2_mcfadden_adj=r2_mcfadden_adj,
                         r2_nagelkerke=r2_nagelkerke)
        pvalue = dict(LLR=LLR_pvalue,
                      r2_coxsnell=None,
                      r2_mcfadden=None,
                      r2_mcfadden_adj=None,
                      r2_nagelkerke=None)
        sstats = pd.DataFrame(dict(statistic=statistic, pvalue=pvalue))
        return sstats
        
         
    
    def fit(self, model_args=(None, None), optimizer_kwargs={}):
        r_args = (self.B1_intercept, self.B2_intercept)
        intercept_init = np.concatenate([self.theta_init, np.zeros(1)], axis=0)
        self.optr = self._optimize(intercept_init, r_args, optimizer_kwargs)
        self.optf = self._optimize(None, model_args, optimizer_kwargs)
        self.params = self.optf.x
        self.H = self.hessian(self.params)
        self.Vcov = np.linalg.pinv(self.H)
        self.SE = np.diag(self.Vcov)**0.5
        self.res, self.theta, self.beta = self._postprocess_params(self.params,
                                                                   self.SE)
        self.sumstats = self._summarystats(self.optf.x, self.optr.x)
        
     
    def predict(self, beta=None, theta=None):
        if beta is None:
            beta = self.beta
        if theta is None:
            theta = self.theta
        yhat = self.X.dot(beta)
        th = np.concatenate([np.array([-1e6]), theta, np.array([1e6])])
        yhat = pd.cut(yhat, th).codes.astype(float)
        return yhat
    
    def bootstrap(self, n_boot=2000):
        self.fit()
        t_init = self.params.copy()
        beta_samples = np.zeros((n_boot, len(t_init)))
        o1, o2 = self.o1.copy(), self.o2.copy()
        B1, B2 = self.B1, self.B2
        n = self.X.shape[0]
        ix = np.random.choice(n, n)
        
        B1b, B2b, o1b, o2b = B1[ix], B2[ix], o1[ix], o2[ix]
        self.o1, self.o2 = o1b, o2b
        pbar = tqdm.tqdm(total=n_boot)
        for i in range(n_boot):
            ix = np.random.choice(n, n)
            B1b, B2b, o1b, o2b = B1[ix], B2[ix], o1[ix], o2[ix]
            self.o1, self.o2 = o1b, o2b
            optf = self._optimize(t_init, (B1b, B2b), 
                                  {'options':dict(verbose=0,  gtol=1e-4, 
                                                  xtol=1e-4)})
            beta_samples[i] = optf.x
            pbar.update(1)
        pbar.close()
        self.beta_samples = beta_samples
        self.o1, self.o2 = o1, o2
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
                    
    
        
        
            
            