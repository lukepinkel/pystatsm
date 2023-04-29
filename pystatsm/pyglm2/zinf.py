#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 17:06:44 2023

@author: lukepinkel
"""


import numpy as np
import scipy as sp
import scipy.stats
import pandas as pd

from .regression_model import RegressionMixin, ModelData
from .likelihood_model import LikelihoodModel
from .links import (CloglogLink, IdentityLink, Link, LogitLink, LogLink, #analysis:ignore
                    LogComplementLink, NegativeBinomialLink, ProbitLink,
                    PowerLink)




class ZeroInflatedPoisson(RegressionMixin, LikelihoodModel):
    
    def __init__(self, formula=None, zero_formula=None, data=None, link=LogLink,
                 zero_link=LogitLink, X=None, Z=None, y=None,  weights=None, *args, **kwargs):
        formulas = [formula, zero_formula] if zero_formula else [formula]
        Xs = [X, Z] if Z is not None else [X]
        super().__init__(formula=formulas, data=data, X=Xs, y=y, weights=weights, 
                         *args, **kwargs)
        self.link = link() if isinstance(link, Link) is False else link
        self.zero_link = zero_link() if isinstance(zero_link, Link) is False else zero_link
        self.n = self.n_obs = self.model_data.data[-1].shape[0]
        self.p = self.model_data.data[0].shape[1]
        self.q = self.model_data.data[1].shape[1]
        self.X, self.Z, self.y, self.weights = self.model_data
        self.params_init = np.zeros(self.p+self.q)
        poiss_labels = [("Poiss",x) for x in self.model_data.columns[0].to_list()]
        binom_labels = [("Binom",x) for x in self.model_data.columns[1].to_list()]
        labels = poiss_labels + binom_labels
        labels = pd.MultiIndex.from_tuples(labels, names=["ModelComponent", "Parameter"])
        self.param_labels = labels#[f"beta{i}" for i in range(1, self.p+1)]+[f"gamma{i}" for i in range(1, self.q+1)]
        self.design_info = self.model_data.design_info
    @staticmethod
    def _loglike_i(params, data, link, zero_link):
        X, Z, y, w = data
        eta1, eta2 = X.dot(params[:X.shape[1]]), Z.dot(params[X.shape[1]:])
        mu1, mu2 = link.inv_link(eta1), zero_link.inv_link(eta2)
        ll = np.zeros(len(y))
        ix = y==0
        ll[ix] = -w[ix] * np.log(mu2[ix] + (1 - mu2[ix]) * np.exp(-mu1[ix]))
        ix = ~ix
        ll[ix] = -w[ix] * (np.log(1 - mu2[ix]) + y[ix] * np.log(mu1[ix]) \
                          - mu1[ix] - sp.special.gammaln(y[ix]))
        return ll
    
    def loglike_i(self, params, data=None, link=None, zero_link=None):
        data = self.model_data if data is None else data 
        link = self.link if link is None else link
        zero_link = self.zero_link if zero_link is None else zero_link
        ll = self._loglike_i(params=params, data=data, link=link, zero_link=zero_link)
        return ll
    
    @staticmethod
    def _loglike(params, data, link, zero_link):
        X, Z, y, w = data
        p = X.shape[1]
        lambd = link.inv_link(X.dot(params[:p]))
        omega = zero_link.inv_link(Z.dot(params[p:]))
        ix = y==0
        ll = np.sum(-w[ix] * np.log(omega[ix] + (1 - omega[ix]) * np.exp(-lambd[ix])))
        ix = ~ix
        ll = ll + np.sum(-w[ix] * (np.log(1 - omega[ix]) \
                         + y[ix] * np.log(lambd[ix]) - lambd[ix]\
                         - sp.special.gammaln(y[ix])))

        return ll
    
    def loglike(self, params, data=None, link=None, zero_link=None):
        data = self.model_data if data is None else data 
        link = self.link if link is None else link
        zero_link = self.zero_link if zero_link is None else zero_link
        ll = self._loglike(params=params, data=data, link=link, zero_link=zero_link)
        return ll
    

    @staticmethod
    def _gradient(params, data, link, zero_link):
        X, Z, y, w = data
        n_poisson_params = X.shape[1]
        
        # Compute lambda_lp, omega_lp, lambd, and omega
        lambda_lp = X.dot(params[:n_poisson_params])  # FLOPs: n_poisson_params * len(y)
        omega_lp = Z.dot(params[n_poisson_params:])   # FLOPs: n_bernoulli_params * len(y)
        lambd = link.inv_link(lambda_lp)              # FLOPs: len(y)
        omega = zero_link.inv_link(omega_lp)          # FLOPs: len(y)
        
        # Compute the gradient for the Poisson part
        exlam = np.exp(-lambd)                          # FLOPs: len(y)
        ix_zero = y == 0
        u_poisson = np.zeros(len(y))
         
        t0 = (1 - omega[ix_zero])
        t1 = t0 * exlam[ix_zero]
        h1 = zero_link.dinv_link(omega_lp[ix_zero])
        den = (t1 + omega[ix_zero]) 

        u_poisson[ix_zero] = (w[ix_zero] * t1 * link.dinv_link(lambda_lp[ix_zero])) / den
        u_poisson[~ix_zero] =(w[~ix_zero] * (lambd[~ix_zero] - y[~ix_zero]) * link.dinv_link(lambda_lp[~ix_zero]) / lambd[~ix_zero])
        g_poisson = np.dot(X.T, u_poisson)             # FLOPs: n_poisson_params * len(y)
        
        # Compute the gradient for the Bernoulli part
        u_bernoulli = np.zeros(len(y))
        u_bernoulli[ix_zero] = -(w[ix_zero] * h1 - h1 * exlam[ix_zero]) / den
        u_bernoulli[~ix_zero] = w[~ix_zero] * zero_link.dinv_link(omega_lp[~ix_zero]) / (1.0 - omega[~ix_zero])
        g_bernoulli = np.dot(Z.T, u_bernoulli)         # FLOPs: n_bernoulli_params * len(y)
        
        # Combine the gradients
        gradient = np.concatenate((g_poisson, g_bernoulli))
        
        # Memory usage (assuming 8 bytes per float):
        # lambda_lp, omega_lp, lambd, omega, exlam: 5 * len(y) * 8 bytes
        # ix_zero: len(y) bytes
        # u_poisson, u_bernoulli, g_poisson, g_bernoulli: 2 * (n_poisson_params + n_bernoulli_params) * 8 bytes
        # den: len(y) * 8 bytes
        # gradient: n_params * 8 bytes
        # Total memory used: (11 * len(y) + 3 * n_params) * 8 bytes
        # Cumulative memory used: Same as total memory used, since no memory is freed during computation.
        
        return gradient
    
    def gradient(self, params, data=None, link=None, zero_link=None):
        data = self.model_data if data is None else data 
        link = self.link if link is None else link
        zero_link = self.zero_link if zero_link is None else zero_link
        g = self._gradient(params=params, data=data, link=link, zero_link=zero_link)
        return g

    
    @staticmethod
    def _gradient_i(params, data, link, zero_link):
        X, Z, y, w = data
        ixb, ixg = np.arange(X.shape[1]), np.arange(X.shape[1], len(params))
        lambda_lp, omega_lp = X.dot(params[:X.shape[1]]), Z.dot(params[X.shape[1]:])
        lambd, omega = link.inv_link(lambda_lp), zero_link.inv_link(omega_lp)
        exlam = np.exp(-lambd)
        g = np.zeros((len(y), len(params)))
        ix = y==0
        
        
        i1, = np.where(ix)
        i2, = np.where(~ix)
        i1, i2, j1, j2 = i1[:, None], i2[:, None], ixb[None], ixg[None]
        
        t0 = (1 - omega[ix])
        t1 = t0 * exlam[ix]
        h1 = zero_link.dinv_link(omega_lp[ix])
        gw1 =  (w[ix] * t1 * link.dinv_link(lambda_lp[ix])) / (t1 + omega[ix]) 
        gw2 = -(w[ix] * h1 - h1 * exlam[ix]) / (t1 + omega[ix])

        g[i1, j1] = X[ix] * (gw1)[:, None]
        g[i1, j2] = Z[ix] * (gw2)[:, None]
        
        ix = ~ix
        g[i2, j1] = X[ix] * (w[ix] * (lambd[ix] - y[ix]) * link.dinv_link(lambda_lp[ix]) / lambd[ix])[:, None]
        g[i2, j2] = Z[ix] * (w[ix] * zero_link.dinv_link(omega_lp[ix])/ (1.0 - omega[ix]))[:, None]
        return g
        
    
    def gradient_i(self, params, data=None, link=None, zero_link=None):
        data = self.model_data if data is None else data 
        link = self.link if link is None else link
        zero_link = self.zero_link if zero_link is None else zero_link
        g = self._gradient_i(params=params, data=data, link=link, zero_link=zero_link)
        return g
    
        
    @staticmethod
    def _hessian_weights(params, data, link, zero_link):
        X, Z, y, w = data
        n, p = X.shape
        
        eta = X.dot(params[:p])
        psi = Z.dot(params[p:])
        lambd = link.inv_link(eta)
        omega = zero_link.inv_link(psi)
        
        exlam = np.exp(-lambd)
        ix_zero = y == 0
        ix_nnz = ~ix_zero
        
        gp1  = link.dinv_link(eta[ix_zero])
        gpp1 = link.d2inv_link(eta[ix_zero])

        gp2 = link.dinv_link(eta[ix_nnz])
        gpp2 = link.d2inv_link(eta[ix_nnz])
        
        hp1 = zero_link.dinv_link(psi[ix_zero])
        hp2 = zero_link.dinv_link(psi[ix_nnz])
        
        hpp1 = zero_link.d2inv_link(psi[ix_zero])
        hpp2 = zero_link.d2inv_link(psi[ix_nnz])
        
        
        tmp1 = (1 - omega[ix_zero]) * exlam[ix_zero]
        tmp2 = tmp1 + omega[ix_zero]
        tmp3 =  exlam[ix_zero] * hp1 - hp1
        
        num0 = gpp1 * w[ix_zero] * tmp1
        num1 = gp1**2 * w[ix_zero] * tmp1
        den1 = omega[ix_zero] + tmp1
        den2 = den1 * den1
        
        weta2, wpsi2, wetapsi = np.zeros(n), np.zeros(n), np.zeros(n)
        
        w1eta2 = num0 / den1 + (num1 * tmp1) / den2 - num1 / den1
        num1 = w[ix_zero] * (hpp1 - exlam[ix_zero] * hpp1)
        num2 = w[ix_zero] * tmp3 * -tmp3    
        w1psi2 = -num1 / den1 - num2 / den2
        
        
        num1 = w[ix_zero] * tmp1 * tmp3  * gp1
        den2 =  tmp2
        den1 = den2 * den2
        num2 = w[ix_zero] * exlam[ix_zero] * gp1 * hp1
        
        w1etapsi = num1 / den1 - num2 / den2

        tmp0 = (1 - omega[ix_nnz])
        w2eta2 = -w[ix_nnz] * (-gpp2 + gpp2 * y[ix_nnz] / lambd[ix_nnz] - gp2*gp2*y[ix_nnz]/lambd[ix_nnz]**2)
        w2psi2 =  w[ix_nnz] * (hpp2 / tmp0 + hp2**2 / (tmp0 * tmp0))
        
        weta2[ix_zero] = w1eta2
        weta2[ix_nnz] = w2eta2
        
        wpsi2[ix_zero] = w1psi2
        wpsi2[ix_nnz] = w2psi2
        
        wetapsi[ix_zero] = w1etapsi       
        
        return weta2, wpsi2, wetapsi
    
    @staticmethod
    def _hessian(params, data, link, zero_link):
        X, Z, y, w = data
        n_params = len(params)
        n, p = X.shape
        
        eta = X.dot(params[:p])
        psi = Z.dot(params[p:])
        lambd = link.inv_link(eta)
        omega = zero_link.inv_link(psi)
        
        exlam = np.exp(-lambd)
        ix_zero = y == 0
        ix_nnz = ~ix_zero
        H = np.zeros((n_params, n_params))
        
        gp1  = link.dinv_link(eta[ix_zero])
        gpp1 = link.d2inv_link(eta[ix_zero])

        gp2 = link.dinv_link(eta[ix_nnz])
        gpp2 = link.d2inv_link(eta[ix_nnz])
        
        hp1 = zero_link.dinv_link(psi[ix_zero])
        hp2 = zero_link.dinv_link(psi[ix_nnz])
        
        hpp1 = zero_link.d2inv_link(psi[ix_zero])
        hpp2 = zero_link.d2inv_link(psi[ix_nnz])
        
        
        tmp1 = (1 - omega[ix_zero]) * exlam[ix_zero]
        tmp2 = tmp1 + omega[ix_zero]
        tmp3 =  exlam[ix_zero] * hp1 - hp1
        
        num0 = gpp1 * w[ix_zero] * tmp1
        num1 = gp1**2 * w[ix_zero] * tmp1
        den1 = omega[ix_zero] + tmp1
        den2 = den1 * den1
        
        weta2, wpsi2, wetapsi = np.zeros(n), np.zeros(n), np.zeros(n)
        
        w1eta2 = num0 / den1 + (num1 * tmp1) / den2 - num1 / den1
        num1 = w[ix_zero] * (hpp1 - exlam[ix_zero] * hpp1)
        num2 = w[ix_zero] * tmp3 * -tmp3    
        w1psi2 = -num1 / den1 - num2 / den2
        
        
        num1 = w[ix_zero] * tmp1 * tmp3  * gp1
        den2 =  tmp2
        den1 = den2 * den2
        num2 = w[ix_zero] * exlam[ix_zero] * gp1 * hp1
        
        w1etapsi = num1 / den1 - num2 / den2

        tmp0 = (1 - omega[ix_nnz])
        w2eta2 = -w[ix_nnz] * (-gpp2 + gpp2 * y[ix_nnz] / lambd[ix_nnz] - gp2*gp2*y[ix_nnz]/lambd[ix_nnz]**2)
        w2psi2 =  w[ix_nnz] * (hpp2 / tmp0 + hp2**2 / (tmp0 * tmp0))
        
        weta2[ix_zero] = w1eta2
        weta2[ix_nnz] = w2eta2
        
        wpsi2[ix_zero] = w1psi2
        wpsi2[ix_nnz] = w2psi2
        
        wetapsi[ix_zero] = w1etapsi
        
        H[:p, :p] = np.dot((X * weta2[:, None]).T, X)
        H[p:, :p] = np.dot((Z * wetapsi[:, None]).T, X)
        H[p:, p:] = np.dot((Z * wpsi2[:, None]).T, Z)
        H[:p, p:] = H[p:, :p].T
        
        return H
        
    def hessian(self, params, data=None, link=None, zero_link=None):
        data = self.model_data if data is None else data 
        link = self.link if link is None else link
        zero_link = self.zero_link if zero_link is None else zero_link
        H = self._hessian(params=params, data=data, link=link, zero_link=zero_link)
        return H

    def _fit(self, params=None, data=None, link=None, zero_link=None, 
             opt_kws=None):
        params = self.params_init.copy() if params is None else params
        opt_kws = {} if opt_kws is None else opt_kws
        default_kws = dict(method='trust-constr', options=dict(verbose=0, gtol=1e-6, xtol=1e-6))
        opt_kws = {**default_kws, **opt_kws}
        link = self.link if link is None else link
        zero_link = self.zero_link if zero_link is None else zero_link
        data = self.model_data if data is None else data 
        args = (data, link, zero_link)
        opt = sp.optimize.minimize(self.loglike, 
                                   jac=self.gradient,
                                   hess=self.hessian,
                                   x0=params,
                                   args=args,
                                   **opt_kws)
        params = opt.x
        return params, opt
        
    def fit(self, opt_kws=None):
        self.params, self.opt = self._fit(opt_kws)
        n, n_params = self.n, len(self.params)
        self.n_params = n_params
        self.params_hess = self.hessian(self.params)
        self.params_cov = np.linalg.pinv(self.params_hess)
        self.params_se = np.sqrt(np.diag(self.params_cov))
        self.res = self._parameter_inference(self.params, self.params_se,
                                             n-n_params,
                                             self.param_labels)
        self.beta_cov = self.coefs_cov = self.params_cov
        self.beta_se = self.coefs_se = self.params_se
        self.beta = self.coefs = self.params
        intercept = np.ones((self.n_obs, 1))
        self.null_data = ModelData(intercept, intercept, self.y, self.weights)
        self.null_params, self.null_opt = self._fit(params=np.zeros(2), 
                                          data=self.null_data, 
                                          )
        self.llf = self.loglike(self.params)
        self.lln = self.loglike(self.null_params, data=self.null_data)
        k = len(self.params)
        sumstats = {}
        self.aic, self.aicc, self.bic, self.caic = self._get_information(
            self.llf, k, self.n_obs)
        r2_cs, r2_nk, r2_mc, r2_mb, r2_es, r2_ea, r2_an, r2_vz, llr = \
            self._get_pseudo_rsquared(self.llf, self.lln, k, self.n_obs)
        self.r2_cs, self.r2_nk, self.r2_mc = r2_cs, r2_nk, r2_mc
        self.r2_mb, self.r2_es, self.r2_ea = r2_mb, r2_es, r2_ea
        self.r2_an, self.r2_vz, self.llr = r2_an, r2_vz, llr
        sumstats["AIC"] = self.aic
        sumstats["AICC"] = self.aicc
        sumstats["BIC"] = self.bic
        sumstats["CAIC"] = self.caic
        sumstats["R2_CS"] = self.r2_cs
        sumstats["R2_NK"] = self.r2_nk
        sumstats["R2_MC"] = self.r2_mc
        sumstats["R2_MB"] = self.r2_mb
        sumstats["R2_ES"] = self.r2_es
        sumstats["R2_EA"] = self.r2_ea
        sumstats["R2_AN"] = self.r2_an
        sumstats["R2_VZ"] = self.r2_vz
        sumstats["LLR"] = self.llr
        sumstats["LLF"] = self.llf
        sumstats["LLN"] = self.lln
        self.sumstats = pd.DataFrame(sumstats, index=["Statistic"]).T
                
        
