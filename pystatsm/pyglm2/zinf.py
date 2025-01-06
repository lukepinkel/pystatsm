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
from .families import NegativeBinomial
from ..utilities.func_utils import symmetric_conf_int, handle_default_kws

from ..utilities.linalg_operations import wdiag_outer_prod, wls_qr, nwls   





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
        self.params, self.opt = self._fit(opt_kws=opt_kws)
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
    
    @staticmethod
    def _predict(params, data, link, zero_link, phi=1.0, dispersion=1.0, coefs_cov=None, 
                 linpred=True, linpred_se=True, mean=True, mean_ci=True, mean_ci_level=0.95,
                 predicted_ci=True, predicted_ci_level=0.95):
        X, Z, y, w = data
        n_poisson_params = X.shape[1]
        
        # Compute lambda_lp, omega_lp, lambd, and omega
        lambda_lp = X.dot(params[:n_poisson_params])  # FLOPs: n_poisson_params * len(y)
        omega_lp = Z.dot(params[n_poisson_params:])   # FLOPs: n_bernoulli_params * len(y)
        
                 # FLOPs: len(y)
        coefs1_cov = coefs_cov[:n_poisson_params, :n_poisson_params]
        coefs2_cov = coefs_cov[n_poisson_params:, n_poisson_params:]

        res = {}        
        if linpred_se or mean_ci:
            lambda_lp_se = np.sqrt(wdiag_outer_prod(X, coefs1_cov, X))
            omega_lp_se = np.sqrt(wdiag_outer_prod(Z, coefs2_cov, Z))

        if mean or mean_ci or predicted_ci:
            lambd = link.inv_link(lambda_lp)              # FLOPs: len(y)
            omega = zero_link.inv_link(omega_lp) 

        if linpred:
            res["lp"] = np.vstack([lambda_lp, omega_lp]).T
        if linpred_se:
            res["lp_se"] = np.vstack([lambda_lp_se, omega_lp_se]).T
        if mean or mean_ci or predicted_ci:
            res["mean"] = np.vstack([lambd, omega]).T

        if mean_ci:
            mean_ci_level = symmetric_conf_int(mean_ci_level)
            mean_ci_lmult = sp.special.ndtri(mean_ci_level)
            res["lp_lower_ci"] = res["lp"] - mean_ci_lmult * res["lp_se"]
            res["lp_upper_ci"] = res["lp"] + mean_ci_lmult * res["lp_se"]
            res["mu_lower_ci"] = np.vstack([
                link.inv_link(res["lp_lower_ci"][:, 0]),
                zero_link.inv_link(res["lp_lower_ci"][:, 1]),
                              ]).T
            res["mu_upper_ci"] = np.vstack([
                link.inv_link(res["lp_upper_ci"][:, 0]),
                zero_link.inv_link(res["lp_upper_ci"][:, 1]),
                              ]).T
        # if predicted_ci:
        #     predicted_ci_level = symmetric_conf_int(predicted_ci_level)
        #     v = .variance(mu=mu, phi=phi, dispersion=dispersion)
        #     res["predicted_lower_ci"] = f.ppf(
        #         1-predicted_ci_level, mu=mu, scale=v)
        #     res["predicted_upper_ci"] = f.ppf(
        #         predicted_ci_level, mu=mu, scale=v)
        return res










class ZeroInflatedModel(RegressionMixin, LikelihoodModel):
    
    def __init__(self, 
                 formula=None,
                 zero_formula=None, 
                 data=None,
                 family=NegativeBinomial,
                 link=LogitLink,
                 X=None,
                 Z=None, 
                 y=None, 
                 weights=None,
                 *args,
                 **kwargs):
        
        formulas = [formula, zero_formula] if zero_formula else [formula]
        Xs = [X, Z] if Z is not None else [X]
        super().__init__(formula=formulas, data=data, X=Xs, y=y, weights=weights, 
                         *args, **kwargs)
        if isinstance(family, ExponentialFamily) is False:
            try:
                family = family()
            except TypeError:
                pass

        self.f = family
        self.link = link() if isinstance(link, Link) is False else link
        self.n = self.n_obs = self.model_data.data[-1].shape[0]
        self.p = self.model_data.data[0].shape[1]
        self.q = self.model_data.data[1].shape[1]
        self.X, self.Z, self.y, self.weights = self.model_data
        self.has_scale = self.f.name == "NegativeBinomial"
        self.n_params = self.p + self.q
        if self.has_scale:
           self.n_params += 1
       
        self.params_init = np.zeros(self.n_params)
        labels = [(self.f.name,x) for x in self.model_data.columns[0].to_list()]
        labels.extend([("Binom",x) for x in self.model_data.columns[1].to_list()])
        if self.has_scale:
            labels.append((self.f.name, "Scale"))
        labels = pd.MultiIndex.from_tuples(labels, names=["ModelComponent", "Parameter"])
        self.param_labels = labels
        self.design_info = self.model_data.design_info
        
    @staticmethod
    def _unpack_params(params, p, q, has_scale):
        b1, b2 = params[:p], params[p:p+q]
        k = np.exp(params[-1]) if has_scale else 1.0
        return b1, b2, k
    
    @staticmethod
    def _unpack_params_data(params, data, p, q, has_scale, f, link):
        X, Z, y, w = data
        b1, b2 = params[:p], params[p:p+q]
        k = np.exp(params[-1]) if has_scale else 1.0
        eta1, eta2 = np.dot(X, b1), np.dot(Z, b2)
        mu1, mu2 = f.inv_link(eta1), link.inv_link(eta2)
        return X, Z, y, w, eta1, eta2, mu1, mu2, b1, b2, k
    
    @staticmethod
    def _generic_loglike_i(params, data, p, q, has_scale, f, link):
        X, Z, y, w = data
        b1, b2 = params[:p], params[p:p+q]
        k = np.exp(params[-1]) if has_scale else 1.0
        eta1, eta2 = np.dot(X, b1), np.dot(Z, b2)
        mu1, mu2 = f.inv_link(eta1), link.inv_link(eta2)
        lnp = -f._full_loglike(y, weights=w, mu=mu1, phi=1.0, dispersion=k)
        ll = np.zeros(len(y), dtype=params.dtype)
        
        i = y==0
        ll[i] = -np.log(mu2[i]+(1-mu2[i])*np.exp(lnp[i]))
        i=~i
        ll[i] = -lnp[i] - np.log(1.0 - mu2[i]) * w[i]
        return ll
        
    

class ZeroInflatedNBinom(ZeroInflatedModel):
    
    def __init__(self, 
                 formula=None,
                 zero_formula=None, 
                 data=None,
                 nonzero_link=LogLink,
                 zero_link=LogitLink,
                 X=None,
                 Z=None, 
                 y=None, 
                 weights=None,
                 *args,
                 **kwargs):
        family = NegativeBinomial(link=nonzero_link)
        super().__init__(
                         formula=formula,
                         zero_formula=zero_formula, 
                         data=data,
                         family=family,
                         link=zero_link,
                         X=X,
                         Z=Z, 
                         y=y, 
                         weights=weights,
                         *args,
                         **kwargs) 
    @staticmethod
    def _loglike_i(params, data, link, f):
        X, Z, y, w = data
        coefs, k = params[:-1], np.exp(params[-1])
        eta1, eta2 = X.dot(coefs[:X.shape[1]]), Z.dot(coefs[X.shape[1]:])
        mu1, mu2 = f.inv_link(eta1), link.inv_link(eta2)
        ll = np.zeros(len(y),dtype=params.dtype)
        ix = y==0
        ll[ix] = - np.log(mu2[ix] + (1-mu2[ix]) * (1 + k / w[ix] * mu1[ix])**(-w[ix] / k))
        ix = ~ix
        ll[ix] = f._loglike(y[ix], weights=w[ix], mu=mu1[ix], phi=1.0, dispersion=k) -np.log(1.0 - mu2[ix])  
        return ll
    
    def loglike_i(self, params, data=None, link=None, f=None):
        data = self.model_data if data is None else data 
        link = self.link if link is None else link
        f = self.f if f is None else f
        ll = self._loglike_i(params=params, data=data, link=link, f=f)
        return ll
    
    @staticmethod
    def _loglike(params, data, link, f):
      X, Z, y, w = data
      coefs, k = params[:-1], np.exp(params[-1])
      eta1, eta2 = X.dot(coefs[:X.shape[1]]), Z.dot(coefs[X.shape[1]:])
      mu1, mu2 = f.inv_link(eta1), link.inv_link(eta2)
      ll = np.zeros(len(y), dtype=params.dtype)
      ix = y==0
      ll[ix] = - np.log(mu2[ix] + (1-mu2[ix]) * (1 + k / w[ix] * mu1[ix])**(-w[ix] / k))
      ix = ~ix
      ll[ix] = f._full_loglike(y[ix], weights=w[ix], mu=mu1[ix], phi=1.0, dispersion=k) -np.log(1.0 - mu2[ix]) 
      return np.sum(ll)
    
    def loglike(self, params, data=None, link=None, f=None):
        data = self.model_data if data is None else data 
        link = self.link if link is None else link
        f = self.f if f is None else f
        ll = self._loglike(params=params, data=data, link=link, f=f)
        return ll
    

    @staticmethod
    def _gradient(params, data, link, f):
        X, Z, y, w = data
        p, q = X.shape[1], Z.shape[1]
        coefs, k = params[:-1], np.exp(params[-1])
        # Compute lambda_lp, omega_lp, lambd, and omega
        eta1 = X.dot(coefs[:p])  
        eta2 = Z.dot(coefs[p:])  
        mu1 = f.inv_link(eta1)             
        mu2 = link.inv_link(eta2)         
        d0 = np.zeros_like(y, dtype=params.dtype)
        d1 = np.zeros_like(y, dtype=params.dtype)
        d2 = np.zeros_like(y, dtype=params.dtype)
        gradient = np.zeros_like(params)
        
        i = y == 0
        
        x0 = mu1[i]
        x1 = mu2[i]
        x2 = x0 * k
        x3 = 1.0 * w[i] + x2
        x4 = x3 / w[i]
        x5 = w[i] / k
        x6 = x4 ** x5
        x7 = 1.0 / (x1 * x6 - x1 + 1.0)
        x8 = x7 * (x1 - 1.0) / x3
        
        d0[i] = -w[i] * x8 * f.dinv_link(eta1[i])
        d1[i] = x7 * (x6 - 1.0) *  link.dinv_link(eta2[i])
        d2[i] = x5 * x8 * (-x2 + x3 * np.log(x4))
        
        i = y != 0
        
        x0 = mu1[i]
        x1 = k
        x2 = x0 * x1
        x3 = x2 + 1.0
        x4 = w[i] / x3
        x5 = mu2[i]
        x6 = 1 / k
        x7 = x1 * y[i]
        x8 = x7 + 1.0
        #x9 = np.log(x3) +  sp.special.polygamma(0, x6) - sp.special.polygamma(0, x6 * x8)
        x9 = np.log(x3) +  sp.special.digamma(x6) - sp.special.digamma(x6 * x8)
        d0[i] = x4 * (x0 - y[i]) *  f.dinv_link(eta1[i]) / x0
        d1[i] = link.dinv_link(eta2[i]) / (x5 - 1.0)
        d2[i] = -x4 * x6 * (-x2 * x8 + x3 * x7 + x3 * x9)
                               
        g1 = np.dot(X.T, d0)
        g2 = np.dot(Z.T, d1)
        g3 = np.sum(d2)
        gradient[:p] = g1
        gradient[p:p+q] = -g2
        gradient[-1] = g3
        return gradient
    
    def gradient(self, params, data=None, link=None, f=None):
        data = self.model_data if data is None else data 
        link = self.link if link is None else link
        f = self.f if f is None else f
        g = self._gradient(params=params, data=data, link=link, f=f)
        return g

    
    @staticmethod
    def _gradient_i(params, data, link, f):
        X, Z, y, w = data
        p, q = X.shape[1], Z.shape[1]
        coefs, k = params[:-1], np.exp(params[-1])
        # Compute lambda_lp, omega_lp, lambd, and omega
        eta1 = X.dot(coefs[:p])  
        eta2 = Z.dot(coefs[p:])  
        mu1 = f.inv_link(eta1)             
        mu2 = link.inv_link(eta2)     
        dmu1 = f.dinv_link(eta1)             
        dmu2 = link.dinv_link(eta2)      
        d0 = np.zeros_like(y, dtype=params.dtype)
        d1 = np.zeros_like(y, dtype=params.dtype)
        d2 = np.zeros_like(y, dtype=params.dtype)
        
        i = y == 0
        
        x0 = mu1[i]
        x1 = 1.0 + x0*k/w[i]
        x2 = 1/x1
        x3 = mu2[i]
        x4 = w[i]*(1/k)
        x5 = x1**(-x4)
        x6 = x5*(1.0 - x3)
        x7 = 1/(x3 + x6)
        x8 = x6*x7
        x9 = dmu2[i]
        
        d0[i] = x2*x8*dmu1[i]
        d1[i] = -x7*(-x5*x9 + x9)
        d2[i] = -x8*(-x0*x2 + x4*np.log(x1))
        
        # x0 = mu1[i]
        # x1 = mu2[i]
        # x2 = x0 * k
        # x3 = 1.0 * w[i] + x2
        # x4 = x3 / w[i]
        # x5 = w[i] / k
        # x6 = x4 ** x5
        # x7 = 1.0 / (x1 * x6 - x1 + 1.0)
        # x8 = x7 * (x1 - 1.0) / x3
        
        # d0[i] = -w[i] * x8 * f.dinv_link(eta1[i])
        # d1[i] = x7 * (x6 - 1.0) *  link.dinv_link(eta2[i])
        # d2[i] = x5 * x8 * (-x2 + x3 * np.log(x4))
        
        i = y != 0
        
        # x0 = mu1[i]
        # x1 = k
        # x2 = x0 * x1
        # x3 = x2 + 1.0
        # x4 = w[i] / x3
        # x5 = mu2[i]
        # x6 = 1 / k
        # x7 = x1 * y[i]
        # x8 = x7 + 1.0
        # x9 = np.log(x3) +  sp.special.polygamma(0, x6) - sp.special.polygamma(0, x6 * x8)
        
        # d0[i] += x4 * (x0 - y[i]) *  f.dinv_link(eta1[i]) / x0
        # d1[i] += link.dinv_link(eta2[i]) / (x5 - 1.0)
        # d2[i] += -x4 * x6 * (-x2 * x8 + x3 * x7 + x3 * x9)
        x0 = mu1[i]
        x1 = dmu1[i]
        x2 = k
        x3 = 1/w[i]
        x4 = (1/k)
        x5 = w[i]*x4
        x6 = x5 + y[i]
        x7 = x0*x2*x3 + 1
        x8 = 1/x7
        x9 = mu2[i]
        d0[i] += x1*x2*x3*x6*x8 - x1*y[i]/x0
        d1[i] += dmu2[i]/(1 - x9)
        d2[i] += w[i]*x4*sp.special.polygamma(0, x6) + x0*x2*x3*x6*x8 - \
                x5*np.log(x7) - x5*sp.special.polygamma(0, x5) - y[i]
                                       
        g1 = d0.reshape(-1, 1) * X
        g2 = d1.reshape(-1, 1) * Z
        g3 = d2.reshape(-1, 1)
        gradient = np.concatenate([g1, g2, g3], axis=1)
        return gradient
        
    
    def gradient_i(self, params, data=None, link=None, f=None):
        data = self.model_data if data is None else data 
        link = self.link if link is None else link
        f = self.f if f is None else f
        g = self._gradient_i(params=params, data=data, link=link, f=f)
        return g
    
        
    @staticmethod
    def _hessian_i(params, data, link,f):
        X, Z, y, w = data
        p, q = X.shape[1], Z.shape[1]
        coefs, k = params[:-1], np.exp(params[-1])
        eta1 = X.dot(coefs[:p])  
        eta2 = Z.dot(coefs[p:])  
        mu1 = f.inv_link(eta1) 
        d1mu1 = f.dinv_link(eta1)      
        d2mu1 = f.d2inv_link(eta1)            
        d1mu2 = link.dinv_link(eta2)      
        d2mu2 = link.d2inv_link(eta2)        
        h0 = np.zeros((6, X.shape[0]), dtype=params.dtype)
        h1 = np.zeros((6, X.shape[0]), dtype=params.dtype)

        mu2 = link.inv_link(eta2)         
        
        i = y == 0
        
        x0 = mu1[i]
        x1 = d1mu1[i]
        x2 = 1 / w[i]
        x3 = k
        x4 = x2 * x3
        x5 = x0 * x4 + 1.0
        x6 = 1.0 / x5
        x7 = x1 * x1 * x6
        x8 = mu2[i]
        x9 = x8 - 1.0
        x10 = w[i] / k
        x11 = x5 ** - x10
        x12 = x11 * x9
        x13 = 1.0 / (x8 - x12)
        x14 = x12 * x13
        x15 = x14 * x6
        x16 = x13 * (1.0 - x11)
        x17 = x16 * x9 - 1.0
        x18 = d1mu2[i]
        x19 = x11 * x13 * x18
        x20 = x0 * x6
        x21 = x10 * np.log(x5)
        x22 = x21 - x20
        x23 = x22 * x22
        
        h0[0,i] = x15 * (x14 * x7 + x4 * x7 + x7 - d2mu1[i])
        h0[1,i] = x1 * x17 * x19 * x6
        h0[2,i] = x1 * x15 * (x0 *x2 * x3 * x6 - x14  * x22 - x22)
        h0[3,i] = x16 * (x16 * x18 * x18 - d2mu2[i])
        h0[4,i] = -x17 * x19 * x22
        h0[5,i] = x14 * (x0 * x0 * x4 / (x5 * x5) + x14 * x23 + x20 - x21 + x23)
        i = ~i
        
        
        x0 = mu1[i]
        x1 = d2mu1[i]
        x2 = x0**2
        x3 = d1mu1[i]
        x4 = x3**2
        x5 = k
        x6 = 1/w[i]
        x7 = w[i] / k
        x8 = x7 + y[i]
        x9 = x0*x5*x6
        x10 = x9 + 1
        x11 = 1/x10
        x12 = w[i]**2
        x13 = 2 * np.log(k)
        x14 = x8 * np.exp(x13)/x12
        x15 = x14 / x10**2
        x16 = x0 * x11
        x17 = mu2[i]
        x18 = 1/(x17 - 1)
        x19 = x12 * np.exp(-x13)
        
        h1[0, i] = x1 * x11 * x5 * x6 * x8 - x15 * x4 + x4 * y[i] / x2 - x1 * y[i] / x0
        h1[2, i] = x11 * x3 * (x5 * x6 * x8 - x14 * x16 - 1)
        h1[3, i] = x18 * (x18 * d1mu2[i]**2 -  d2mu2[i])
        h1[5, i] = x11 * x8 * x9 - x15 * x2 - 2 * x16 + x19 * sp.special.polygamma(1, x7) \
            - x19 * sp.special.polygamma(1, x8) + x7 * np.log(x10) +\
                x7 * sp.special.polygamma(0, x7) - x7 * sp.special.polygamma(0, x8)
                
        h = h0 + h1        
      
        h00 = np.einsum("ij,ik->ijk", X*h[[0]].T, X)
        h10 = np.einsum("ij,ik->ijk", Z*h[[1]].T, X)
        h20 = np.einsum("ij,ik->ijk", h[[2]].T, X)
        h11 = np.einsum("ij,ik->ijk", Z*h[[3]].T, Z)
        h21 = np.einsum("ij,ik->ijk", h[[4]].T, Z)
        h22 = h[5].reshape(-1, 1, 1)
        
        H = np.zeros((X.shape[0], p+q+1, p+q+1), dtype=params.dtype)
        
        H[:, 0:p, 0:p] = h00
        H[:, p:p+q, 0:p] = h10
        H[:, p+q:, 0:p] = h20
        H[:, p:p+q, p:p+q] = h11
        H[:, p+q:, p:p+q] = h21
        H[:, p+q:, p+q:] = h22
        
        H[:, 0:p, p:p+q] = np.swapaxes(h10, 1, 2)
        H[:, 0:p, p+q:] = np.swapaxes(h20, 1, 2)
        H[:, p:p+q, p+q:] = np.swapaxes(h21, 1, 2)
        return H
            
    def hessian_i(self, params, data=None, link=None, f=None):
        data = self.model_data if data is None else data 
        link = self.link if link is None else link
        f = self.f if f is None else f
        g = self._hessian_i(params=params, data=data, link=link, f=f)
        return g
    @staticmethod
    def _hessian(params, data, link,f):
        X, Z, y, w = data
        p, q = X.shape[1], Z.shape[1]
        coefs, k = params[:-1], np.exp(params[-1])
        eta1 = X.dot(coefs[:p])  
        eta2 = Z.dot(coefs[p:])  
        mu1 = f.inv_link(eta1) 
        d1mu1 = f.dinv_link(eta1)      
        d2mu1 = f.d2inv_link(eta1)            
        d1mu2 = link.dinv_link(eta2)      
        d2mu2 = link.d2inv_link(eta2)        
        h0 = np.zeros((6, X.shape[0]), dtype=params.dtype)
        h1 = np.zeros((6, X.shape[0]), dtype=params.dtype)

        mu2 = link.inv_link(eta2)         
        
        i = y == 0
        
        x0 = mu1[i]
        x1 = d1mu1[i]
        x2 = 1 / w[i]
        x3 = k
        x4 = x2 * x3
        x5 = x0 * x4 + 1.0
        x6 = 1.0 / x5
        x7 = x1 * x1 * x6
        x8 = mu2[i]
        x9 = x8 - 1.0
        x10 = w[i] / k
        x11 = x5 ** - x10
        x12 = x11 * x9
        x13 = 1.0 / (x8 - x12)
        x14 = x12 * x13
        x15 = x14 * x6
        x16 = x13 * (1.0 - x11)
        x17 = x16 * x9 - 1.0
        x18 = d1mu2[i]
        x19 = x11 * x13 * x18
        x20 = x0 * x6
        x21 = x10 * np.log(x5)
        x22 = x21 - x20
        x23 = x22 * x22
        
        h0[0,i] = x15 * (x14 * x7 + x4 * x7 + x7 - d2mu1[i])
        h0[1,i] = x1 * x17 * x19 * x6
        h0[2,i] = x1 * x15 * (x0 *x2 * x3 * x6 - x14  * x22 - x22)
        h0[3,i] = x16 * (x16 * x18 * x18 - d2mu2[i])
        h0[4,i] = -x17 * x19 * x22
        h0[5,i] = x14 * (x0 * x0 * x4 / (x5 * x5) + x14 * x23 + x20 - x21 + x23)
        i = ~i
        
        
        x0 = mu1[i]
        x1 = d2mu1[i]
        x2 = x0**2
        x3 = d1mu1[i]
        x4 = x3**2
        x5 = k
        x6 = 1/w[i]
        x7 = w[i] / k
        x8 = x7 + y[i]
        x9 = x0*x5*x6
        x10 = x9 + 1
        x11 = 1/x10
        x12 = w[i]**2
        x13 = 2 * np.log(k)
        x14 = x8 * np.exp(x13)/x12
        x15 = x14 / x10**2
        x16 = x0 * x11
        x17 = mu2[i]
        x18 = 1/(x17 - 1)
        x19 = x12 * np.exp(-x13)
        
        h1[0, i] = x1 * x11 * x5 * x6 * x8 - x15 * x4 + x4 * y[i] / x2 - x1 * y[i] / x0
        h1[2, i] = x11 * x3 * (x5 * x6 * x8 - x14 * x16 - 1)
        h1[3, i] = x18 * (x18 * d1mu2[i]**2 -  d2mu2[i])
        h1[5, i] = x11 * x8 * x9 - x15 * x2 - 2 * x16 + x19 * sp.special.polygamma(1, x7) \
            - x19 * sp.special.polygamma(1, x8) + x7 * np.log(x10) +\
                x7 * sp.special.polygamma(0, x7) - x7 * sp.special.polygamma(0, x8)
                
        h = h0 + h1        
      
        h00 = np.einsum("ij,ik->jk", X*h[[0]].T, X)
        h10 = np.einsum("ij,ik->jk", Z*h[[1]].T, X)
        h20 = np.einsum("ij,ik->jk", h[[2]].T, X)
        h11 = np.einsum("ij,ik->jk", Z*h[[3]].T, Z)
        h21 = np.einsum("ij,ik->jk", h[[4]].T, Z)
        h22 = h[5].sum()
        
        H = np.zeros((p+q+1, p+q+1), dtype=params.dtype)
        
        H[0:p, 0:p] = h00
        H[p:p+q, 0:p] = h10
        H[p+q:, 0:p] = h20
        H[p:p+q, p:p+q] = h11
        H[p+q:, p:p+q] = h21
        H[p+q:, p+q:] = h22
        
        H[0:p, p:p+q] = np.swapaxes(h10, 0, 1)
        H[0:p, p+q:] = np.swapaxes(h20, 0, 1)
        H[p:p+q, p+q:] = np.swapaxes(h21, 0, 1) 
        return H
        
    def hessian(self, params, data=None, link=None, f=None):
        data = self.model_data if data is None else data 
        link = self.link if link is None else link
        f = self.f if f is None else f
        H = self._hessian(params=params, data=data, link=link, f=f)
        return H

    def _fit(self, params=None, data=None, f=None, link=None, 
             opt_kws=None):
        params = self.params_init.copy() if params is None else params
        opt_kws = {} if opt_kws is None else opt_kws
        default_kws = dict(method='trust-constr', options=dict(verbose=0, gtol=1e-6, xtol=1e-6))
        opt_kws = {**default_kws, **opt_kws}
        f = self.f if f is None else f
        link = self.link if link is None else link
        data = self.model_data if data is None else data 
        args = (data, link, f)
        opt = sp.optimize.minimize(self.loglike, 
                                   jac=self.gradient,
                                   hess=self.hessian,
                                   x0=params,
                                   args=args,
                                   **opt_kws)
        params = opt.x
        return params, opt
        
    def fit(self, opt_kws=None):
        self.params, self.opt = self._fit(opt_kws=opt_kws)
        n, n_params = self.n, len(self.params)
        self.n_params = n_params
        self.params_hess = self.hessian(self.params)
        self.params_cov = np.linalg.pinv(self.params_hess)
        self.params_se = np.sqrt(np.diag(self.params_cov))
        self.res = self._parameter_inference(self.params, self.params_se,
                                             n-n_params,
                                             self.param_labels)
        self.beta_cov = self.coefs_cov = self.params_cov[:-1, :-1]
        self.beta_se = self.coefs_se = self.params_se[:-1]
        self.beta = self.coefs = self.params[:-1]
        intercept = np.ones((self.n_obs, 1))
        self.null_data = ModelData(intercept, intercept, self.y, self.weights)
        self.null_params, self.null_opt = self._fit(params=np.zeros(3), 
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
    
    @staticmethod
    def _predict(params, data, link, f, params_cov=None, 
                 linpred=True, linpred_se=True, mean=True, mean_ci=True, mean_ci_level=0.95,
                 predicted_ci=True, predicted_ci_level=0.95):
        if params_cov is None:
            hess = ZeroInflatedNBinom._hessian(params, data, link, f)
            params_cov = np.linalg.pinv(hess)
            
        X, Z, y, w = data
        coefs, k = params[:-1], np.exp(params[-1])
        eta1, eta2 = X.dot(coefs[:X.shape[1]]), Z.dot(coefs[X.shape[1]:])

        p, q = X.shape[1], Z.shape[1]
        
        params_cov1 = params_cov[:p, :p]
        params_cov2 = params_cov[p:p+q, p:p+q]

        res = {}        
        if linpred_se or mean_ci:
            eta1_se = np.sqrt(wdiag_outer_prod(X, params_cov1, X))
            eta2_se = np.sqrt(wdiag_outer_prod(Z, params_cov2, Z))

        if mean or mean_ci or predicted_ci:
            mu1, mu2 = f.inv_link(eta1), link.inv_link(eta2)

        if linpred:
            res["lp"] = np.vstack([eta1, eta2]).T
        if linpred_se:
            res["lp_se"] = np.vstack([eta1_se, eta2_se]).T
        if mean or mean_ci or predicted_ci:
            res["mean"] = np.vstack([mu1, mu2]).T

        if mean_ci:
            mean_ci_level = symmetric_conf_int(mean_ci_level)
            mean_ci_lmult = sp.special.ndtri(mean_ci_level)
            res["lp_lower_ci"] = res["lp"] - mean_ci_lmult * res["lp_se"]
            res["lp_upper_ci"] = res["lp"] + mean_ci_lmult * res["lp_se"]
            res["mu_lower_ci"] = np.vstack([
                f.inv_link(res["lp_lower_ci"][:, 0]),
                link.inv_link(res["lp_lower_ci"][:, 1]),
                              ]).T
            res["mu_upper_ci"] = np.vstack([
                f.inv_link(res["lp_upper_ci"][:, 0]),
                link.inv_link(res["lp_upper_ci"][:, 1]),
                              ]).T
        if predicted_ci:
             predicted_ci_level = symmetric_conf_int(predicted_ci_level)
             v = f.variance(mu=mu1, phi=1.0, dispersion=k)
             res["predicted_lower_ci"] = f.ppf(
                 1-predicted_ci_level, mu=mu1, dispersion=v)
             res["predicted_upper_ci"] = f.ppf(
                 predicted_ci_level, mu=mu1, dispersion=v)
        return res  

    def predict(self, X=None, Z=None, params=None, params_cov=None, **kws):
        X = self.X if X is None else X
        Z = self.Z if Z is None else Z
        params = self.params if params is None else params
        params_cov = self.params_cov if params_cov is None else params_cov
        data = (X, Z, None, None)
        return self._predict(params, data, self.link, self.f, params_cov, **kws)
  

