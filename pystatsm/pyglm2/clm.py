#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 16:40:37 2023

@author: lukepinkel
"""

import tqdm
import numba
import numpy as np
import scipy as sp
import pandas as pd
from .regression_model import RegressionMixin, ModelData, OrdinalModelData
from .likelihood_model import LikelihoodModel
import scipy.stats
from ..utilities.data_utils import _check_type, _check_shape
from ..utilities.linalg_opertaions import wdiag_outer_prod
from ..utilities.optimizer_utils import process_optimizer_kwargs
from .links import LogitLink, ProbitLink, Link
from ..utilities.func_utils import symmetric_conf_int, handle_default_kws

from ..utilities import indexing_utils

class OrderedTransform(object):
    
    @staticmethod
    def _rvs(y, q):
        x = y.copy()
        x[1:q] = np.cumsum(np.exp(y[1:q]))+x[0]
        return x

    @staticmethod
    def _fwd(x, q):
        y = x.copy()
        y[1:q] = np.log(np.diff(x[:q]))
        return y
    
    
    @staticmethod
    def _jac_rvs(y, q):
        p = len(y)
        dx_dy = np.zeros((p, p))
        dx_dy[indexing_utils.tril_indices(q)] = np.repeat(np.r_[1, np.exp(y[1:q])], 
                                                          np.arange(q, 0, -1))
        return dx_dy

    @staticmethod
    def _hess_rvs(y, q):
        p = len(y)
        d2x_dy2 = np.zeros((p, p, p))
        z = np.exp(y)
        for i in range(1, q):
            for j in range(1, i+1):
                d2x_dy2[i, j, j] = z[j]
        return d2x_dy2
    

    @staticmethod
    def _jac_fwd(x, q):
        p = len(x)
        dy_dx = np.zeros((p, p))
        dy_dx[0, 0] = 1
        ii = np.arange(1, q)
        dy_dx[ii, ii] = 1 / (x[ii] - x[ii-1])
        dy_dx[ii, ii-1] =  -1 / (x[ii] - x[ii - 1])
        # for i in range(1, q):
        #     dy_dx[i, i] = 1 / (x[i] - x[i - 1])
        #     dy_dx[i, i - 1] = -1 / (x[i] - x[i - 1])
        return dy_dx
    
    @staticmethod
    def _hess_fwd(x, q):
        p = len(x)
        d2y_dx2 = np.zeros((p, p, p))
        for i in range(1, q):
            d2y_dx2[i, i, i] = -1 / (x[i] - x[i - 1])**2
            d2y_dx2[i, i - 1, i - 1] = -1 / (x[i] - x[i - 1])**2
            d2y_dx2[i, i, i - 1] = 1 / (x[i] - x[i - 1])**2
            d2y_dx2[i, i - 1, i] = d2y_dx2[i, i, i - 1]  # Use symmetry
        return d2y_dx2



class CLM(RegressionMixin, LikelihoodModel):
    
    
    def __init__(self, formula, data,X=None, y=None, weights=None, link=LogitLink,
                 *args,**kwargs):
        super().__init__(formula=formula, data=data, X=X, y=y, weights=weights, 
                         data_class=OrdinalModelData, *args, **kwargs)
        self.xinds, self.yinds = self.model_data.indexes
        self.xcols, self.ycols = self.model_data.columns
        self.X, self.y = self.model_data._regression_data
        self.n = self.n_obs = self.X.shape[0]
        self.p = self.n_var = self.X.shape[1]
        self.x_design_info, self.y_design_info = self.model_data.design_info
        self.formula = formula
        self.beta_labels = list(self.xcols)
        self.A1, self.A2 =  self.model_data.A1, self.model_data.A2
        self.o1, self.o2 =  self.model_data.o1, self.model_data.o2
        self.o1ix = self.model_data.o1ix
        self.weights = self.model_data.weights
        unique = self.model_data.unique
        if isinstance(link, Link) is False:
            link = link()
        self.link = link
        self.q = self.model_data.q
        self.tau_labels = [f"{unique[i-1]}|{unique[i]}" for i in range(1, self.q+1)]
        self.param_labels = self.tau_labels+self.beta_labels
        params_init = np.r_[self.model_data.tau_init,  np.zeros(self.p)]
        self.params_init = OrderedTransform._fwd(params_init, self.q)
        
        
    
    @staticmethod
    def _loglike_unconstrained(params, data, link, q):
        params = OrderedTransform._rvs(params, q)
        B1, B2, o1, o2, o1ix, w = data
        eta1, eta2 = B1.dot(params) + o1, B2.dot(params) + o2
        mu1, mu2 = link.inv_link(eta1), link.inv_link(eta2)
        prob = mu1 - mu2
        ll = -np.sum(w * np.log(prob))
        return ll
    
    def loglike_unconstrained(self, params, data=None, link=None, q=None):
        link = self.link if link is None else link
        data = self.model_data if data is None else data 
        q = self.q if q is None else q
        ll = self._loglike_unconstrained(params=params, data=data, link=link, q=q)
        return ll
    
    @staticmethod
    def _loglike(params, data, link, q):
        B1, B2, o1, o2, o1ix, w = data
        eta1, eta2 = B1.dot(params) + o1, B2.dot(params) + o2
        mu1, mu2 = link.inv_link(eta1), link.inv_link(eta2)
        prob = mu1 - mu2
        ll = -np.sum(w * np.log(prob))
        return ll
    
    def loglike(self, params, data=None, link=None, q=None):
        link = self.link if link is None else link
        data = self.model_data if data is None else data 
        q = self.q if q is None else q
        ll = self._loglike(params=params, data=data, link=link, q=q)
        return ll
    
    @staticmethod
    def _gradient_unconstrained(params, data, link, q):
        pars = OrderedTransform._rvs(params, q)
        B1, B2, o1, o2, o1ix, w = data
        
        eta1, eta2 = B1.dot(pars) + o1, B2.dot(pars) + o2
        mu1, mu2 = link.inv_link(eta1), link.inv_link(eta2)
        mu1[o1ix] = 1.0
        prob = mu1 - mu2
        
        d1eta1, d1eta2 = link.dinv_link(eta1), link.dinv_link(eta2)
        d1eta1[o1ix] = 0.0
        
        dprob = (B1 * d1eta1[:, None]).T - (B2 * d1eta2[:, None]).T
        g = -np.dot(dprob, w / prob)
        g[:q] = np.dot(g[:q], OrderedTransform._jac_rvs(params[:q], q))
        return g
        
    def gradient_unconstrained(self, params, data=None, link=None, q=None):
        link = self.link if link is None else link
        data = self.model_data if data is None else data 
        q = self.q if q is None else q
        g = self._gradient_unconstrained(params=params, data=data, link=link, q=q)
        return g
    
    @staticmethod
    def _gradient(params, data, link, q):
        B1, B2, o1, o2, o1ix, w = data
        
        eta1, eta2 = B1.dot(params) + o1, B2.dot(params) + o2
        mu1, mu2 = link.inv_link(eta1), link.inv_link(eta2)
        mu1[o1ix] = 1.0
        prob = mu1 - mu2
        
        d1eta1, d1eta2 = link.dinv_link(eta1), link.dinv_link(eta2)
        d1eta1[o1ix] = 0.0
        
        dprob = (B1 * d1eta1[:, None]).T - (B2 * d1eta2[:, None]).T
        g = -np.dot(dprob, w / prob)
        return g
        
    def gradient(self, params, data=None, link=None, q=None):
        link = self.link if link is None else link
        data = self.model_data if data is None else data 
        q = self.q if q is None else q
        g = self._gradient(params=params, data=data, link=link, q=q)
        return g
    
    @staticmethod
    def _hessian_unconstrained(params, data, link, q):
        pars = OrderedTransform._rvs(params, q)
        B1, B2,o1, o2, o1ix, w = data
        eta1, eta2 = B1.dot(pars), B2.dot(pars) + o2
        mu1, mu2 = link.inv_link(eta1+o1), link.inv_link(eta2)
        mu1[o1ix] = 1.0
        prob = mu1 - mu2
        
        
        d1eta1, d1eta2 = link.dinv_link(eta1+o1), link.dinv_link(eta2)
        d2eta1, d2eta2 = link.d2inv_link(eta1), link.d2inv_link(eta2)
        
         
        d1eta1[o1ix], d2eta1[o1ix] = 0.0, 0.0
        
        w3 =  w / prob**2
        dprob = (B1 * d1eta1[:, None]) - (B2 * d1eta2[:, None])
        T0 = (B1 * (d2eta1 / prob)[:, None]).T.dot(B1)
        T1 = (B2 * (d2eta2 / prob)[:, None]).T.dot(B2)
        T2 = (dprob * w3[:, None]).T.dot(dprob)
        H = -(T0 - T1 - T2)
        g = -np.dot(dprob[:, :q].T, w / prob)
        k = len(params)
        D1 = np.eye(k)
        D1[:q,:q] = OrderedTransform._jac_rvs(params[:q], q)
        D2 = OrderedTransform._hess_rvs(params[:q], q)
        H = D1.T.dot(H).dot(D1)
        H[:q, :q] = H[:q,:q] + np.einsum("i,ijk->jk", g, D2)
        return H
    
    def hessian_unconstrained(self, params, data=None, link=None, q=None):
        link = self.link if link is None else link
        data = self.model_data if data is None else data 
        q = self.q if q is None else q
        H = self._hessian_unconstrained(params=params, data=data, link=link, q=q)
        return H
      
    @staticmethod
    def _hessian(params, data, link, q):
        B1, B2,o1, o2, o1ix, w = data
        eta1, eta2 = B1.dot(params), B2.dot(params) + o2
        mu1, mu2 = link.inv_link(eta1+o1), link.inv_link(eta2)
        mu1[o1ix] = 1.0
        prob = mu1 - mu2
        
        
        d1eta1, d1eta2 = link.dinv_link(eta1+o1), link.dinv_link(eta2)
        d2eta1, d2eta2 = link.d2inv_link(eta1), link.d2inv_link(eta2)
        
         
        d1eta1[o1ix], d2eta1[o1ix] = 0.0, 0.0
        
        w3 =  w / prob**2
        dprob = (B1 * d1eta1[:, None]) - (B2 * d1eta2[:, None])
        T0 = (B1 * (d2eta1 / prob)[:, None]).T.dot(B1)
        T1 = (B2 * (d2eta2 / prob)[:, None]).T.dot(B2)
        T2 = (dprob * w3[:, None]).T.dot(dprob)
        H = -(T0 - T1 - T2)
        return H
    
    def hessian(self, params, data=None, link=None, q=None):
        link = self.link if link is None else link
        data = self.model_data if data is None else data 
        q = self.q if q is None else q
        H = self._hessian(params=params, data=data, link=link, q=q)
        return H
    
    def _fit(self, params=None, data=None, link=None, q=None,
                  opt_kws=None):
        params = self.params_init.copy() if params is None else params
        
        opt_kws = {} if opt_kws is None else opt_kws
        default_kws = dict(method='trust-constr',
                           options=dict(verbose=0, gtol=1e-6, xtol=1e-6))
        opt_kws = {**default_kws, **opt_kws}
        link = self.link if link is None else link
        data = self.model_data if data is None else data 
        q = self.q if q is None else q 
        args = (data, link, q)
        opt = sp.optimize.minimize(self.loglike_unconstrained, 
                                   jac=self.gradient_unconstrained,
                                   hess=self.hessian_unconstrained,
                                   x0=params,
                                   args=args,
                                   **opt_kws)
        params = OrderedTransform._rvs(opt.x, q)
        return params, opt
    
    def fit(self, method=None, opt_kws=None):
        self.params, self.opt = self._fit(method, opt_kws)
        n, n_params = self.n, len(self.params)
        q = self.q
        self.n_params = n_params
        self.params_hess = self.hessian(self.params)
        self.params_cov = np.linalg.pinv(self.params_hess)
        self.params_se = np.sqrt(np.diag(self.params_cov))
        self.res = self._parameter_inference(self.params, self.params_se,
                                             n-n_params,
                                             self.param_labels)

        self.beta_cov = self.coefs_cov = self.params_cov[q:, q:]
        self.beta_se = self.coefs_se = self.params_se[q:]
        self.beta = self.coefs = self.params[q:]
                
        intercept = np.ones((self.n_obs, 1))
        B1 = np.block([self.A1.A, -intercept])
        B2 = np.block([self.A2.A, -intercept])
        dat = ModelData(B1, B2,  self.o1, self.o2, self.o1ix, self.weights)
        
        self.paramsn, self.optn = self._fit(self.params_init[:B1.shape[1]], data=dat)
        self.llf = self.loglike(self.params)
        self.lln = self.loglike(self.paramsn, data=dat)
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


    def predict(self, params=None, q=None, X=None, se=True, params_cov=None,
                ci_level=0.95):
        params = self.params if params is None else params
        params_cov = self.params_cov if params_cov is None else params_cov
        cols = self.model_data.unique
        if X is not None:
            if type(X) is pd.DataFrame:
                xinds = X.index
            else:
                xinds = None
        else:
            xinds = self.xinds
        q = self.q if q is None else q
        X = self.X if X is None else X
        tau, beta = params[:q].reshape(1, -1), params[q:]
        eta = tau - (X.dot(beta)).reshape(-1, 1)
        cmu = self.link.inv_link(eta)
        mu = np.hstack([np.zeros((cmu.shape[0], 1)), cmu, np.ones((cmu.shape[0], 1))])
        mu = np.diff(mu, axis=1)
        if se:
            ind = np.ones(len(params), dtype=bool)
            ind[:q] = False
            eta_se = np.zeros_like(eta)
            Xi = np.concatenates([np.ones((X.shape[0], 1)), -X], axis=1)
            for i in range(q):
                ind[i] = True
                V = params_cov[ind][:, ind]
                eta_se[: i] = np.sqrt(wdiag_outer_prod(Xi, V, Xi))
                ind[i] = False
            ci_level = symmetric_conf_int(ci_level)
            ci_lmult = sp.special.ndtri(ci_level)
            eta_lower = eta - eta_se * ci_lmult
            eta_upper = eta + eta_se * ci_lmult
            mu_lower = self.link.inv_link(eta_lower)
            mu_upper = self.link.inv_link(eta_upper)
            
        
      
            mu_lower = pd.DataFrame(mu_lower, index=xinds, columns=cols)
            mu_upper = pd.DataFrame(mu_upper, index=xinds, columns=cols)
            
            eta_se = pd.DataFrame(eta_se, index=xinds, columns=cols)
            eta_lower = pd.DataFrame(eta_lower, index=xinds, columns=cols)
            eta_upper = pd.DataFrame(eta_upper, index=xinds, columns=cols)
        else:
            mu_lower, mu_upper = None, None
            eta_se, eta_lower, eta_upper = None, None, None
            
        mu = pd.DataFrame(mu, index=xinds, columns=cols)
        eta = pd.DataFrame(eta, index=xinds, columns=cols)
        return mu, eta, mu_lower, mu_upper, eta_se, eta_lower, eta_upper
        
    def _jacknife(self, method="optimize", verbose=True):
        if type(self.f.weights) is np.ndarray:
            weights = self.f.weights
        else:
            weights = np.ones(self.n_obs)
        if method == "optimize":
            jacknife_samples = np.zeros((self.n_obs, self.n_params))
            ii = np.ones(self.n_obs, dtype=bool)
            pbar = tqdm.tqdm(total=self.n_obs) if verbose else None
            for i in range(self.n_obs):
                ii[i] = False
                opt = self._optimize(data=(self.X[ii], self.y[ii], weights[ii]), f=self.f)
                jacknife_samples[i] = opt.x
                ii[i] = True
                if verbose:
                    pbar.update(1)
            if verbose:
                pbar.close()
        elif method == "one-step":    
            w = self.f.get_ehw(self.y, self.mu, phi=self.phi,
                               dispersion=self.dispersion,
                               weights=weights).reshape(-1, 1)
            WX = self.X * w
            h = self._compute_leverage_qr(WX)
            one_step = self._one_step_approx(WX, h, self.resid_pearson_s)
            jacknife_samples = self.params.reshape(1, -1) - one_step
        jacknife_samples = pd.DataFrame(jacknife_samples, index=self.xinds,
                                        columns=self.param_labels)
        return jacknife_samples
    
    
    def _bootstrap(self, n_boot=1000, verbose=True, return_info=False, rng=None):
        rng = np.random.default_rng() if rng is None else rng 
        n_obs, n_pars = self.n_obs, self.n_params
        B1, B2, o1, o2, o1ix, w = self.model_data
        params = OrderedTransform._fwd(self.params.copy(), self.q)
        pbar = tqdm.tqdm(total=n_boot, smoothing=1e-6) if verbose else None
        boot_samples = np.zeros((n_boot, n_pars))
        info = np.zeros((n_boot, 10)) if return_info else None
        for i in range(n_boot):
            ii = rng.choice(n_obs, size=n_obs, replace=True)
            params, opt = self._fit(params=params.copy(), data=(B1[ii], B2[ii], o1[ii],
                                                         o2[ii], o1ix[ii], w[ii]))
            boot_samples[i] = params
            if return_info:
                info[i] = [opt.nfev, opt.nhev, opt.nit, opt.niter, opt.njev, 
                           opt.optimality, opt.success, np.abs(opt.grad).max(),
                           opt.fun, opt.execution_time]
            if verbose:
                pbar.update(1)
        
        if return_info:
            info = pd.DataFrame(info,
                                columns=["nfev", "nhev", "nit", "niter",  "njev",
                                         "optimality", "success", "grad",  "fun",
                                         "time"])
        if verbose:
            pbar.close()
            
        boot_samples = pd.DataFrame(boot_samples,columns=self.param_labels)
        return boot_samples, info
      
            
            