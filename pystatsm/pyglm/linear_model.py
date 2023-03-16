#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 19:14:22 2023

@author: lukepinkel
"""
import tqdm
import patsy  # analysis:ignore
import numpy as np # analysis:ignore
import scipy as sp # analysis:ignore
import scipy.stats # analysis:ignore
import pandas as pd # analysis:ignore
from scipy.special import loggamma, digamma, polygamma
from functools import cached_property
from abc import ABCMeta, abstractmethod
from scipy.linalg.lapack import dtrtri
from ..utilities import output
from ..utilities.linalg_operations import wdiag_outer_prod, wls_qr, nwls
from ..utilities.func_utils import symmetric_conf_int
from ..utilities.data_utils import _check_shape, _check_type
from .links import LogitLink, ProbitLink, Link, LogLink, ReciprocalLink, PowerLink # analysis:ignore
from .families import (Binomial, ExponentialFamily, Gamma, Gaussian,  # analysis:ignore
                      IdentityLink, InverseGaussian, NegativeBinomial,  # analysis:ignore
                      Poisson) # analysis:ignore


LN2PI = np.log(2 * np.pi)

class LikelihoodModel(metaclass=ABCMeta):
    
    @staticmethod
    @abstractmethod
    def _loglike(params, data):
        pass
    
    @staticmethod
    @abstractmethod
    def _gradient(params, data):
        pass
    
    @staticmethod
    @abstractmethod
    def _hessian(params, data):
        pass
    
    @staticmethod
    @abstractmethod
    def _fit(params, data):
        pass
    
    @staticmethod
    def _get_information(ll, n_params, n_obs):
        logn = np.log(n_obs)
        tll = 2 * ll
        aic = tll + 2 * n_params
        aicc= tll + 2 * n_params * n_obs / (n_obs - n_params - 1)
        bic = tll + n_params * logn
        caic= tll + n_params * (logn + 1)
        return aic, aicc, bic, caic
    
    @staticmethod
    def _get_pseudo_rsquared(ll_model, ll_null, n_params, n_obs):
        r2_cs = 1-np.exp(2.0/n_obs * (ll_model - ll_null))
        r2_nk = r2_cs / (1-np.exp(2.0/ n_obs * -ll_null))
        r2_mc = 1.0 - ll_model / ll_null
        r2_mb = 1.0 - (ll_model - n_params) / ll_null
        llr = 2.0 * (ll_null - ll_model)
        return r2_cs, r2_nk, r2_mc, r2_mb, llr
    
    @staticmethod
    def _parameter_inference(params, params_se, degfree, param_labels):
        res = output.get_param_table(params, params_se,degfree, param_labels)
        return res

class RegressionMixin(object):
    
    def __init__(self, formula=None, data=None, X=None, y=None, *args, **kwargs):
        X, xc, xi, y, yc, yi, d = self._process_data(formula, data, X, y)
        self.X, self.y = X, y
        self.xcols, self.xinds = xc, xi
        self.ycols, self.yinds = yc, yi
        self.n = self.n_obs = X.shape[0]
        self.p = self.n_var = X.shape[1]
        self.design_info = d
        self.formula = formula
        self.data = data
        
    @staticmethod
    def _process_data(formula=None, data=None, X=None, y=None, default_varname='x'):
        if formula is not None and data is not None:
            y, X = patsy.dmatrices(formula, data=data, return_type='dataframe')
            xcols, xinds = X.columns, X.index
            ycols, yinds = y.columns, y.index
            design_info = X.design_info
            X, y = X.values, y.values[:, 0]
        elif X is not None and y is not None:
            design_info = None
            if type(X) not in [pd.DataFrame, pd.Series]:
                if X.ndim==1:
                    xcols = [f'{default_varname}']
                else:
                    xcols = [f'{default_varname}{i}' for i in range(1, X.shape[1]+1)]
                xinds = np.arange(X.shape[0])
                ycols, yinds = ['y'], np.arange(y.shape[0])
            else:
                X, xcols, xinds = X.values, X.columns, X.index
            if type(y) not in [pd.DataFrame, pd.Series]:
                ycols, yinds = ['y'], np.arange(y.shape[0])
            else:
                y, ycols, yinds = y.values, y.columns, y.index
        return X, xcols, xinds, y, ycols, yinds, design_info
    
    @staticmethod
    def sandwich_cov(grad_weight, X, leverage=None, kind="HC0"):
        w, h = grad_weight, leverage
        n, p = X.shape
        w = w ** 2
        if kind == "HC0":
            omega = w
        elif kind == "HC1":
            omega = w * n / (n - p)
        elif kind == "HC2":
            omega = w / (1.0 - h)
        elif kind == "HC3":
            omega = w / (1.0 - h)**2
        elif kind == "HC4":
            omega = w / (1.0 - h)**np.minimum(4.0, h / np.mean(h))
        B = np.dot((X * omega.reshape(-1, 1)).T, X)
        return B
    
    @staticmethod
    def _compute_leverage_cholesky(WX=None, Linv=None):
        if Linv is None:
            G = np.dot(WX.T, WX)
            L = np.linalg.cholesky(G)
            Linv, _ = scipy.linalg.lapack.dtrtri(L, lower=1)
        Q = Linv.dot(WX.T)
        h = np.sum(Q**2, axis=0)
        return h
            
    @staticmethod
    def _compute_leverage_qr(WX=None, Q=None):
        if Q is None:
            Q, R = np.linalg.qr(WX)
        h = np.sum(Q**2, axis=1)
        return h
    
    @staticmethod
    def _rsquared(y, yhat):
        rh = y - yhat
        rb = y - np.mean(y)
        r2 = 1.0 - np.sum(rh**2) / np.sum(rb**2)
        return r2
    
    
    
class LinearModel(RegressionMixin, LikelihoodModel):
    
    
    def __init__(self, formula=None, data=None, X=None, y=None, *args, **kwargs):
        super().__init__(formula=formula, data=data, X=X, y=y, *args, **kwargs)
 
        
    @staticmethod
    def _loglike(params, data, reml=True):
        beta, logvar = params[:-1], params[-1]
        tau = np.exp(-logvar)
        X, y = data
        n, p = X.shape
        n = n - p if reml else n
        const =  (-n / 2) * (LN2PI + logvar)
        yhat = X.dot(beta)
        resid = y - yhat
        sse = np.dot(resid.T, resid)
        ll = const - (tau * sse) / 2
        return -ll
    
    def loglike(self, params, reml=True):
        return self._loglike(params, (self.X, self.y), reml=reml)
    
    @staticmethod
    def _gradient(params, data, reml=True):
        beta, logvar = params[:-1], params[-1]
        tau = np.exp(-logvar)
        X, y = data
        n, p = X.shape
        n = n - p if reml else n
        grad = np.zeros_like(params)
        yhat = X.dot(beta)
        resid = y - yhat
        grad[:-1] = tau * np.dot(resid.T, X)
        grad[ -1] = (-n / 2) + (tau * np.dot(resid.T, resid)) / 2.0
        return -grad
    
    def gradient(self, params, reml=True):
        return self._gradient(params, (self.X, self.y), reml=reml)
    
    
    @staticmethod
    def _hessian(params, data, reml=True):
        beta, logvar = params[:-1], params[-1]
        tau = np.exp(-logvar)
        X, y = data
        hess = np.zeros((params.shape[0],)*2)
        yhat = X.dot(beta)
        resid = y - yhat
        hess[:-1, :-1] = -tau * np.dot(X.T, X)
        hess[:-1, -1] =  -tau * np.dot(X.T, resid)
        hess[-1, :-1] =  -tau * np.dot(X.T, resid)
        hess[-1, -1] = -(tau * np.dot(resid.T, resid)) / 2 
        return -hess 
    
    def hessian(self, params, reml=True):
        return self._hessian(params, (self.X, self.y), reml=reml)
    
    @staticmethod
    def _fit1(params, data, reml=True):
        X, y = data
        n, p = X.shape
        G = X.T.dot(X)
        c = X.T.dot(y)
        L = np.linalg.cholesky(G)
        w = sp.linalg.solve_triangular(L, c, lower=True)
        sse =  y.T.dot(y) - w.T.dot(w)
        beta = sp.linalg.solve_triangular(L.T, w, lower=False)
        Linv = np.linalg.inv(L)
        Ginv = np.dot(Linv.T, Linv)
        n = n - p if reml else n
        params = np.zeros(len(beta)+1)
        params[:-1], params[-1] = beta, np.log(sse / n)
        return G, Ginv, L, Linv, sse, beta, params
    
        
    @staticmethod
    def _fit2(params, data, reml=True):
        X, y = data
        n, p = X.shape
        G = np.dot(X.T, X)
        c = X.T.dot(y)
        L = np.linalg.cholesky(G)
        Linv, _ = scipy.linalg.lapack.dtrtri(L, lower=1)
        w = Linv.dot(c)
        sse =  y.T.dot(y) - w.T.dot(w)
        beta = np.dot(Linv.T, w)
        Ginv = np.dot(Linv.T, Linv)
        n = n - p if reml else n
        params = np.zeros(len(beta)+1)
        params[:-1], params[-1] = beta, np.log(sse / n)
        return G, Ginv, L, Linv, sse, beta, params
    
    def _fit(self, reml=True):
        G, Ginv, L, Linv, sse, beta, params = self._fit2(None, (self.X, self.y), reml=reml)
        self.G, self.Ginv, self.L, self.Linv = G, Ginv, L, Linv
        self.sse = sse
        self.scale = np.exp(params[-1]/2)
        self.scale_ml = np.sqrt(sse / self.n)
        self.scale_unbiased = np.sqrt(sse / (self.n - self.p))
        self.beta = beta
        self.params = params
        self.params_hess = self.hessian(self.params)
        self.params_cov = np.linalg.inv(self.params_hess)
        self.params_se = np.diag(np.linalg.inv(self.hessian(self.params)))**0.5
        self.res = output.get_param_table(self.params, self.params_se, 
                                          self.n-self.p,
                                          list(self.xcols)+["log_scale"],
                                          )
        

    @staticmethod
    def _get_coef_constrained(params, sse, data, C, d=None, L=None, Linv=None):
        if Linv is None:
            if L is None:
                X, y = data
                L = np.linalg.cholesky(X.T.dot(X))
            Linv, _ = scipy.linalg.lapack.dtrtri(L, lower=1)
        d = np.zeros(len(C)) if d is None else d
        G = Linv.dot(C.T)
        Q, R = np.linalg.qr(G)
        Cb = C.dot(params[:-1])
        w = sp.linalg.solve_triangular(R.T, Cb-d, lower=True)
        sse_constrained = sse + np.dot(w.T, w)
        beta_constrained = params[:-1] - Linv.T.dot(Q.dot(w))
        return beta_constrained, sse_constrained
        



class GLM(RegressionMixin, LikelihoodModel):
    
    def __init__(self, formula=None, data=None, X=None, y=None, family=Gaussian, 
                 scale_estimator="M", *args, **kwargs):
        super().__init__(formula=formula, data=data, X=X, y=y, *args, **kwargs)

        if isinstance(family, ExponentialFamily)==False:
            try:
                family = family()
            except TypeError:
                pass
        
        self.f = family
        self.param_labels = list(self.xcols) 
        self.beta_init, self.phi_init = self.get_start_values()
        self.params_init = self.beta_init
        
        if isinstance(self.f, (Binomial, Poisson)):
            self.scale_estimator = 'fixed'
        else:
            self.scale_estimator = scale_estimator   
        
        if self.scale_estimator == 'NR':
            self.params_init = np.concatenate([self.params_init,
                                              np.atleast_1d(np.log(self.phi_init))])
            self.param_labels += ['log_scale']
     

    @staticmethod
    def _loglike(params, data, scale_estimator, f):
        X, y = data
        if scale_estimator == "NR":
            beta, phi = params[:-1], np.exp(params[-1])
        else:
            beta = params
        mu = f.inv_link(np.dot(X, beta))
        if scale_estimator == "M":
            phi = f.pearson_chi2(y, mu=mu) / (X.shape[0] - X.shape[1])
        elif scale_estimator != "NR":
            phi = 1.0
        ll = f.loglike(y, mu=mu, scale=phi)
        return ll
        
        
    def loglike(self, params, data=None, scale_estimator=None, f=None):
        data = (self.X, self.y) if data is None else data
        s = self.scale_estimator if scale_estimator is None else scale_estimator
        f = self.f if f is None else f
        ll = self._loglike(params=params, data=data, scale_estimator=s, f=f)
        return ll
    
    @staticmethod
    def _full_loglike(params, data, scale_estimator, f):
        X, y = data
        if scale_estimator == "NR":
            beta, phi = params[:-1], np.exp(params[-1])
        else:
            beta = params
        mu = f.inv_link(np.dot(X, beta))
        if scale_estimator == "M":
            phi = f.pearson_chi2(y, mu=mu) / (X.shape[0] - X.shape[1])
        elif scale_estimator != "NR":
            phi = 1.0
        ll = f.full_loglike(y, mu=mu, scale=phi)
        return ll
        
        
    def full_loglike(self, params, data=None, scale_estimator=None, f=None):
        data = (self.X, self.y) if data is None else data
        s = self.scale_estimator if scale_estimator is None else scale_estimator
        f = self.f if f is None else f
        ll = self._full_loglike(params=params, data=data, scale_estimator=s, f=f)
        return ll
        
    @staticmethod
    def _gradient(params, data, scale_estimator, f):
        X, y = data
        if scale_estimator == "NR":
            beta, tau = params[:-1], params[-1]
            phi = np.exp(tau)
        else:
            beta = params
        mu = f.inv_link(np.dot(X, beta))
        
        if scale_estimator == "M":
            phi = f.pearson_chi2(y, mu=mu) / (X.shape[0] - X.shape[1])
        elif scale_estimator != "NR":
            phi = 1.0
        w = f.gw(y, mu=mu, phi=phi)
        g = np.dot(X.T, w)
        if scale_estimator == 'NR':
            dt = np.atleast_1d(np.sum(f.dtau(tau, y, mu)))
            g = np.concatenate([g, dt])
        return g
        
    def gradient(self, params, data=None, scale_estimator=None, f=None):
        data = (self.X, self.y) if data is None else data
        s = self.scale_estimator if scale_estimator is None else scale_estimator
        f = self.f if f is None else f
        ll = self._gradient(params=params, data=data, scale_estimator=s, f=f)
        return ll

    @staticmethod
    def _hessian(params, data, scale_estimator, f):
        X, y = data
        if scale_estimator == "NR":
            beta, tau = params[:-1], params[-1]
            phi = np.exp(tau)
        else:
            beta = params
        mu = f.inv_link(np.dot(X, beta))
        if scale_estimator == "M":
            phi = f.pearson_chi2(y, mu=mu) / (X.shape[0] - X.shape[1])
        elif scale_estimator != "NR":
            phi = 1.0
        gw, hw = f.get_ghw(y, mu=mu, phi=phi)
        H = np.dot((X * hw.reshape(-1, 1)).T, X)
        if isinstance(f, NegativeBinomial):
            dbdt = -np.dot(X.T, phi * (y - mu) / ((1 + phi * mu)**2 * f.dlink(mu)))
        else:
            dbdt = np.dot(X.T, gw)
        if scale_estimator == 'NR':
            d2t = np.atleast_2d(f.d2tau(tau, y, mu))
            dbdt = -np.atleast_2d(dbdt)
            H = np.block([[H, dbdt.T], [dbdt, d2t]])
        return H 
        
    def hessian(self, params, data=None, scale_estimator=None, f=None):
        data = (self.X, self.y) if data is None else data
        s = self.scale_estimator if scale_estimator is None else scale_estimator
        f = self.f if f is None else f
        ll = self._hessian(params=params, data=data, scale_estimator=s, f=f)
        return ll
        
    
    def _optimize(self, t_init=None, opt_kws=None, data=None, s=None, f=None):
        t_init = self.params_init if t_init is None else t_init
        data = (self.X, self.y) if data is None else data
        s = self.scale_estimator if s is None else s
        f = self.f if f is None else f
        opt_kws = {} if opt_kws is None else opt_kws
        default_kws = dict(method='trust-constr', options=dict(verbose=0, gtol=1e-6, xtol=1e-6))
        opt_kws = {**default_kws, **opt_kws}
        args = (data, s, f)
        optimizer = sp.optimize.minimize(self.loglike, t_init, args=args,
                                         jac=self.gradient, hess=self.hessian, 
                                         **opt_kws)
        return optimizer
    
    def get_start_values(self):
        if isinstance(self.f, Binomial):
            mu_init = (self.y * self.f.weights + 0.5) / (self.f.weights + 1)
        else:
            mu_init = (self.y + self.y.mean()) / 2.0
        eta_init = self.f.link(mu_init)
        gp = self.f.dlink(mu_init)
        vmu = self.f.var_func(mu=mu_init)
        eta_init[np.isnan(eta_init)|np.isinf(eta_init)] = 1.0
        gp[np.isnan(gp)|np.isinf(gp)] = 1.0
        vmu[np.isnan(vmu)|np.isinf(vmu)] = 1.0
        den = vmu * (gp**2)
        we = 1 / den
        we[den==0] = 0.0
        if isinstance(self.f, Binomial):
            z = eta_init + (self.y - mu_init) * gp
        else:
            z = eta_init
        if np.any(we<0):
            beta_init = nwls(self.X, z, we)
        else:
            beta_init = wls_qr(self.X, z, we)
        mu = self.f.inv_link(self.X.dot(beta_init))
        phi_init = self.f.pearson_chi2(self.y, mu=mu) / (self.n - self.p)
        return beta_init, phi_init
         

    def _fit(self, method=None, opt_kws=None):
        opt = self._optimize(opt_kws=opt_kws)
        params = opt.x
        return params, opt
            
    def fit(self, method=None, opt_kws=None):
        self.params, self.opt = self._fit(method, opt_kws)
        f, scale_estimator, X, y = self.f, self.scale_estimator, self.X, self.y
        n, n_params = self.n, len(self.params)
        self.params_hess = self.hessian(self.params)
        self.params_cov = np.linalg.inv(self.params_hess)
        self.params_se = np.sqrt(np.diag(self.params_cov))
        self.res = self._parameter_inference(self.params, self.params_se, 
                                             n-n_params,
                                             self.param_labels)
        
        if scale_estimator == "NR":
            beta, phi = self.params[:-1], np.exp(self.params[-1])
            beta_cov = self.params_cov[:-1, :-1]
            beta_se = self.params_se[:-1]
            beta_labels = self.param_labels[:-1]
        else:
            beta = self.params
            beta_cov = self.params_cov
            beta_se = self.params_se
            beta_labels = self.param_labels
        self.beta_cov = self.coefs_cov = beta_cov
        self.beta_se = self.coefs_se = beta_se
        self.beta = self.coefs = beta
        self.n_beta = len(beta)
        self.beta_labels = self.coefs_labels = beta_labels
        eta = np.dot(X, beta)
        self.eta = self.linpred = eta
        mu = f.inv_link(eta)
        self.chi2 = f.pearson_chi2(y, mu=mu)
        if scale_estimator == "M":
            phi = self.chi2 / (X.shape[0] - X.shape[1])
        elif scale_estimator != "NR":
            phi = 1.0
        self.scale = self.phi = phi
        WX = X * f.hw(y, mu, phi).reshape(-1, 1)
        h = self._compute_leverage_qr(WX)
        
        self.resid_raw = y - mu
        self.resid_pearson = self.f.pearson_resid(y, mu=mu, scale=phi)
        self.resid_deviance = self.f.deviance_resid(y, mu=mu, scale=phi)
        self.resid_signed = self.f.signed_resid(y, mu=mu, scale=phi)

        self.resid_pearson_s   = self.resid_pearson / np.sqrt(phi)
        self.resid_pearson_sst = self.resid_pearson_s / np.sqrt(1 - h)
        self.resid_deviance_s   = self.resid_deviance / np.sqrt(phi)
        self.resid_deviance_sst = self.resid_deviance_s / np.sqrt(1 - h)
        

        self.resid_likelihood = np.sign(self.resid_raw) \
                                * np.sqrt(h * self.resid_pearson_sst**2\
                                          +(1 - h) * self.resid_deviance_sst**2)
        self.cooks_distance = (h * self.resid_pearson_sst**2) / (self.p * (1 - h))
        
        self.resids = np.vstack([self.resid_raw, self.resid_pearson, self.resid_deviance,
                                 self.resid_signed, self.resid_pearson_s, 
                                 self.resid_pearson_sst, self.resid_deviance_s,
                                 self.resid_deviance_sst, self.resid_likelihood,
                                 self.cooks_distance]).T
        self.resids = pd.DataFrame(self.resids,
                                   columns=["Raw", "Pearson", "Deviance",
                                            "Signed", "PearsonS", "PearsonSS",
                                            "DevianceS", "DevianceSS", "Likelihood",
                                            "Cooks"])

        self.llf = self.f.full_loglike(y, mu=mu, scale=phi)
        if self.f.name == "NegativeBinomial":
            opt_null = self._optimize(t_init=np.zeros(2), data=(np.ones((self.n, 1)), self.y))
            self.lln = self.full_loglike(opt_null.x, data=(np.ones((self.n, 1)), self.y))
        else:
            self.lln = self.f.full_loglike(y, mu=np.ones(mu.shape[0])*y.mean(), scale=phi)
        
        k = len(self.params)
        sumstats = {}
        self.aic, self.aicc, self.bic, self.caic = self._get_information(self.llf, k, self.n_obs)
        self.r2_cs, self.r2_nk, self.r2_mc, self.r2_mb, self.llr = self._get_pseudo_rsquared(self.llf, self.lln, k, self.n_obs)
        
        sumstats["AIC"] = self.aic
        sumstats["AICC"] = self.aicc
        sumstats["BIC"] = self.bic
        sumstats["CAIC"] = self.caic
        sumstats["R2_CS"] = self.r2_cs
        sumstats["R2_NK"] = self.r2_nk
        sumstats["R2_MC"] = self.r2_mc
        sumstats["R2_MB"] = self.r2_mb
        sumstats["LLR"] = self.llr
        sumstats["LLF"] = self.llf
        sumstats["LLN"] = self.lln
        sumstats["Deviance"] = np.sum(f.deviance(y=y, mu=mu, scale=phi))
        sumstats["Chi2"] = self.chi2
        sumstats = pd.DataFrame(sumstats, index=["Statistic"]).T
        self.sumstats = sumstats
        self.mu = mu
        self.h = h

    
    def get_robust_res(self, kind="HC3"):
        w = self.f.gw(self.y, mu=self.mu, phi=self.phi)
        B = self.sandwich_cov(w, self.X, self.h, kind=kind)
        A = self.coefs_cov
        V = A.dot(B).dot(A)
        res = self._parameter_inference(self.coefs, np.sqrt(np.diag(V)), 
                                        self.n-len(self.params), 
                                        self.param_labels)
        return res

        
    
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
            elif method == "ee":
                theta_samples[i], _ = self._fit_estimating_equations(theta_init=t_init, 
                                               X=self.X[ix], 
                                               Y=self.Y[ix])
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
        
    @staticmethod
    def _predict(coefs, X, f, scale=1.0, coefs_cov=None, linpred=True, linpred_se=True, 
                 mean=True, mean_ci=True, mean_ci_level=0.95, predicted_ci=True, 
                 predicted_ci_level=0.95):
        res = {}
        eta = np.dot(X, coefs)
        if linpred_se or mean_ci:
            eta_se = wdiag_outer_prod(X, coefs_cov, X)
            
        if mean or mean_ci or predicted_ci:
            mu = f.inv_link(eta)
        
        if linpred:
            res["eta"] = eta
        if linpred_se:
            res["eta_se"] = eta_se
        if mean or mean_ci or predicted_ci:
            mu = f.inv_link(eta)
            res["mu"] = mu
        
        if mean_ci:
            mean_ci_level = symmetric_conf_int(mean_ci_level)
            mean_ci_lmult = sp.special.ndtri(mean_ci_level)
            res["eta_lower_ci"] = eta - mean_ci_lmult * eta_se
            res["eta_upper_ci"] = eta + mean_ci_lmult * eta_se
            res["mu_lower_ci"] = f.inv_link(res["eta_lower_ci"])
            res["mu_upper_ci"] = f.inv_link(res["eta_upper_ci"])
        
        if predicted_ci:
            predicted_ci_level = symmetric_conf_int(predicted_ci_level)
            var = scale * f.var_func(mu=mu)
            res["predicted_lower_ci"] = f.ppf(1-predicted_ci_level, mu=mu, scale=var)
            res["predicted_upper_ci"] = f.ppf(predicted_ci_level, mu=mu, scale=var)
        return res