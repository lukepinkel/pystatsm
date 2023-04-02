#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 06:50:24 2023

@author: lukepinkel
"""
import tqdm
import numba
import numpy as np
import scipy as sp
import pandas as pd
from .regression_model import RegressionMixin
from .likelihood_model import LikelihoodModel
from .links import (CloglogLink, IdentityLink, Link, LogitLink, LogLink,
                    LogComplementLink, NegativeBinomialLink, ProbitLink,
                    PowerLink)
from .families import (Binomial, Gamma, Gaussian, InverseGaussian, Poisson, 
                       NegativeBinomial, ExponentialFamily)
from ..utilities import output
from ..utilities.func_utils import symmetric_conf_int, handle_default_kws

from ..utilities.linalg_operations import wdiag_outer_prod, wls_qr, nwls   
LN2PI = np.log(2 * np.pi)

class GLM(RegressionMixin, LikelihoodModel):
    """
    Generalized Linear Model (GLM) class for fitting regression models.

    Parameters
    ----------
    formula : str, optional
        A patsy formula specifying the model to be fitted.
    data : pandas.DataFrame, optional
        A DataFrame containing the data to be used for model fitting.
    X : ndarray, optional
        The predictor variables matrix.
    y : ndarray, optional
        The response variable.
    family : ExponentialFamily, optional
        The distribution family to use. Default is Gaussian.
    scale_estimator : str, optional
        The scale estimator to use. Default is "M".
    weights : ndarray, optional
        Optional array of weights to be used in the model fitting.
    *args
        Additional positional arguments.
    **kwargs
        Additional keyword arguments.

    Attributes
    ----------
    xinds, yinds : array-like
        Indexes for predictor and response variables.
    xcols, ycols : array-like
        Columns for predictor and response variables.
    X, y, weights : ndarray
        Predictor variables matrix, response variable, and weights.
    n, n_obs : int
        Number of observations.
    p, n_var : int
        Number of variables.
    x_design_info, y_design_info : DesignInfo
        Design information for predictor and response variables.
    formula : str
        Patsy formula specifying the model.
    f : ExponentialFamily
        The distribution family to use.
    param_labels : list
        Labels for the model parameters.
    beta_init, phi_init : ndarray
        Initial values for model parameters.
    params_init : ndarray
        Initial values for model parameters.
    scale_estimator : str
        Scale estimator method.
    """
    def __init__(self, formula=None, data=None, X=None, y=None,
                 family=Gaussian, scale_estimator="M", weights=1.0, *args,
                 **kwargs):
        super().__init__(formula=formula, data=data, X=X, y=y, weights=weights, 
                         *args, **kwargs)
        self.xinds, self.yinds = self.model_data.indexes
        self.xcols, self.ycols = self.model_data.columns
        self.X, self.y, self.weights = self.model_data
        self.n = self.n_obs = self.X.shape[0]
        self.p = self.n_var = self.X.shape[1]
        self.x_design_info, self.y_design_info = self.model_data.design_info
        self.formula = formula
        #self.data = data
        if isinstance(family, ExponentialFamily) is False:
            try:
                family = family()
            except TypeError:
                pass

        self.f = family
        self.param_labels = list(self.xcols)
        self.beta_init, self.phi_init = self.get_start_values()
        self.params_init = self.beta_init

        if isinstance(self.f, (Binomial, Poisson)) \
            or self.f.name in ["Binomial", "Poisson"]:
            self.scale_estimator = 'fixed'
        else:
            self.scale_estimator = scale_estimator

        if self.scale_estimator == 'NR':
            self.params_init = np.r_[self.params_init, np.log(self.phi_init)]
            self.param_labels += ['log_scale']
        
        
    @staticmethod
    def _unpack_params(params, scale_estimator):
        if scale_estimator == "NR":
            beta, phi, tau = params[:-1], np.exp(params[-1]), params[-1]
        else:
            beta, phi, tau = params, 1.0,  0.0
        return beta, phi, tau
    
    @staticmethod
    def _unpack_params_data(params, data, scale_estimator, f):
        X, y, weights = data
        if scale_estimator == "NR":
            beta, phi, tau = params[:-1], np.exp(params[-1]), params[-1]
        else:
            beta, phi, tau = params, 1.0,  0.0
            
        mu = f.inv_link(np.dot(X, beta))
        
        if f.name == "NegativeBinomial":
            dispersion, phi = phi, 1.0
        else:
            dispersion = 1.0

        if scale_estimator == "M":
            phi = f.pearson_chi2(y, mu=mu, dispersion=dispersion) / (X.shape[0] - X.shape[1])
        elif scale_estimator != "NR":
            phi = 1.0
        return X, y, weights, mu, beta, phi, tau, dispersion

    @staticmethod
    def _loglike(params, data, scale_estimator, f):
        X, y, weights, mu, beta, phi, tau, dispersion = GLM._unpack_params_data(params, data, 
                                                           scale_estimator, f)
        ll = f.loglike(y, weights=weights, mu=mu, phi=phi, dispersion=dispersion)
        return ll

    def loglike(self, params, data=None, scale_estimator=None, f=None):
        data = self.model_data if data is None else data  #data = (self.X, self.y, None) if data is None else data
        s = self.scale_estimator if scale_estimator is None \
            else scale_estimator
        f = self.f if f is None else f
        ll = self._loglike(params=params, data=data, scale_estimator=s, f=f)
        return ll

    @staticmethod
    def _full_loglike(params, data, scale_estimator, f):
        X, y, weights, mu, beta, phi, tau, dispersion = GLM._unpack_params_data(params, data,
                                                           scale_estimator, f)
        ll = f.full_loglike(y, weights=weights, mu=mu, phi=phi, dispersion=dispersion)
        return ll

    def full_loglike(self, params, data=None, scale_estimator=None, f=None):
        data = self.model_data if data is None else data  #data = (self.X, self.y, None) if data is None else data
        s = self.scale_estimator if scale_estimator is None\
            else scale_estimator
        f = self.f if f is None else f
        ll = self._full_loglike(params=params, data=data, scale_estimator=s, f=f)
        return ll

    @staticmethod
    def _gradient(params, data, scale_estimator, f):
        X, y, weights, mu, beta, phi, tau, dispersion = GLM._unpack_params_data(params, data, 
                                                           scale_estimator, f)
        w = f.gw(y, mu=mu, phi=phi, dispersion=dispersion, weights=weights)
        g = np.dot(X.T, w)
        if scale_estimator == 'NR':
            dt = np.atleast_1d(np.sum(f.dtau(tau, y, mu, weights=weights)))
            g = np.concatenate([g, dt])
        return g

    def gradient(self, params, data=None, scale_estimator=None, f=None):
        data = self.model_data if data is None else data  #data = (self.X, self.y, None) if data is None else data
        s = self.scale_estimator if scale_estimator is None\
            else scale_estimator
        f = self.f if f is None else f
        ll = self._gradient(params=params, data=data, scale_estimator=s, f=f)
        return ll
    
    @staticmethod
    def _gradient_i(params, data, scale_estimator, f):
        X, y, weights, mu, beta, phi, tau, dispersion = GLM._unpack_params_data(params, data, 
                                                           scale_estimator, f)
        w = f.gw(y, mu=mu, phi=phi, dispersion=dispersion, weights=weights)
        g = X * w.reshape(-1, 1)
        if scale_estimator == 'NR':
            dt = f.dtau(tau, y, mu, weights=weights, reduce=False).reshape(-1, 1)
            g = np.concatenate([g, dt], axis=1)
        return g

    def gradient_i(self, params, data=None, scale_estimator=None, f=None):
        data = self.model_data if data is None else data  #data = (self.X, self.y, None) if data is None else data
        s = self.scale_estimator if scale_estimator is None\
            else scale_estimator
        f = self.f if f is None else f
        ll = self._gradient_i(params=params, data=data, scale_estimator=s, f=f)
        return ll

    @staticmethod
    def _hessian(params, data, scale_estimator, f):
        X, y, weights, mu, beta, phi, tau, dispersion = GLM._unpack_params_data(params, data, 
                                                           scale_estimator, f)
        gw, hw = f.get_ghw(y, mu=mu, phi=phi, dispersion=dispersion, weights=weights)
        H = np.dot((X * hw.reshape(-1, 1)).T, X)
        if isinstance(f, NegativeBinomial) or f.name=="NegativeBinomial":
            dbdt = -np.dot(X.T, dispersion * (y - mu) /
                           ((1 + dispersion * mu)**2 * f.dlink(mu)))
        else:
            dbdt = np.dot(X.T, gw)
        #dbdt = np.dot(X.T, gw)
        if scale_estimator == 'NR':
            d2t = np.atleast_2d(f.d2tau(tau, y, mu, weights=weights))
            dbdt = -np.atleast_2d(dbdt)
            H = np.block([[H, dbdt.T], [dbdt, d2t]])
        return H

    def hessian(self, params, data=None, scale_estimator=None, f=None):
        data = self.model_data if data is None else data  #data = (self.X, self.y, None) if data is None else data
        s = self.scale_estimator if scale_estimator is None\
            else scale_estimator
        f = self.f if f is None else f
        ll = self._hessian(params=params, data=data, scale_estimator=s, f=f)
        return ll

    def _optimize(self, t_init=None, opt_kws=None, data=None, s=None, f=None):
        t_init = self.params_init if t_init is None else t_init
        data = self.model_data if data is None else data #(self.X, self.y, None) if data is None else data
        s = self.scale_estimator if s is None else s
        if s=="M":
            s = "fixed"
        f = self.f if f is None else f
        opt_kws = {} if opt_kws is None else opt_kws
        default_kws = dict(method='trust-constr',
                           options=dict(verbose=0, gtol=1e-6, xtol=1e-6))
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
        eta_init[np.isnan(eta_init) | np.isinf(eta_init)] = 1.0
        gp[np.isnan(gp) | np.isinf(gp)] = 1.0
        vmu[np.isnan(vmu) | np.isinf(vmu)] = 1.0
        den = vmu * (gp**2)
        we = 1 / den
        we[den == 0] = 0.0
        if isinstance(self.f, Binomial):
            z = eta_init + (self.y - mu_init) * gp
        else:
            z = eta_init
        if np.any(we < 0):
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
        weights = self.model_data.weights
        n, n_params = self.n, len(self.params)
        self.n_params = n_params
        self.params_hess = self.hessian(self.params)
        self.params_cov = np.linalg.inv(self.params_hess)
        self.params_se = np.sqrt(np.diag(self.params_cov))
        self.res = self._parameter_inference(self.params, self.params_se,
                                             n-n_params,
                                             self.param_labels)
        if scale_estimator == "NR":
            beta, phi = self.params[:-1], np.exp(self.params[-1])
            if f.name == "NegativeBinomial":
                dispersion, phi = phi, 1.0
            else:
                dispersion = 1.0
            beta_cov = self.params_cov[:-1, :-1]
            beta_se = self.params_se[:-1]
            beta_labels = self.param_labels[:-1]
        else:
            beta = self.params
            beta_cov = self.params_cov
            beta_se = self.params_se
            beta_labels = self.param_labels
            dispersion, phi = 1.0, 1.0
        self.beta_cov = self.coefs_cov = beta_cov
        self.beta_se = self.coefs_se = beta_se
        self.beta = self.coefs = beta
        self.n_beta = len(beta)
        self.beta_labels = self.coefs_labels = beta_labels
        eta = np.dot(X, beta)
        self.eta = self.linpred = eta
        mu = f.inv_link(eta)
        self.chi2 = f.pearson_chi2(y, mu=mu, phi=1.0, dispersion=dispersion, weights=weights)
        if scale_estimator == "M":
            phi = self.chi2 / (X.shape[0] - X.shape[1])
        elif scale_estimator != "NR":
            phi = 1.0
        self.phi = phi
        self.dispersion = dispersion
        WX = X * np.sqrt(f.get_ehw(y, mu, phi=phi, dispersion=dispersion).reshape(-1, 1))
        h = self._compute_leverage_qr(WX)

        self.resid_raw = y - mu
        self.resid_pearson = self.f.pearson_resid(y, mu=mu, phi=1.0, dispersion=dispersion)
        self.resid_deviance = self.f.deviance_resid(y, mu=mu, phi=1.0, dispersion=dispersion)
        self.resid_signed = self.f.signed_resid(y, mu=mu, phi=1.0, dispersion=dispersion)

        self.resid_pearson_s = self.resid_pearson / np.sqrt(phi)
        self.resid_pearson_sst = self.resid_pearson_s / np.sqrt(1 - h)
        self.resid_deviance_s = self.resid_deviance / np.sqrt(phi)
        self.resid_deviance_sst = self.resid_deviance_s / np.sqrt(1 - h)

        self.resid_likelihood = np.sign(self.resid_raw) \
            * np.sqrt(h * self.resid_pearson_sst**2
                      + (1 - h) * self.resid_deviance_sst**2)
        self.cooks_distance = (
            h * self.resid_pearson_sst**2) / (self.p * (1 - h))

        self.resids = np.vstack([self.resid_raw,
                                 self.resid_pearson,
                                 self.resid_deviance,
                                 self.resid_signed,
                                 self.resid_pearson_s,
                                 self.resid_pearson_sst,
                                 self.resid_deviance_s,
                                 self.resid_deviance_sst,
                                 self.resid_likelihood,
                                 self.cooks_distance]).T
        self.resids = pd.DataFrame(self.resids,
                                   columns=["Raw",
                                            "Pearson",
                                            "Deviance",
                                            "Signed",
                                            "PearsonS",
                                            "PearsonSS",
                                            "DevianceS",
                                            "DevianceSS",
                                            "Likelihood",
                                            "Cooks"])

        self.llf = self.f.full_loglike(y, mu=mu, phi=phi, dispersion=dispersion, weights=weights)
        if self.f.name == "NegativeBinomial":
            opt_null = self._optimize(t_init=np.zeros(
                2), data=(np.ones((self.n, 1)), self.y, 1.0))
            self.lln = self.full_loglike(
                opt_null.x, data=(np.ones((self.n, 1)), self.y, weights))
        else:
            self.lln = self.f.full_loglike(
                y, mu=np.ones(mu.shape[0])*y.mean(), phi=phi, weights=weights)

        k = len(self.params)
        sumstats = {}
        self.aic, self.aicc, self.bic, self.caic = self._get_information(
            self.llf, k, self.n_obs)
        self.r2_cs, self.r2_nk, self.r2_mc, self.r2_mb, self.llr = \
            self._get_pseudo_rsquared(self.llf, self.lln, k, self.n_obs)
        self.r2 = self._rsquared(y, mu)
        sumstats["AIC"] = self.aic
        sumstats["AICC"] = self.aicc
        sumstats["BIC"] = self.bic
        sumstats["CAIC"] = self.caic
        sumstats["R2_CS"] = self.r2_cs
        sumstats["R2_NK"] = self.r2_nk
        sumstats["R2_MC"] = self.r2_mc
        sumstats["R2_MB"] = self.r2_mb
        sumstats["R2_SS"] = self.r2
        sumstats["LLR"] = self.llr
        sumstats["LLF"] = self.llf
        sumstats["LLN"] = self.lln
        sumstats["Deviance"] = np.sum(f.deviance(y=y, mu=mu, phi=1.0, 
                                                 dispersion=dispersion,
                                                 weights=weights))
        sumstats["Chi2"] = self.chi2
        sumstats = pd.DataFrame(sumstats, index=["Statistic"]).T
        self.sumstats = sumstats
        self.mu = mu
        self.h = h

    
    def _one_step_approx(self, WX, h, rp):
        y = rp / np.sqrt(1 - h)
        db = WX.dot(np.linalg.inv(np.dot(WX.T, WX))) * y.reshape(-1, 1)
        return db
        
    def get_robust_res(self, grad_kws=None):
        default_kws = dict(params=self.params,
                           data=(self.X, self.y, self.f.weights), 
                           scale_estimator=self.scale_estimator,
                           f=self.f)
        grad_kws = handle_default_kws(grad_kws, default_kws)
        G = self.gradient_i(**grad_kws)
        B = np.dot(G.T, G)
        A = self.params_cov
        V = A.dot(B).dot(A)
        res = self._parameter_inference(self.params, np.sqrt(np.diag(V)),
                                        self.n-len(self.params),
                                        self.param_labels)
        return res

    @staticmethod
    def _predict(coefs, X, f, phi=1.0, dispersion=1.0, coefs_cov=None, linpred=True,
                 linpred_se=True,  mean=True, mean_ci=True, mean_ci_level=0.95,
                 predicted_ci=True, predicted_ci_level=0.95):
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
            v = f.variance(mu=mu, phi=phi, dispersion=dispersion)
            res["predicted_lower_ci"] = f.ppf(
                1-predicted_ci_level, mu=mu, scale=v)
            res["predicted_upper_ci"] = f.ppf(
                predicted_ci_level, mu=mu, scale=v)
        return res
    
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
        n, q = self.n_obs, self.n_params
        weights = self.f.weights
        w = weights if type(weights) is np.ndarray else np.ones(n)
        X, y = self.X, self.y
        pbar = tqdm.tqdm(total=n_boot, smoothing=1e-6) if verbose else None
        boot_samples = np.zeros((n_boot, q))
        info = np.zeros((n_boot, 10)) if return_info else None
        for i in range(n_boot):
            ii = rng.choice(n, size=n, replace=True)
            opt = self._optimize(data=(X[ii], y[ii], w[ii]), f=self.f)
            boot_samples[i] = opt.x
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
    

            
