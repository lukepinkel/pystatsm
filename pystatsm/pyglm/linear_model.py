#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 19:14:22 2023

@author: lukepinkel
"""
import tqdm                                                                    # analysis:ignore
import patsy
import numba
import numpy as np
import scipy as sp
import scipy.stats
import pandas as pd
from scipy.special import loggamma, digamma, polygamma                         # analysis:ignore
from functools import cached_property                                          # analysis:ignore
from patsy import PatsyError
from abc import ABCMeta, abstractmethod
from ..utilities import output
from ..utilities.linalg_operations import wdiag_outer_prod, wls_qr, nwls       # analysis:ignore
from ..utilities.func_utils import symmetric_conf_int, handle_default_kws
from ..utilities.data_utils import _check_shape, _check_type                   # analysis:ignore
from ..utilities.formula import design_matrices                  # analysis:ignore
from .links import (LogitLink, ProbitLink, Link, LogLink, ReciprocalLink,      # analysis:ignore
                    PowerLink)                                                 # analysis:ignore
from .families import (Binomial, ExponentialFamily, Gamma, Gaussian,           # analysis:ignore
                       IdentityLink, InverseGaussian, NegativeBinomial,        # analysis:ignore
                       Poisson)                                                # analysis:ignore

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
        aicc = tll + 2 * n_params * n_obs / (n_obs - n_params - 1)
        bic = tll + n_params * logn
        caic = tll + n_params * (logn + 1)
        return aic, aicc, bic, caic

    @staticmethod
    def _get_pseudo_rsquared(ll_model, ll_null, n_params, n_obs):
        r2_cs = 1-np.exp(2.0/n_obs * (ll_model - ll_null))
        r2_nk = r2_cs / (1-np.exp(2.0 / n_obs * -ll_null))
        r2_mc = 1.0 - ll_model / ll_null
        r2_mb = 1.0 - (ll_model - n_params) / ll_null
        llr = 2.0 * (ll_null - ll_model)
        return r2_cs, r2_nk, r2_mc, r2_mb, llr

    @staticmethod
    def _parameter_inference(params, params_se, degfree, param_labels):
        res = output.get_param_table(params, params_se, degfree, param_labels)
        return res

    @staticmethod
    def _make_constraints(fixed_indices, fixed_values):
        constraints = []
        if np.asarray(fixed_indices).dtype==bool:
            fixed_indices = np.where(fixed_indices)
        for i, xi in list(zip(fixed_indices, fixed_values)):
            constraints.append({"type":"eq", "fun":lambda x: x[i] - xi})
        return constraints

    @staticmethod
    def _profile_loglike_constrained(par, ind, params, loglike, grad, hess,
                                     minimize_kws=None, return_info=False):
        par, ind = np.atleast_1d(par), np.atleast_1d(ind)
        constraints = []
        for i, xi in list(zip(ind, par)):
            constraints.append({"type":"eq", "fun":lambda x: x[i] - xi})
        default_minimize_kws = dict(fun=loglike, jac=grad, hess=hess,
                                    method="trust-constr",
                                    constraints=constraints)
        minimize_kws = handle_default_kws(minimize_kws, default_minimize_kws)
        x0 = params.copy()
        x0[ind] = par
        opt = sp.optimize.minimize(x0=x0, **minimize_kws)
        if return_info:
            res = opt
        else:
            res = loglike(opt.x)
        return res

    @staticmethod
    def _profile_loglike_restricted(par, ind, params, loglike, grad, hess,
                                    minimize_kws=None, return_info=False):
        par, ind = np.atleast_1d(par), np.atleast_1d(ind)
        free_ind = np.setdiff1d(np.arange(len(params)), ind)
        full_params = params.copy()
        full_params[ind] = par

        def restricted_func(free_params):
            full_params[free_ind] = free_params
            full_params[ind] = par
            fval = loglike(full_params)
            return fval

        def restricted_grad(free_params):
            full_params[free_ind] = free_params
            full_params[ind] = par
            g = grad(full_params)[free_ind]
            return g

        def restricted_hess(free_params):
            full_params[free_ind] = free_params
            full_params[ind] = par
            H = hess(full_params)[free_ind][:, free_ind]
            return H


        default_minimize_kws = dict(fun=restricted_func,
                                    jac=restricted_grad,
                                    hess=restricted_hess,
                                    method="trust-constr")
        minimize_kws = handle_default_kws(minimize_kws, default_minimize_kws)
        x0 = params.copy()[free_ind]
        opt = sp.optimize.minimize(x0=x0, **minimize_kws)
        if return_info:
            res = opt
        else:
            res = restricted_func(opt.x)
        return res


    @staticmethod
    def _solve_interval(profile_loglike, par, par_se, lli, method="root",
                       return_info=False):
        if method == "lstq":
            left=sp.optimize.minimize(lambda x:(profile_loglike(x)-lli)**2,
                                        par-2 * par_se)
            right=sp.optimize.minimize(lambda x:(profile_loglike(x)-lli)**2,
                                         par+2 * par_se)
            if return_info:
                left, right = left, right
            else:
                left, right = left.x, right.x
        else:
            for kl in range(1, 20):
                if profile_loglike(par - kl * par_se) > lli:
                    break
            for kr in range(1, 20):
                if profile_loglike(par + kr * par_se) > lli:
                    break
            left_bracket = (par - kl * par_se, par)
            right_bracket = (par, par + kr * par_se)
            left = sp.optimize.root_scalar(lambda x:profile_loglike(x)-lli,
                                           bracket=left_bracket)
            right = sp.optimize.root_scalar(lambda x:profile_loglike(x)-lli,
                                            bracket=right_bracket)
            if return_info:
                left, right = left, right
            else:
                left, right = left.root, right.root
        return left, right
    
    @staticmethod
    def _lr_test(ll_unconstrained, ll_constrained, constraint_dim, 
                 return_dataframe=True):
        ll_full = ll_unconstrained
        ll_null = ll_constrained
        df = constraint_dim
        lr_stat = 2.0 * (ll_full - ll_null)
        p_value = sp.stats.chi2(df=df).sf(lr_stat)
        res = {"Stat":lr_stat, "df":df ,"P Value":p_value}
        if return_dataframe:  
            res = pd.DataFrame(res, index=["LR Test"])
        return res
    
    @staticmethod
    def _wald_test(params_unconstrained, 
                   params_constrained,
                   constraint_derivative,
                   hess_unconstrained_inv,
                   grad_unconstrained_cov,
                   return_dataframe=True):
        theta, theta0 = params_unconstrained, params_constrained
        A, B = hess_unconstrained_inv, grad_unconstrained_cov
        C = constraint_derivative
        V = np.dot(A, np.dot(B, A))
        M = np.linalg.inv(C.dot(V).dot(C.T))
        a = np.dot(C, theta - theta0)
        wald_stat = np.dot(a, M.dot(a))
        df = constraint_derivative.shape[0]
        p_value = sp.stats.chi2(df=df).sf(wald_stat)
        res = {"Stat":wald_stat, "df":df ,"P Value":p_value}
        if return_dataframe:  
            res = pd.DataFrame(res, index=["Wald Test"])
        return res
    
    @staticmethod
    def _score_test(grad_constrained, 
                    constraint_derivative,
                    hess_constrained_inv, 
                    grad_constrained_cov,
                    return_dataframe=True):
        A, B = hess_constrained_inv, grad_constrained_cov
        C, D = constraint_derivative, grad_constrained
        V = np.dot(A, np.dot(B, A))
        M = np.linalg.inv(C.dot(V).dot(C.T))
        a = np.dot(C, np.dot(A, D))
        score_stat = np.dot(a, M.dot(a))
        df = constraint_derivative.shape[0]
        p_value = sp.stats.chi2(df=df).sf(score_stat)
        res = {"Stat":score_stat, "df":df ,"P Value":p_value}
        if return_dataframe:  
            res = pd.DataFrame(res, index=["Score Test"])
        return res

class ModelData(object):
    def __init__(self, *args, weights=None):
        self.data = []
        self.design_info = []
        self.indexes = []
        self.columns = []
        self.names = []
        self.weights = weights
        for arg in args:
            self.store(arg)
            
    @property
    def has_weights(self):
        return False if self.weights is None else True
    
    def store(self, data, varname="x"):
        if isinstance(data, (pd.DataFrame, pd.Series)):
            self.indexes.append(data.index)
            if isinstance(data, pd.DataFrame):
                self.columns.append(data.columns)
                self.data.append(data.to_numpy())
                if hasattr(data, "design_info"):
                    design_info = data.design_info
                else:
                    design_info = None
                self.design_info.append(design_info)
            else:  # pd.Series
                self.names.append(data.name)
                self.data.append(data.to_numpy().reshape(-1, 1))
        else:  # numpy array
            self.design_info.append(None)
            self.indexes.append(np.arange(data.shape[0]))
            self.columns.append([f"{varname}{i}" for i in range(data.shape[1])] 
                                if data.ndim==2 else [f"{varname}0"])
            self.data.append(data)
    
    def __getitem__(self, index):
        indexed_data = tuple(x[index] for x in self.data)
        if self.weights is not None:
            indexed_weights = self.weights[index]
            return (*indexed_data, indexed_weights)
        return (*indexed_data,)
    
    def __iter__(self):
        if self.weights is not None:
            return iter((*self.data, self.weights))
        return iter(self.data)
    
    def __len__(self):
        l = len(self.data) if self.weights is None else len(self.data)+1
        return l

    def __repr__(self):
        s = [str(d.shape) for d in self.data if d is not None]
        if self.weights is not None:
            s.append(str(self.weights.shape))
        return f"ModelData({', '.join(s)})"
    
    @classmethod
    def from_formulas(cls, data, main_formula, *formula_args, **formula_kwargs):
        formulas = list(formula_args) + list(formula_kwargs.values())
        y, X_main = patsy.dmatrices(main_formula, data=data, return_type='dataframe')
        
        model_matrices = [X_main]
        for formula in formulas:
            lhs, X = design_matrices(formula, data)
            model_matrices.append(X)
        model_matrices.append(y)
        return cls(*model_matrices)
    
    def add_weights(self, weights=None):
        self.weights = np.ones(len(self.data[-1])) if weights is None else weights
        
    def flatten_y(self):
        self.data[-1] = self.data[-1].flatten()

class RegressionMixin(object):

    def __init__(self, formula=None, data=None, X=None, y=None, *args,
                 **kwargs):
        self.model_data = self._process_data(formula, data, X, y, *args, **kwargs)
        
        
    @staticmethod
    def _process_data(formula=None, data=None, X=None, y=None,
                      default_varname='x', *args, **kwargs):
        if formula is not None and data is not None:
            model_data = ModelData.from_formulas(data, formula, *args, **kwargs)
            y, X = patsy.dmatrices(formula, data=data, return_type='dataframe')
        elif X is not None and y is not None:
            model_data = ModelData(*((X,)+args+(y,)))
        
        if np.ndim(model_data.data[-1]) == 2:
            if model_data.data[-1].shape[1] == 1:
                model_data.flatten_y()
        return model_data

                           
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

    def __init__(self, formula=None, data=None, X=None, y=None, *args,
                 **kwargs):
        super().__init__(formula=formula, data=data, X=X, y=y, *args,
                         **kwargs)
        self.xinds, self.yinds = self.model_data.indexes
        self.xcols, self.ycols = self.model_data.columns
        self.X, self.y = self.model_data
        self.n = self.n_obs = self.X.shape[0]
        self.p = self.n_var = self.X.shape[1]
        self.x_design_info, self.y_design_info = self.model_data.design_info
        self.formula = formula
        self.data = data

    @staticmethod
    def _loglike(params, data, reml=True):
        beta, logvar = params[:-1], params[-1]
        tau = np.exp(-logvar)
        X, y = data
        n, p = X.shape
        n = n - p if reml else n
        const = (-n / 2) * (LN2PI + logvar)
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
        grad[-1] = (-n / 2) + (tau * np.dot(resid.T, resid)) / 2.0
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
        hess[:-1, -1] = -tau * np.dot(X.T, resid)
        hess[-1, :-1] = -tau * np.dot(X.T, resid)
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
        sse = y.T.dot(y) - w.T.dot(w)
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
        sse = y.T.dot(y) - w.T.dot(w)
        beta = np.dot(Linv.T, w)
        Ginv = np.dot(Linv.T, Linv)
        n = n - p if reml else n
        params = np.zeros(len(beta)+1)
        params[:-1], params[-1] = beta, np.log(sse / n)
        return G, Ginv, L, Linv, sse, beta, params

    def _fit(self, reml=True):
        G, Ginv, L, Linv, sse, beta, params = self._fit2(
            None, (self.X, self.y), reml=reml)
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
        self.params_labels = list(self.xcols)+["log_scale"]

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
        
    
    def _bootstrap(self, n_boot, verbose=True):
        pbar = tqdm.tqdm(total=n_boot, smoothing=0.001) if verbose else None
        params = np.zeros((n_boot, self.p+1))
        i = 0
        while i < n_boot:
            try:
                ix = np.random.choice(self.n, self.n)
                Xb, yb = self.X[ix],  self.y[ix]
                G, c = Xb.T.dot(Xb), Xb.T.dot(yb)
                L = np.linalg.cholesky(G)
                w = sp.linalg.solve_triangular(L, c, lower=True)
                params[i, :-1] = sp.linalg.solve_triangular(L.T, w, lower=False)
                params[i, -1] =  (yb.T.dot(yb) - w.T.dot(w)) 
                i += 1
                if verbose:
                    pbar.update(1)
            except np.linalg.LinAlgError:
                pass
        if verbose:
            pbar.close()
        return params
    
    @staticmethod
    @numba.jit(nopython=True,parallel=True)
    def bootstrap_chol(X, y, params, n_boot):
        n = X.shape[0]
        for i in numba.prange(n_boot):
            ii = np.random.choice(n, n)
            Xboot = X[ii]
            yboot = y[ii]
            G = np.dot(Xboot.T, Xboot)
            c = np.dot(Xboot.T, yboot)
            L = np.linalg.cholesky(G)
            w = np.linalg.solve(L, c)#numba_dtrtrs(L, c, "L")
            params[i, :-1] = np.linalg.solve(L.T, w).T#numba_dtrtrs(L.T, w).T
            params[i, -1] = (np.dot(yboot.T, yboot) - np.dot(w.T, w))[0, 0]
        return params

    def _bootstrap_jitted(self, n_boot):
        params = np.zeros((n_boot, self.p+1))
        params = self.bootstrap_chol(self.X, self.y.reshape(-1, 1), params, n_boot)
        return params
    
    

class GLM(RegressionMixin, LikelihoodModel):

    def __init__(self, formula=None, data=None, X=None, y=None,
                 family=Gaussian, scale_estimator="M", *args,
                 **kwargs):
        super().__init__(formula=formula, data=data, X=X, y=y, *args, **kwargs)
        self.xinds, self.yinds = self.model_data.indexes
        self.xcols, self.ycols = self.model_data.columns
        self.X, self.y = self.model_data
        self.n = self.n_obs = self.X.shape[0]
        self.p = self.n_var = self.X.shape[1]
        self.x_design_info, self.y_design_info = self.model_data.design_info
        self.formula = formula
        self.data = data
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
        data = (self.X, self.y, None) if data is None else data
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
        data = (self.X, self.y, None) if data is None else data
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
        data = (self.X, self.y, None) if data is None else data
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
        data = (self.X, self.y, None) if data is None else data
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
        data = (self.X, self.y, None) if data is None else data
        s = self.scale_estimator if scale_estimator is None\
            else scale_estimator
        f = self.f if f is None else f
        ll = self._hessian(params=params, data=data, scale_estimator=s, f=f)
        return ll

    def _optimize(self, t_init=None, opt_kws=None, data=None, s=None, f=None):
        t_init = self.params_init if t_init is None else t_init
        data = (self.X, self.y, None) if data is None else data
        s = self.scale_estimator if s is None else s
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
        self.chi2 = f.pearson_chi2(y, mu=mu, phi=1.0, dispersion=dispersion)
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

        self.llf = self.f.full_loglike(y, mu=mu, phi=phi, dispersion=dispersion)
        if self.f.name == "NegativeBinomial":
            opt_null = self._optimize(t_init=np.zeros(
                2), data=(np.ones((self.n, 1)), self.y, None))
            self.lln = self.full_loglike(
                opt_null.x, data=(np.ones((self.n, 1)), self.y, None))
        else:
            self.lln = self.f.full_loglike(
                y, mu=np.ones(mu.shape[0])*y.mean(), phi=phi)

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
        sumstats["Deviance"] = np.sum(f.deviance(y=y, mu=mu, phi=1.0, dispersion=dispersion))
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
    
    
    def _bootstrap(self, n_boot=1000, verbose=True, rng=None):
        rng = np.random.default_rng() if rng is None else rng 
        n, q = self.n_obs, self.n_params
        weights = self.f.weights
        w = weights if type(weights) is np.ndarray else np.ones(n)
        X, y = self.X, self.y
        pbar = tqdm.tqdm(total=n_boot) if verbose else None
        boot_samples = np.zeros((n_boot, q))
        for i in range(n_boot):
            ii = rng.choice(n, size=n, replace=True)
            opt = self._optimize(data=(X[ii], y[ii], w[ii]), f=self.f)
            boot_samples[i] = opt.x
            pbar.update(1)
        if verbose:
            pbar.close()
        boot_samples = pd.DataFrame(boot_samples,columns=self.param_labels)

        return boot_samples
    
