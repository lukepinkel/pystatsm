#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 23:16:43 2020

@author: lukepinkel
"""
import patsy
import numpy as np
import scipy as sp
import scipy.stats
import pandas as pd
from ..utilities.data_utils import _check_type, _check_shape
from ..utilities.linalg_operations import lsqr, nwls
from .dispersion_estimators import mad

from .m_estimators import (Huber, Bisquare) # analysis:ignore



class RobustLM:
    
    def __init__(self, formula, data, c=1.345):
        self.Ydf, self.Xdf = patsy.dmatrices(formula, data, 
                                             return_type='dataframe')
        
        self.X, self.xcols, self.xix, _ = _check_type(self.Xdf)
        self.Y, self.ycols, self.yix, _ = _check_type(self.Ydf)
        self.y = _check_shape(self.Y, 1)
        self.n_obs, self.n_var = self.X.shape
        
        beta_init = lsqr(self.X, self.y)
        scale_init = np.log(mad(self.y - self.X.dot(beta_init)))
        self.theta_init = np.r_[beta_init, scale_init]
        self.a = 0.5
        self.f = Huber()

        
    def _update_y(self, y):
        self.y = y
        beta_init = lsqr(self.X, self.y)
        scale_init = np.log(mad(self.y - self.X.dot(beta_init)))
        self.theta_init = np.r_[beta_init, scale_init]
        
    def _optimize(self, opt_kws=None):
        if opt_kws is not None:
            if "options" in opt_kws.keys():
                opt_options = opt_kws.pop("options")
            else:
                opt_options = {}
        else:
            opt_kws = {}
            opt_options = {}
        default_opt_options = dict(xtol=1e-14, gtol=1e-9)
        opt_options = {**default_opt_options, **opt_options}
        default_opt_kws = dict(fun=self.rho,
                               x0=self.theta_init,
                               jac=self.grad,
                               hess=self.hessian,
                               method="trust-constr",
                               options=opt_options)
        
        opt_kws = {**default_opt_kws, **opt_kws}
        opt = sp.optimize.minimize(**opt_kws)
        if not opt.success:
            x, _ = self.irwls()
            opt = {}
        else:
            x = opt.x
        return opt, x
        
        
    def rho(self, theta, a=None):
        a = self.a if a is None else a
        b, s = theta[:-1], np.exp(theta[-1])
        r = self.y - self.X.dot(b)
        f = np.sum(self.f.rho(r / s) * s + s*a)
        return f
    
    def grad(self, theta, a=None):
        a = self.a if a is None else a
        b, s = theta[:-1], np.exp(theta[-1])
        r = self.y - self.X.dot(b)
        u = r / s
        psi = self.f.psi(u)
        gb = -self.X.T.dot(psi)
        gs = np.sum(self.f.rho(u) - psi * u + a) * s
        g = np.r_[gb, gs]
        return g
    
    def hessian(self, theta, a=None):
        a = self.a if a is None else a
        b, s = theta[:-1], np.exp(theta[-1])
        r = self.y - self.X.dot(b)
        u = r / s
        rho = self.f.rho(u)
        psi = self.f.psi(u)
        phi = self.f.phi(u)
        Hbb = (self.X * phi[:, None]).T.dot(self.X) / s
        Hbs = self.X.T.dot(phi * r / s)
        Hss = np.sum(phi * r**2 / s + (rho - psi * u + a) * s)
        
        H = np.zeros((theta.size, theta.size))
        H[:-1, :-1] = Hbb
        H[-1, :-1] = H[:-1, -1] = Hbs
        H[-1, -1] = Hss
        return H
    
    def irwls(self, scale="mad", theta_init=None, n_iters=200, tol=1e-12):
        theta_init = self.theta_init if theta_init is None else theta_init
        b, s = theta_init[:-1], np.exp(theta_init[-1])
        r = self.y - self.X.dot(b)
        u = r / s
        fit_hist = np.zeros((n_iters, 3))
        for i in range(n_iters):
            if scale == "mad":
                s = mad(r)
            else:
                v = (u * self.f.psi(u) -  self.f.rho(u)) * s**2
                s = np.sqrt(np.sum(v) / ((self.n_obs - self.n_var) * self.E_psi2))
            u = r /s
            w = np.sqrt(self.f.psi(u) / u)
            b_new = nwls(self.X, self.y, w)
            r_new = self.y - self.X.dot(b_new)
            
            d = np.sqrt(np.sum((b - b_new)**2) / (np.sum(b**2)+1e-16))
            fit_hist[i] = [d, np.sum(self.grad(np.r_[b_new, np.log(s)])**2),
                           self.rho(np.r_[b_new, np.log(s)])]
            if d < tol:
                break
            b = b_new
            r = r_new
        fit_hist = fit_hist[:i]
        return np.r_[b, np.log(s)], fit_hist
    
    def fit(self, opt_kws=None):
        default_opt_kws = dict(method="trust-constr")
        opt_kws = {} if opt_kws is None else opt_kws
        opt_kws = {**default_opt_kws, **opt_kws}
        self.opt = sp.optimize.minimize(self.rho, self.theta_init,
                                            jac=self.grad, hess=self.hessian,
                                            **opt_kws)
        self.theta = self.opt.x
        self.beta = self.theta[:-1]
        self.scale = np.exp(self.theta[-1])
        self.resids = self.y - self.X.dot(self.beta)
        
        self.psi_r = self.f.psi(self.resids)
        self.phi_r = self.f.phi(self.resids)
        
        self.dfe = (self.n_obs-self.n_var)
        
        self.E_psi_sq = np.sum(self.psi_r**2) / self.dfe
        self.E_phi = np.sum(self.phi_r) / self.n_obs
        
        self.k = 1.0 + self.n_var / self.n_obs * np.var(self.phi_r) / self.E_phi**2
        
        self.k1 = self.k**2 * self.E_psi_sq / self.E_phi**2
        self.k2 = self.k * self.E_psi_sq / self.E_phi
        self.k3 = self.E_psi_sq / self.k
        
        self.XtX = self.X.T.dot(self.X)
        self.G = np.linalg.inv(self.XtX)
        self.W = np.linalg.inv((self.X * self.phi_r[:, None]).T.dot(self.X))
        
        self.H1 = self.k1 * self.G
        self.H2 = self.k2 * self.W
        self.H3 = self.k3 * self.W.dot(self.XtX).dot(self.W)
        
        self.se_beta = np.vstack((np.sqrt(np.diag(self.H1)),
                                  np.sqrt(np.diag(self.H2)),
                                  np.sqrt(np.diag(self.H3)))).T
        
        self.t_values = self.beta[:, None] / self.se_beta 
        self.p_values = 2.0*sp.stats.t(self.dfe).sf(np.abs(self.t_values))
        self.res = pd.DataFrame(np.hstack((self.beta[:, None], 
                                           self.se_beta,
                                           self.t_values, 
                                           self.p_values)))
        self.res.columns = ["beta", "SE1", "SE2", "SE3", "t1", "t2",
                            "t3", "p1", "p2", "p3"]
        self.res.index = self.xcols
        
        self._rho_mu = np.sum(self.f.rho((self.y - np.median(self.y)) / self.scale))
        self._rho_yh = np.sum(self.f.rho(self.resids / self.scale))
        self.rsquared = (self._rho_mu - self._rho_yh) / self._rho_mu
        self.deviance = 2.0 * self.scale**2 * self._rho_yh
        self.pdim = 2.0 * self.E_psi_sq / self.E_phi
        self.aicr = 2.0 * self._rho_yh + self.pdim * self.n_var
        self.bicr = 2.0 * self._rho_yh + np.log(self.n_obs) * self.n_var
        self.sumstats = pd.DataFrame(np.array([self.rsquared,
                                               self.deviance,
                                               self.aicr,
                                               self.bicr]),
                                     index=["r2", "dev", "AICr", "BICr"])
        
        
        
        


class RobustSLM:
    
    def __init__(self, formula, data, c=1.548):
        self.Ydf, self.Xdf = patsy.dmatrices(formula, data, 
                                             return_type='dataframe')
        
        self.X, self.xcols, self.xix, _ = _check_type(self.Xdf)
        self.Y, self.ycols, self.yix, _ = _check_type(self.Ydf)
        self.y = _check_shape(self.Y, 1)
        self.n_obs, self.n_var = self.X.shape
        
        self.beta_init = lsqr(self.X, self.y)
        self.scale_init = np.log(mad(self.y - self.X.dot(self.beta_init)))
        self.f = Bisquare(c=c)
    
    def _s_est_eq(self, r, s, bd=0.5):
        d = np.mean(self.f.rho(r / s)) - bd
        return d
    
    def fit(self, n_iters=200, n_refining=2, n_subsample=5, bd=0.5, tol=1e-12, 
            rng=None):
        rng = np.random.default_rng(123) if rng is None else rng
        X, y = self.X, self.y
        n_obs, n_var = self.n_obs, self.n_var
        beta, scale = self.beta_init, self.scale_init
        r = y - X.dot(beta)
        r, beta, scale = self.f.irls_step(X, y, beta, scale, bd=bd, n_iters=n_refining,
                                          tol=tol)
        betas = np.zeros((n_subsample, len(beta)))
        scales= np.zeros((n_subsample, )) + 1e12
        s_max = np.inf
        for i in range(n_iters):
            for _ in range(100):
                ix = rng.choice(n_obs, n_var, replace=False)
                Xi, yi = X[ix], y[ix]
                if np.linalg.matrix_rank(Xi)==Xi.shape[1]:
                    break
            beta = lsqr(Xi, yi)
            if n_refining>0:
                rc, bc, sc = self.f.irls_step(X, y, beta, bd=bd, n_iters=n_refining)
            else:
                rc, bc = y-X.dot(beta), beta
                sc = mad(rc)
                

            if i>0:
                d = self._s_est_eq(rc, s_max, bd)
                if d < 0:
                    s_min = self.f.m_estimate_scale(rc, sc, bd, n_iters=200)
                    
                    ii = np.argsort(scales)[-1]
                    scales[ii] = s_min
                    betas[ii] = bc
                    s_max = np.max(scales)
                    
            else:
                sc = self.f.m_estimate_scale(rc, sc, bd, n_iters=200)
                scales[n_subsample-1] = sc
                betas[n_subsample-1] = bc
        s_min = np.inf
        b_min = beta
        for i in range(n_subsample-1, -1, -1):
            rc, bc, sc = self.f.irls_step(X, y, betas[i], scales[i], 
                                          bd=bd, n_iters=80)
            if sc < s_min:
                s_min = sc
                b_min = bc
        return s_min, b_min
                    
                
    
                    
                    
                
                
                
                
            

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    