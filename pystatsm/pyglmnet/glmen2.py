#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 15:19:16 2022

@author: lukepinkel
"""

import tqdm  # analysis:ignore
import numba # analysis:ignore
import numpy as np # analysis:ignore
import scipy as sp # analysis:ignore
import scipy.stats # analysis:ignore
import pandas as pd
from .eln_utils import crossval_mats, kfold_indices
from ..utilities.func_utils import soft_threshold
from ..pyglm.families import Binomial, Gaussian
from ..pyglm.links import LogitLink
from ..utilities.linalg_operations import wdcrossp, wcrossp

def overflow_adjust(x, lower=-36, upper=36):
    x[x<lower] = lower
    x[x>upper] = upper
    return x


@numba.jit(nopython=True)
def _coordinate_descent_cycle(b, b0, r, v, X, Xsq, xv, xv_ind, active, index, 
                              lam, alpha):
    la, dla = lam * alpha, lam * (1.0 - alpha)
    active_vars = index[active]
    dlx = 0.0
    xmz = np.sum(v)
    for j in active_vars:
        bj, xj, xjsq = b[j], X[:, j], Xsq[:, j]
        if xv_ind[j]==False:
            xv[j], xv_ind[j] = np.dot(xjsq, v), True
        gj = np.dot(xj, r)
        u = gj + xv[j] * bj
        b[j] = soft_threshold(u, la) / (xv[j]+dla)
        d = b[j] - bj
        if abs(b[j]) ==0:
            b[j] = 0.0
            active[j] = False
        if abs(d)>0:
            r = r - d * v * xj
            dlx = max(xv[j] * d**2, dlx)
            d = np.sum(r) / xmz
            b0 = b0 + d
            r = r - d * v
            dlx = max(dlx, xmz * d**2)
    return b, b0, active, dlx, r

@numba.jit(nopython=True)
def weighted_elnet(beta, y, v, X, Xsq, active, index, lam, lam0, alpha,
                   max_iters, dtol=1e-21):
    r = v * y
    xv = np.zeros(X.shape[1], dtype=numba.float64)
    xv_ind = np.zeros(X.shape[1], dtype=numba.boolean)
    #g = np.abs(np.dot(X.T, r-np.mean(r)))
    #tlam = alpha * (2.0 * lam - lam0)
    #active[g<tlam] = False
    for i in range(max_iters):
        beta[1:], beta[0], active, dlx, r = _coordinate_descent_cycle(beta[1:],
                                                                      beta[0], 
                                                                      r, v, X, 
                                                                      Xsq, xv, 
                                                                      xv_ind,
                                                                      active,
                                                                      index,
                                                                      lam, 
                                                                      alpha)
        if dlx<dtol:
            break
    return beta, i, xv
   


class ElasticNetGLM(object):
    
    def __init__(self, X, y, family, wobs=None, alpha=0.99):
        n, p = X.shape
        
        wobs = np.ones(n, dtype=np.float64) if wobs is None else wobs
        wobs = wobs / np.sum(wobs)
        self.X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        self.y = y
        self.n_obs = self.n = n
        self.n_var = self.p = p
        self.wobs = wobs
        self.family = family
        self.alpha = alpha
        self.Xsq = X**2
        self.halc = (1.0 - alpha) / 2.0
        
    def _dev_mu(self, mu, y=None, wobs=None):
        y = self.y if y is None else y
        wobs = self.wobs if wobs is None else wobs
        return np.sum(self.family.deviance(y=y, mu=mu, weights=wobs))
    
    
    def _get_eta(self, beta, X=None):
        X = self.X if X is None else X
        eta = overflow_adjust(np.dot(X, beta[1:])+beta[0])
        return eta
    
    def _get_mu(self, beta, X=None):
        return self.family.inv_link(self._get_eta(beta, X))
    
    def _dev_beta(self, beta, X=None, y=None, wobs=None):
        eta = self._get_eta(beta, X)
        mu = self.family.inv_link(eta)
        return self._dev_mu(mu, y, wobs)
    
    
    def pen_func(self, beta, lam):
        b = beta[1:]
        pen = lam * (self.alpha * np.sum(np.abs(b)) + self.halc * np.dot(b, b))
        return pen
    
    def obj_func(self, beta, lam, X=None, y=None, wobs=None):
        mu = self._get_mu(beta, X)
        dev = self._dev_mu(mu, y, wobs) / 2.0
        pen = self.pen_func(beta, lam)
        f = dev + pen
        return f
    
    def _obj_func_mu(self, mu, beta, lam, X, y, wobs):
        dev = self._dev_mu(mu, y, wobs) / 2.0
        pen = self.pen_func(beta, lam)
        return dev + pen
    
    def pen_grad(self, beta, lam):
        b = beta[1:]
        g = lam * (1.0 - self.alpha) * b + np.sign(b) * lam * self.alpha
        return g
        
    def dev_grad(self, beta, lam, X=None, y=None, wobs=None):
        mu = self._get_mu(beta, X)
        y = self.y if y is None else y
        wobs = self.wobs if wobs is None else wobs
        dll = -wdcrossp(X, wobs, y-mu)
        return dll
    
    
    def obj_grad(self, beta, lam, X=None, y=None, wobs=None):
        X = self.X if X is None else X
        y = self.y if y is None else y
        wobs = self.wobs if wobs is None else wobs
        b = beta[1:]
        mu = self._get_mu(beta, X)
        dL = -wdcrossp(X, wobs, y-mu)
        dP = lam * (1.0 - self.alpha) * b + np.sign(b) * lam * self.alpha
        dF = np.r_[-np.dot(wobs, y-mu), (dL + dP)]
        return dF
    
    def dev_hess(self, beta, lam, X=None, y=None, wobs=None):
        X = self.X if X is None else X
        y = self.y if y is None else y
        wobs = self.wobs if wobs is None else wobs
        mu = self._get_mu(beta, X)
        v = wobs * self.family.get_w(y, mu)
        H = np.zeros((self.n_var+1, self.n_var+1))
        Xv = X * v[:, None]
        H[1:, 1:] = np.dot(Xv.T, X)
        H[1:, 0] = H[0, 1:] = np.sum(Xv, axis=0)
        H[0, 0] = np.sum(v)
        return H*2.0
    
    
    def penalty_hess(self, beta, lam):
        H = np.diag(np.r_[0.0, np.ones(self.n_var) * lam * (1.0 - self.alpha)])
        return H


    def obj_hess(self, beta, lam, X=None, y=None, wobs=None):
        HL = self.dev_hess(beta, lam, X, y, wobs)
        HP = self.penalty_hess(beta, lam)
        H = HL  / 2.0 + HP
        return H
    
    def get_intercept(self, y=None, wobs=None):
        y = self.y if y is None else y
        wobs = self.wobs if wobs is None else wobs
        beta = np.zeros(self.n_var+1)
        ymean = np.dot(wobs, y)
        beta[0] = self.family.link(ymean)
        return beta

    def get_start_param(self, n_lambdas=200, alpha_min=1e-3, lambda_min_ratio=1e-4):
        beta = np.zeros(self.n_var+1)
        ymean = np.dot(self.wobs, self.y)
        mu = ymean * np.ones(self.n_obs)
        beta[0] = self.family.link(ymean)
        r = self.y - mu
        eta = self.family.link(mu)
        var = self.family.var_func(mu)
        dmu_deta = self.family.dinv_link(eta)
        rv = r  / var * dmu_deta * self.wobs
        g = np.dot(self.X.T, rv)  / max(self.alpha, alpha_min)
        lmax = np.max(np.abs(g))
        lambdas = np.exp(np.linspace(np.log(lmax), np.log(lmax*lambda_min_ratio), n_lambdas))
        null_dev = self._dev_mu(mu)
        return lambdas, null_dev, beta, eta, mu
    
    
    def crossval_path(self, betas, lambdas, X, y, wobs):
        n = betas.shape[0]
        fvals = np.zeros(n)
        for i in range(n):
            mu = self._get_mu(betas[i], X)
            fvals[i] =  np.sum(self.family.deviance(y=y, mu=mu, weights=wobs))
        return fvals
    
    def _glm_elnet(self, beta, X, Xsq, y, wobs, lam, lam0=None, active=None, n_iters=1000,
                   ftol=1e-8, lower=-36, upper=36, inner_kws=None):
        default_inner_kws = dict(max_iters=1000, dtol=1e-8)
        inner_kws = {} if inner_kws is None else inner_kws
        inner_kws = {**default_inner_kws, **inner_kws}
        lam0 = lam if lam0 is None else lam0
        
        active = np.ones(self.n_var, dtype=bool) if active is None else active
        index = np.arange(self.n_var)
        eta = overflow_adjust(beta[0] + X.dot(beta[1:]), lower, upper)
        mu = self.family.inv_link(eta)
        f_curr = self._obj_func_mu(mu, beta, lam, X, y, wobs)
        f_hist = [[f_curr, 0]]
        for i in range(n_iters):
            var_mu = self.family.var_func(mu=mu)
            dmu_deta = self.family.dinv_link(eta)
            z = eta + (y - mu) / dmu_deta
            v = wobs * dmu_deta**2 / var_mu
            r = z - eta
            beta_new, i, xv = weighted_elnet(beta.copy(), r, v, X, Xsq, active, index,
                                             lam, lam0, self.alpha, **inner_kws)
            eta_new = overflow_adjust(beta_new[0] + X.dot(beta_new[1:]), lower, upper)
            mu_new = self.family.inv_link(eta_new)
            f_new = self._obj_func_mu(mu_new, beta_new, lam, X, y, wobs)
            f_hist.append([f_new, i])
            fdiff = np.abs(f_new - f_curr) / (0.1 + np.abs(f_new))
            if fdiff<ftol:
                beta = beta_new
                break
            else:
                eta, mu, beta, f_curr = eta_new, mu_new, beta_new, f_new
        return beta, f_hist#, np.vstack(beta_hist)
    
    def _glm_elnet_path(self, X, Xsq, y, wobs, lambdas, progress_bar=None, 
                        kws=None, warm_start=True):
        n_lambdas = len(lambdas)
        betas = np.zeros((n_lambdas+1, self.n_var+1))
        fvals = np.zeros((n_lambdas+1, 2))
        fvals_hist = []
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        Xsq = X**2
        default_kws = dict(n_iters=1000, ftol=1e-8, lower=-36, upper=36, inner_kws=None)
        kws = {} if kws is None else kws
        kws = {**default_kws, **kws}
        betas[0] = self.get_intercept(y, wobs)
        pbar_kws = dict(total=n_lambdas, smoothing=1e-3, miniters=1,
                        maxinterval=1e-3)
        pbar = tqdm.tqdm(**pbar_kws) if progress_bar is None else progress_bar
        lam0 = lambdas[0]
        betas[-1] = betas[0]
        for i, lam in enumerate(lambdas):
            active = np.ones(self.n_var, dtype=bool)
            beta_start = betas[i-1].copy() if  warm_start else betas[i-1].copy()*0.0
            beta, fval = self._glm_elnet(beta_start, X, Xsq, y, wobs, lam, lam0,
                                         active,  **kws)
            betas[i+1] = beta
            fvals_hist.append(fval)
            fvals[i+1] = fval[-1]
            pbar.update(1)
        if progress_bar is None:
            pbar.close()
        return betas[1:], fvals[1:], fvals_hist
    
    def _glm_elnet_cv(self, X, Xsq, y, wobs, lambdas=None, cv=10, n_rep=1, 
                      kws=None, progress_bar=None):
        n_lambdas = len(lambdas)
        betas = np.zeros((n_rep, cv, n_lambdas,  1+self.n_var))
        fvals = np.zeros((n_rep, n_lambdas, cv,))
        pbar_kws = dict(total=n_lambdas*cv*n_rep, smoothing=1e-4)
        pbar = tqdm.tqdm(**pbar_kws) if progress_bar is None else progress_bar
        rng = np.random.default_rng()
        randomize = False
        for j in range(n_rep):
            if j>0:
                randomize = True
            kfix = kfold_indices(self.n_obs, cv, y, categorical=True, randomize=randomize, 
                                 random_state=rng)
            for k, (f_ix, v_ix) in enumerate(kfix):
                Xf, Xsqf, yf, wf = X[f_ix], Xsq[f_ix], y[f_ix], wobs[f_ix]
                Xt, _, yt, wt = X[v_ix], Xsq[v_ix], y[v_ix], wobs[v_ix]
                betas[j, k], _, _ = self._glm_elnet_path(Xf, Xsqf, yf, wf, lambdas,
                                                         progress_bar=pbar, 
                                                         kws=kws)
                fvals[j, :, k]  = self.crossval_path(betas[j, k], lambdas*0.0, Xt, yt, wt)
        if progress_bar is None:
            pbar.close()
        return betas, fvals, lambdas
    
    
    def fit(self, n_lambdas=200, cv=10, n_rep=1, lambda_min_ratio=1e-4, kws=None):
        lambdas, null_dev, b_init, _, _ = self.get_start_param(n_lambdas,
                                                               lambda_min_ratio)
        
        betas, fvals, lambdas = self._glm_elnet_cv(self.X, self.Xsq, self.y, 
                                                   self.wobs, lambdas, cv=cv,
                                                   n_rep=n_rep, kws=kws)
        self.lambdas = lambdas
        self.null_dev = null_dev
        self.beta_path = np.swapaxes(np.swapaxes(betas, 0, 2), 1, 3)
        self.f_paths = np.swapaxes(fvals, 0, 1)
        kws = dict(inner_kws=dict(max_iters=10_000, dtol=1e-12),
                   n_iters=10_000, ftol=1e-12)
        self.betas, _, self.fvals_hist = self._glm_elnet_path(self.X, self.Xsq, self.y, 
                                                self.wobs, lambdas, kws=kws,
                                                warm_start=False)
        
        

        
        


        
    