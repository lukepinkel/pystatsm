# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 23:24:03 2022

@author: lukepinkel
"""

import tqdm  # analysis:ignore
import numba # analysis:ignore
import numpy as np # analysis:ignore
import scipy as sp # analysis:ignore
import scipy.stats # analysis:ignore
from .eln_utils import crossval_mats
from ..utilities.func_utils import soft_threshold, expit


@numba.jit(nopython=True)
def penalty_func(b, alpha, halc, lam):
    pen = lam * (alpha * np.sum(np.abs(b)) + halc * np.dot(b, b))
    return pen


@numba.jit(nopython=True)
def binomial_deviance(y, mu, weights):
    ixa = y==0
    ixc = y==1
    ixb = (~ixa)&(~ixc)
    d = np.zeros_like(y)
    u = (1 - y)[ixb]
    v = (1 - mu)[ixb]
    d[ixa] = -np.log(1-mu[ixa])
    d[ixc] = -np.log(mu[ixc])
    d[ixb] = y[ixb]*np.log(y[ixb]/mu[ixb]) + u*np.log(u/v)
    return 2*weights*d

@numba.jit(nopython=True)
def cv_dev(betas, y, X, weights):
    dev = np.zeros(betas.shape[0])
    for i in range(betas.shape[0]):
        eta = X.dot(betas[i, 1:])+betas[i, 0]
        mu = expit(eta)
        dev[i] = np.sum(binomial_deviance(y, mu, weights))
    return dev

@numba.jit(nopython=True)
def objective_function(y, mu, weights, b, alpha, halc, lam):
    dev = np.sum(binomial_deviance(y, mu, weights))
    pen = penalty_func(b, alpha, halc, lam)
    f = dev / 2.0 + pen
    return f

@numba.jit(nopython=True)
def _welnet_cd(X, Xsq, r, w, alpha, lam, beta, active, index):
    active_vars = index[active]
    xv = np.zeros(X.shape[1])
    la, dla = lam * alpha, lam * (1.0 - alpha)
    b0, b = beta[0], beta[1:]
    n = len(r)
    dlx = 0.0
    for j in active_vars:
        bj, xj, xjsq = b[j], X[:, j], Xsq[:, j]
        xwx = np.sum(w * xjsq) / n
        gj = np.sum(r * xj) / n
        u = gj + xwx * bj
        b[j] = soft_threshold(u, la) / (xwx+dla)
        if b[j]!=bj:   
            d = b[j] - bj
            xv[j] = d**2 * xwx
            r = r - d * w * xj
            dlx = max(xv[j]*d**2, dlx)
        if b[j]==0:
            active[j] = False
    d = np.sum(r) / np.sum(w)
    b0 = b0 + d
    r = r - d * w
    return b, b0, active, dlx
    
    
def make_start_params(X, y, weights, n_lambdas, alpha, lmin=1e-4):
    wmean = np.dot(weights, y) / np.sum(weights)
    pmean =  np.log(wmean /(1.0 - wmean))
    mu = np.ones_like(y) * wmean
    eta = np.ones_like(y) * pmean

    r = y - mu
    nweights = weights / np.sum(weights)
    exp_eta = np.exp(eta)
    dmu_deta = exp_eta / (1.0 + exp_eta)**2
    var_mu = mu *  (1.0 - mu)

    g = X.T.dot(r / var_mu * dmu_deta * nweights)
    lambda_max = np.max(np.abs(g)) / alpha

    lambda_min = lmin * lambda_max

    lams = np.exp(np.linspace(np.log(lambda_max), np.log(lambda_min), n_lambdas))
    return lams


@numba.jit(nopython=True)
def _binom_elnet_cd(lam, beta, active, X, Xsq, y, weights, alpha, halc, index, 
                    n_iters=100, dtol=1e-9, store_betas=False, max_halves=50):
    fvals = np.zeros((n_iters,))
    step_halves = np.zeros((n_iters,), dtype=numba.int32)
    if store_betas:
        betas = np.zeros((n_iters, len(beta)))
        betas[0] = beta
    else:
        betas = None
    eta = X.dot(beta[1:])+beta[0]
    mu = expit(eta)
    r = y - mu
    w = mu * (1.0 - mu)
    fvals[0] = objective_function(y, mu, w, beta[1:], alpha, halc, lam)
    for i in range(1, n_iters):
        b, b0, active, dlx = _welnet_cd(X, Xsq, r, w, alpha, lam, beta.copy(),
                                        active, index)
        eta = X.dot(b)+b0
        mu = expit(eta)
        r = y - mu
        w = mu * (1.0 - mu)
        fvals[i] = objective_function(y, mu, w, b, alpha, halc, lam)
        ii = 0
        if fvals[i]>fvals[i-1]+1e-7:
            while fvals[i]>fvals[i-1]+1e-7:
                ii += 1
                if ii>max_halves:
                    break
                b = (beta[1:] + b) / 2
                b0 = (beta[0] + b0) / 2
                eta = X.dot(b)+b0
                mu = expit(eta)
                w = mu * (1.0 - mu)
                fvals[i] = objective_function(y, mu, w, b, alpha, halc, lam)
        step_halves[i] = ii
        if store_betas:
            betas[i, 1:], betas[i, 0] = b, b0
        if abs(dlx)<1e-9:
            fvals = fvals[:i+1]
            step_halves = step_halves[:i+1]
            if store_betas:
                betas = betas[:i+1]
            break
        beta[1:], beta[0] = b, b0    
    return beta, fvals, betas, step_halves        
    

def binom_elnet_path(X, y, weights, alpha, lambdas, cd_kws=None, progress_bar=None):
    n_lambdas = len(lambdas)
    wmean = np.dot(weights, y) / np.sum(weights)
    b0 =  np.log(wmean /(1.0 - wmean))
    halc = (1 - alpha) / 2.0
    Xsq = X**2
    n_obs, n_var = X.shape
    betas = np.zeros((n_lambdas+1, n_var+1))
    fvals_hist = []
    fvals = np.zeros(n_lambdas+1)
    active = np.ones(n_var, dtype=bool)
    index = np.arange(n_var)
    weights = np.ones_like(y)
    betas[0, 0] = b0
    
    default_cd_kws = dict(X=X, Xsq=Xsq, y=y, weights=weights, alpha=alpha,
                          halc=halc, index=index, n_iters=100, dtol=1e-9, 
                          store_betas=False, max_halves=50)
    cd_kws = {} if cd_kws is None else cd_kws
    cd_kws = {**default_cd_kws, **cd_kws}
    
    pbar = tqdm.tqdm(total=n_lambdas, smoothing=0.0001) if progress_bar is None else progress_bar
    for i, lam in enumerate(lambdas):
        active = np.ones(n_var, dtype=bool)
        beta, fval, _, _ = _binom_elnet_cd(lam, betas[i], active, **cd_kws)
        betas[i+1] = beta
        fvals_hist.append(fval)
        fvals[i+1] = fval[-1]
        pbar.update(1)
    if progress_bar is None:
        pbar.close()
    return betas[1:], fvals[1:], fvals_hist
        

def binom_glmnet_cv(X, y, alpha=0.99, lambdas=None, cv=10, progress_bar=None):
    if (lambdas is None) or (type(lambdas) in [int, float]):
        if lambdas is None:
            n_lambdas = 150
        else:
            n_lambdas = int(lambdas)
        lambdas = make_start_params(X, y, np.ones_like(y), n_lambdas, alpha)
    else:
        n_lambdas = len(lambdas)
    n_obs, n_vars = X.shape
    Xf, yf, Xt, yt = crossval_mats(X, y, X.shape[0], cv, categorical=True)
    betas = np.zeros((cv, n_lambdas,  1+n_vars))
    fvals = np.zeros((n_lambdas, cv))
    pbar_kws = dict(total=n_lambdas*10, mininterval=0.01, maxinterval=1, smoothing=0.00001)
    pbar = tqdm.tqdm(**pbar_kws) if progress_bar is None else progress_bar
    for i in range(cv):
        betas[i], _, _ = binom_elnet_path(Xf[i], yf[i], np.ones_like(yf[i]), alpha, lambdas,
                                          progress_bar=pbar)
        w  = np.ones_like(yt[i])
        w /= np.sum(w)
        fvals[:, i] = cv_dev(betas[i], yt[i], Xt[i], w)
    if progress_bar is None:
        pbar.close()
    return betas, fvals, lambdas