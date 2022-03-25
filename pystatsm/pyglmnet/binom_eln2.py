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
from .eln_utils import crossval_mats, kfold_indices
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
def cv_dev(betas,  lambdas, alpha, y, X, weights):
    dev = np.zeros((betas.shape[0], 2))
    for i in range(betas.shape[0]):
        eta = X.dot(betas[i, 1:])+betas[i, 0]
        mu = expit(eta)
        dev[i, 0] = np.sum(binomial_deviance(y, mu, weights)) / 2.0
        dev[i, 1] = penalty_func(betas[i, 1:], alpha, (1 - alpha) / 2.0, lambdas[i])
    return dev

@numba.jit(nopython=True)
def objective_function(y, mu, weights, b, alpha, halc, lam):
    dev = np.sum(binomial_deviance(y, mu, weights))
    pen = penalty_func(b, alpha, halc, lam)
    f = dev / 2.0 + pen
    return f

@numba.jit(nopython=True)
def _welnet_cd(X, Xsq, r, w, alpha, lam, beta, active, index, max_iters=1000, 
               dtol=1e-10):
    active_vars = index[active]
    xv = np.zeros(X.shape[1])
    la, dla = lam * alpha, lam * (1.0 - alpha)
    b0, b = beta[0], beta[1:]
    n = len(r)
    xwx = np.zeros(X.shape[1])
    xwx_ind = np.zeros(X.shape[1])
    wsum = np.sum(w)
    for i in range(max_iters):
        dlx = 0.0
        for j in active_vars:
            bj, xj, xjsq = b[j], X[:, j], Xsq[:, j]
            if xwx_ind[j]!=1:
                xwx[j] = np.dot(w, xjsq) / n
                xwx_ind[j] = 1
            gj = np.dot(r, xj) / n
            u = gj + xwx[j] * bj
            b[j] = soft_threshold(u, la) / (xwx[j]+dla)
            if b[j]!=bj:   
                d = b[j] - bj
                xv[j] = d**2 * xwx[j]
                r = r - d * w * xj
                dlx = max(xv[j]*d**2, dlx)
            if b[j]==0:
                active[j] = False
        if i==0:
            active_vars = index[active]
        d = np.sum(r) / wsum
        b0 = b0 + d
        r = r - d * w
        dlx = max(dlx, wsum * d**2)
        if abs(dlx)<dtol:
            break
    return b, b0, active
    
    
def get_intercept(y, weights):
    wmean = np.dot(weights, y) / np.sum(weights)
    b0 =  np.log(wmean /(1.0 - wmean))
    mu = np.ones_like(y) * wmean
    eta = np.ones_like(y) * b0
    return b0, mu, eta
    

def get_lambdas(X, y, weights, n_lambdas, alpha, lmin=1e-4):
    b0, mu, eta = get_intercept(y, weights)

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
def get_eta_mu(X, b, b0):
    eta = X.dot(b)+b0
    mu = expit(eta)
    w = mu * (1.0 - mu)
    return eta, mu, w

@numba.jit(nopython=True)
def _binom_elnet_cd(lam, beta, active, X, Xsq, y, weights, alpha, halc, index, 
                    n_iters=100, dtol=1e-6, ftol=1e-7, store_betas=False, 
                    max_halves=50, max_iters=1000):
    fvals = np.zeros((n_iters,))
    step_halves = np.zeros((n_iters,), dtype=numba.int32)
    if store_betas:
        betas = np.zeros((n_iters, len(beta)))
        betas[0] = beta
    else:
        betas = None
    eta, mu, w = get_eta_mu(X, beta[1:], beta[0])
    weights = weights / np.sum(weights)
    r = (y - mu)
    fvals[0] = objective_function(y, mu, weights, beta[1:], alpha, halc, lam)
    for i in range(1, n_iters):
        b, b0, active = _welnet_cd(X, Xsq, r, w , alpha, lam, beta.copy(),
                                   active, index, max_iters, dtol)
        eta, mu, w = get_eta_mu(X, b, b0)
        r = (y - mu)
        fvals[i] = objective_function(y, mu, weights, b, alpha, halc, lam)
        ii = 0
        if fvals[i]>fvals[i-1]+1e-7:
            while fvals[i]>fvals[i-1]+1e-7:
                ii += 1
                if ii>max_halves:
                    break
                b = (beta[1:] + b) / 2
                b0 = (beta[0] + b0) / 2
                eta, mu, w = get_eta_mu(X, b, b0)
                r = (y - mu)
                fvals[i] = objective_function(y, mu, weights, b, alpha, halc, lam)
        step_halves[i] = ii
        if store_betas:
            betas[i, 1:], betas[i, 0] = b, b0
        beta[1:], beta[0] = b, b0 
        fdiff = np.abs(fvals[i]-fvals[i-1]) / (np.abs(fvals[i]) + 0.1)
        if fdiff < ftol:
            fvals = fvals[:i+1]
            step_halves = step_halves[:i+1]
            if store_betas:
                betas = betas[:i+1]
            break
    return beta, fvals, betas, step_halves     


def binom_elnet(X, y, lam, alpha=0.99, weights=None, cd_kws=None):
    weights = np.ones_like(y) if weights is None else weights
    weights = weights / np.sum(weights)
    Xsq = X**2
    n_obs, n_var = X.shape
    b0, mu, eta = get_intercept(y, weights)
    index = np.arange(n_var)
    halc = (1 - alpha) / 2.0
    active = np.ones(n_var, dtype=bool)
    default_cd_kws = dict(X=X, Xsq=Xsq, y=y, weights=weights, alpha=alpha,
                          halc=halc, index=index, n_iters=100, dtol=1e-9, 
                          ftol=1e-7, store_betas=False, max_halves=50,
                          max_iters=1000)
    cd_kws = {} if cd_kws is None else cd_kws
    cd_kws = {**default_cd_kws, **cd_kws}
    beta = np.zeros(n_var+1)
    beta[0] = b0
    beta, fval, beta_hist, step_halves = _binom_elnet_cd(lam, beta, active, **cd_kws)
    return beta, fval, beta_hist, step_halves
    

def binom_elnet_path(X, y, weights, alpha, lambdas=None, cd_kws=None, progress_bar=None):
    if (lambdas is None) or (type(lambdas) in [int, float]):
        if lambdas is None:
            n_lambdas = 150
        else:
            n_lambdas = int(lambdas)
        lambdas = get_lambdas(X, y, np.ones_like(y), n_lambdas, alpha)
    else:
        n_lambdas = len(lambdas)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    b0, mu, eta = get_intercept(y, weights)
    halc = (1 - alpha) / 2.0
    Xsq = X**2
    n_obs, n_var = X.shape
    betas = np.zeros((n_lambdas+1, n_var+1))
    fvals_hist = []
    fvals, active, index = np.zeros(n_lambdas+1), np.ones(n_var, dtype=bool), np.arange(n_var)
    weights = np.ones_like(y)
    betas[0, 0] = b0
    
    default_cd_kws = dict(X=X, Xsq=Xsq, y=y, weights=weights, alpha=alpha,
                          halc=halc, index=index, n_iters=100, dtol=1e-5, 
                          ftol=1e-4, store_betas=False, max_halves=50,
                          max_iters=20)
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
    return betas[1:], fvals[1:], fvals_hist, lambdas
        

def binom_glmnet_cv(X, y, alpha=0.99, lambdas=None, cv=10, n_rep=1, 
                    cd_kws=None, progress_bar=None):
    if (lambdas is None) or (type(lambdas) in [int, float]):
        if lambdas is None:
            n_lambdas = 150
        else:
            n_lambdas = int(lambdas)
        lambdas = get_lambdas(X, y, np.ones_like(y), n_lambdas, alpha)
    else:
        n_lambdas = len(lambdas)
    n_obs, n_vars = X.shape
    Xf, yf, Xt, yt = crossval_mats(X, y, X.shape[0], cv, categorical=True)
    betas = np.zeros((n_rep, cv, n_lambdas,  1+n_vars))
    fvals = np.zeros((n_rep, n_lambdas, cv, 2))
    pbar_kws = dict(total=n_lambdas*cv*n_rep, mininterval=0.01, maxinterval=1, 
                    smoothing=0.00001)
    pbar = tqdm.tqdm(**pbar_kws) if progress_bar is None else progress_bar
    ind = np.arange(n_obs)
    rng = np.random.default_rng()
    randomize = False
    for j in range(n_rep):
        if j>0:
            randomize = True
        kfix = kfold_indices(n_obs, cv, y, categorical=True, randomize=randomize, 
                             random_state=rng)
        for k, (f_ix, v_ix) in enumerate(kfix):
            Xf, yf = X[f_ix], y[f_ix]
            Xt, yt = X[v_ix], y[v_ix]
            betas[j, k], _, _ , _= binom_elnet_path(Xf, yf, np.ones_like(yf), alpha, lambdas,
                                              progress_bar=pbar, cd_kws=cd_kws)
            w  = np.ones_like(yt)
            w /= np.sum(w)
            fvals[j, :, k] = cv_dev(betas[j, k], lambdas, alpha, yt, Xt, w)
    if progress_bar is None:
        pbar.close()
    return betas, fvals, lambdas