#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 00:46:43 2020

@author: lukepinkel
"""

import tqdm  # analysis:ignore
import numba # analysis:ignore
import numpy as np # analysis:ignore
import scipy as sp # analysis:ignore
import scipy.stats # analysis:ignore
from .eln_utils import crossval_mats

@numba.jit(nopython=True)
def sft(x, t):
    '''
    Soft Thresholding
    Parameters
    ----------
    x : array_like
        Values to be subject to soft thresholding
    t : array_like
        Threshold
    Returns
    -------
    y : array_like
        the input x subject to soft thresholding max(abs(x) - t, 0) * sign(x) 
    '''
    y = np.maximum(np.abs(x) - t, 0.0) * np.sign(x)
    return y

@numba.jit(nopython=True)
def inv_logit(x):
    '''
    Parameters
    ----------
    x : array_like
    
    Returns
    -------
    y : array_like
        The inverse logit of x, exp(x) / (1.0 + exp(x))
    
    '''
    u = np.exp(x)
    y = u / (u + 1.0)
    return y

@numba.jit(nopython=True)
def elnet_penalty(b, alpha, lambda_):
    '''
    Parameters
    ----------
    b : array_like
        Regression coefficients
        
    alpha : float
        The elastic net penalty ratio
    
    lambda_ : float
        The elastic net penalty size
    
    Returns
    -------
    penalty : float
        The elastic net penalty
    
    '''
    l1 = np.abs(b).sum() * alpha
    l2 = np.sum(b**2) * (1.0 - alpha) / 2.0
    penalty = (l1 + l2) * lambda_
    return penalty

@numba.jit(nopython=True)
def binom_eval(y, eta, b, alpha, lambda_):
    '''
    Parameters
    ----------
    y : array_like
        Dependent variable
        
    eta : array_like
        Linear predictor 
        
    b : array_like
        Regression coefficients
        
    alpha : float
        The elastic net penalty ratio
        
    lambda_: float
        The elastic net penalty size
    
    Returns
    -------
    ll : float
        The unpenalized binomial deviance
    
    P : float
        The elastic net penalty
    
    f : float
        The penalized deviance
    '''
    P = elnet_penalty(b, alpha, lambda_)
    mu = inv_logit(eta)
    ll = (y * np.log(mu)).sum() + ((1.0 - y) * np.log(1.0 - mu)).sum()
    ll = ll / y.shape[0]
    f = P - ll
    ll = -2.0 * ll
    return ll, P, f

@numba.jit(nopython=True)
def binom_glm_cd(b, X, Xsq, r, w, xv, la, dla, active, index, n):
    '''
    Binomial GLM Coordinate descent.  This function performs one cycle
    of coordinate descent
    
    
    Parameters
    ----------
    b : array_like
        Regression coefficients
        
    X : array_like
        Regression design matrix/predictor variables/independent variables
        
    Xsq : array_like
        Squared Regression design matrix
        
    y : array_like
        Dependent variable
        
    w : array_like
        The regression weights, which in the case of the binomial glm
        (without other weights) is equal to the model variance (mu (1-mu))
    
    xv: array_like
        array used to store (weighted) coefficient changes for checking 
        convergence
    
    la : float
        The L1 penalty term lambda_ * alpha
    
    dla : float
        The L2 penalty term (1.0 - alpha) * lambda_
        
    active : array_like
        Array containing boolean indicators of variables status
    
    index : array_like
        Array containing int index values for each variable
    
    n : int
        Number of observations

    
    Returns
    -------
    b : array_like
        The updated coefficients
    
    active : array_like
        The updated active set 
    
    xv : array_like
        Weighted coeffecient differences
    
    
    
    '''
    active_vars = index[active]
    for j in active_vars:
        bj, xj, xjsq = b[j], X[:, j], Xsq[:, j]
        xwx = np.sum(w * xjsq) / n
        gj = np.sum(r * xj) / n
        u = gj + xwx * bj
        b[j] = sft(u, la) / (xwx+dla)
        d = b[j] - bj
        xv[j] = (b[j] - bj)**2 * xwx
        if abs(b[j]) <= 1e-12:
            b[j] = 0.0
            active[j] = False
        if abs(d)>0:
            r = r - d * w * xj
    return b, active, xv
    
  
@numba.jit(nopython=True)
def _binom_glmnet(b, X, Xsq, y, la, dla, acs, ix, n, n_iters=2000, 
                  btol=1e-4, dtol=1e-4, pmin=1e-9, nr_ent=False):
    '''
    Binomial glmnet.  This function fits a binomial GLM via a doubly iterative
    outer approximation followed by an inner cycle of coordinate descent. 
    
    Parameters
    ----------
    b : array_like
        Regression coefficients
        
    X : array_like
        Regression design matrix/predictor variables/independent variables
        
    Xsq : array_like
        Squared Regression design matrix
        
    y : array_like
        Dependent variable
    
    la : float
        The L1 penalty term lambda_ * alpha
    
    dla : float
        The L2 penalty term (1.0 - alpha) * lambda_
        
    acs : array_like
        Array containing boolean indicators of variables status
    
    ix : array_like
        Array containing int index values for each variable
    
    n : int
        Number of observations
    
    n_iters : int, optional
        Number iterations, default 2000
        
    btol: float, optional
        Coefficient change tolerance, default 1e-4
    
    dtol: float, optional
        Objective function change tolerance, default 1e-4
    
    pmin: float, optional
        Minimum allowed estimated probability, default 1e-9
    
    nr_ent: bool, optional
        If variable reentry is allowed, default false

    
    Returns
    -------
    b : array_like
        The updated coefficients
    
    acs : array_like
        The updated active set 
    
    fvals : array_like
        Array with three columns, respectively containing the deviance, penalty,
        and deviance+penalty
    '''
    fvals = np.zeros(n_iters+1)
    xv = np.zeros_like(b)
    yconj = 1.0 - y
    for i in range(n_iters):
        eta = X.dot(b)
        mu = inv_logit(eta)
        muconj = 1.0 - mu
        w = mu * muconj
        r = y - mu
        fvals[i] = -2.0*np.sum(y * np.log(mu) + np.log(muconj) * yconj)/n
        b_new, acs_new, xvd = binom_glm_cd(b.copy(), X, Xsq, r, w, xv, la, dla, 
                                      acs.copy(), ix, n)
        
        if np.max(xvd) < btol:
            break
        if i>0 and 0<(fvals[i-1]-fvals[i])<dtol:
            break
        else:
            b = b_new
        if nr_ent:
            acs = acs_new
    return b, acs, fvals[:i]


def binom_glmnet(X, y, lambda_, alpha, b=None, active=None, n_iters=2000, 
                 btol=1e-4, dtol=1e-4, pmin=1e-9, nr_ent=False, 
                 intercept=True):
    '''
    Binomial glmnet.  This function fits a binomial GLM via a doubly iterative
    outer approximation followed by an inner cycle of coordinate descent. 
    
    Parameters
    ----------
   
        
    X : array_like
        Regression design matrix/predictor variables/independent variables

    y : array_like
        Dependent variable
    
    lambda_ : float
        The penalty size
    
    alpha : float
        The penalty ratio
        
    b : array_like, optional
        Regression coefficients, default None
        
    active : array_like, optional
        Array containing boolean indicators of variables status, default None
        
    n_iters : int, optional
        Number iterations, default 2000
        
    btol: float, optional
        Coefficient change tolerance, default 1e-4
    
    dtol: float, optional
        Objective function change tolerance, default 1e-4
    
    pmin: float, optional
        Minimum allowed estimated probability, default 1e-9
    
    nr_ent: bool, optional
        If variable reentry is allowed, default false
        
    
    Returns
    -------
    b : array_like
        The estimated coefficients
    
    active : array_like
        The updated active set 
    
    fvals : array_like
        Array with three columns, respectively containing the deviance, penalty,
        and deviance+penalty
    
    Notes
    -----
    This function calls the JIT-ed function _binom_glmnet after precomputing
    some values and performing other peripheral but necessary operations.
    '''
    n, p = X.shape
    if b is None:
        b = np.zeros(p)
        
    if active is None:
        active = np.ones(p, dtype=bool)
    
    if intercept:
        ybar = y.mean()
        y = y - np.log(ybar / (1 - ybar))
    index = np.arange(p)
    la, dla = alpha * lambda_, (1 - alpha) * lambda_
    Xsq = X**2
    b, active, fvals = _binom_glmnet(b, X, Xsq, y, la, dla, active, index, n, 
                                     n_iters, btol, dtol, pmin, nr_ent)
    return b, active, fvals, len(fvals)


def cv_binom_glmnet(cv, X, y, alpha=0.99, lambdas=None, b=None, btol=1e-4, dtol=1e-4, 
              n_iters=1000, warm_start=True, refit=True, lmin_pct=0,
              lmin=None, pmin=1e-9, nr_ent=True, seq_rule=True, intercept=True,
              rng=None):
    '''
    Cross validated grid search for optimal elastic net penalty for a binomial
    GLM
    Parameters
    ----------
    
    cv : int
        Number of cross validation folds
    
    X : array_like
        Regression design matrix/predictor variables/independent variables

    y : array_like
        Dependent variable
    
    alpha : float
        The penalty ratio
    
    lambdas : list, array_like, int
        Penalty terms
        
    b : array_like, optional
        Regression coefficients, default None
        
    btol: float, optional
        Coefficient change tolerance, default 1e-4
    
    dtol: float, optional
        Objective function change tolerance, default 1e-4
    
    n_iters : int, optional
        Number iterations, default 1000
    
    warm_start: bool, optional
        Whether or not to use a warm start (provide the previous estimate 
        as starting values for the next lambda_), default True
    
    refit: bool, optional
        Whether or not to fit the full model at the end of cross validation for
        the optimal lambda_, default True
    
    lmin_pct: float, optional
        The percentile of the abs(X'y) to set as the minimum lambda, default 
        None
        
    pmin: float, optional
        Minimum allowed estimated probability, default 1e-9
    
    nr_ent: bool, optional
        If variable reentry is allowed, default false
    
    seq_rule : bool, optional
        Whether or not use the sequential rule to speed up crossvalidation
        default, True

    
    Returns
    -------
    betas_cv : array_like
        The coefficients
    
    fvals : array_like
        Array with three columns, respectively containing the deviance, penalty,
        and deviance+penalty
    
    lambdas_ : array_like
        Lambdas used
    n_its:
        Number of iterations for each model fit, used for diagnostics esp of
        convergence issues.
    
   
    '''
    if b is None:
        b = X.T.dot(y) / X.shape[0]
    if (lambdas is None) or (type(lambdas) in [int, float]):
        if lambdas is None:
            nl = 150
        else:
            nl = int(lambdas)
        b0 = X.T.dot(y - np.log(y.mean() / (1.0 - y.mean()))) / X.shape[0]
        if lmin_pct is None:
            lmin_pct = X.shape[1] / X.shape[0] * 100.0
        lambda_min = sp.stats.scoreatpercentile(np.abs(b0), lmin_pct) if lmin is None else lmin
        lambda_max = np.abs(b0).max() / alpha
        lambdas = np.exp(np.linspace(np.log(lambda_max), np.log(lambda_min), nl))
    if rng is None:
        rng = np.random.default_rng()
    p = X.shape[1]
    n_its = np.zeros((len(lambdas), cv))
    betas_cv = np.zeros((len(lambdas)+1, cv, p))
    betas = np.zeros((len(lambdas)+1, p))
    fvals = np.zeros((len(lambdas), cv, 3))
    
    Xf, yf, Xt, yt = crossval_mats(X, y, X.shape[0], cv, categorical=True)
    progress_bar = tqdm.tqdm(total=len(lambdas)*cv)
    beta_start = rng.normal(size=X.shape[1]) / X.shape[0]
    for i in range(cv):
        betas_cv[0, i] = beta_start
    betas[0] = beta_start
    for i, lambda_ in enumerate(lambdas):
        for k in range(cv):
            if warm_start:
                beta_start = betas_cv[i, k].copy()
            else:
                beta_start = b.copy()
            if seq_rule:
                if i==0:
                    active = np.ones(p, dtype=bool)
                else:
                    resid = yf[k] - Xf[k].dot(betas_cv[i-1, k])
                    active = np.abs(Xf[k].T.dot(resid)) > 2.0 * alpha * (lambda_ - lambdas.max())
            else:
                active = np.ones(p, dtype=bool)
            bi, _, _, ni = binom_glmnet(Xf[k], yf[k], lambda_, alpha, beta_start,
                                       active=active, btol=btol, dtol=dtol,
                                       n_iters=n_iters, pmin=pmin,nr_ent=nr_ent, 
                                       intercept=intercept)
            fi = binom_eval(yt[k], Xt[k].dot(bi), bi, alpha, lambda_)
            betas_cv[i+1, k] = bi
            fvals[i, k] = fi
            n_its[i, k] = ni
            progress_bar.update(1)
        if refit:
            if warm_start:
                beta_start = betas[i].copy()
            else:
                beta_start = rng.normal(size=X.shape[1]) / X.shape[0] * 0.0
            
            if seq_rule:
                if i==0:
                    active = np.ones(p, dtype=bool)
                else:
                    resid = y - X.dot(betas[i-1])
                    active = np.abs(X.T.dot(resid)) > 2.0 * alpha * (lambda_ - lambdas.max())
            else:
                active = np.ones(p, dtype=bool)
            bfi, _, _, _ = binom_glmnet(X, y, lambda_, alpha, beta_start, active=active,
                                     btol=btol, dtol=dtol, n_iters=n_iters,
                                     pmin=pmin,  nr_ent=nr_ent,
                                     intercept=intercept)
            betas[i+1] = bfi
    progress_bar.close()
    return betas_cv[1:], fvals, lambdas, betas[1:], n_its

