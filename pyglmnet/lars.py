# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 11:17:57 2021

@author: lukepinkel
"""
import numpy as np
import scipy as sp
import pandas as pd

def get_cmax(C):
    if len(C)>0:
        j = np.argmax(np.abs(C))
        cj_ = C[j]
        cj = np.abs(cj_)
    else:
        j = None
        cj_ = 0.
        cj = 0.
    return j, cj, cj_

def _update_chol(xtx, xold, L=None):
    norm_xtx = np.sqrt(xtx)
    if L is None:
        L = np.atleast_2d(norm_xtx)
    else:
        r = sp.linalg.solve_triangular(L, xold, lower=True, check_finite=False)
        rpp = np.sqrt(xtx - np.sum(r**2))
        L = np.block([[L, np.zeros((L.shape[0], 1))],
                      [r, np.atleast_1d(rpp)]])
    return L

def lars_add_var(cj_, j, G, L, C, active, s, ind):
    C = C[np.arange(C.shape[0])!=j]
    L = _update_chol(G[ind[j], ind[j]], G[ind[j], active], L)
    active.append(ind[j])
    s = np.append(s, np.sign(cj_))
    ind = np.delete(ind, j)
    return L, C, s, active, ind
        

def handle_lars_setup(X, y, intercept, normalize, standardize):
    n_obs, n_var = X.shape
    if intercept:
        X = X - np.mean(X, axis=0)
        y = y - np.mean(y, axis=0)
    if standardize:
        X = X / np.std(X, axis=0)
    if normalize:
        X = X / np.sqrt(n_obs)
    XtX, Xty = np.dot(X.T, X), np.dot(X.T, y)
    G, C = XtX.copy(), Xty.copy()
    return X, y, XtX, Xty, G, C

def check_lambda(i, lambdas, betas, lambda_):
    if i>0:
        hs = (lambdas[i-1] - lambda_) / (lambdas[i-1] - lambdas[i])
        betas[i] = betas[i-1] + hs * (betas[i] - betas[i-1])
    lambdas[i] = lambda_
    return lambdas, betas

def get_lars_gamma(C, cj, A, aj):
    gamma1 = np.min(np.maximum((cj - C) / (A - aj), 0), initial=1e16)
    gamma2 = np.min(np.maximum((cj + C) / (A + aj), 0), initial=1e16)
    gamma3 = np.min(np.maximum(cj / A, 0), initial=1e16)
    
    gamma = np.min([gamma1, gamma2, gamma3])
    return gamma

def _lars(X, y, method="lasso", lambda_=1.0, intercept=True, normalize=False,
          standardize=False):
    n_obs, n_var = X.shape
    X, y, XtX, Xty, G, C = handle_lars_setup(X, y, intercept, normalize, standardize)
    betas, lambdas = np.zeros((n_var + 1, n_var)), np.zeros(n_var + 1)
    i = 0
    active, ind, s = list(), np.arange(n_var), np.array([])
    L = None
    for t in range(n_var):
        j, cj, cj_ = get_cmax(C)
        lambdas[i] = cj / n_obs
        if lambdas[i] <= lambda_:
            lambdas, betas = check_lambda(i, lambdas, betas, lambda_)
            break
        L, C, s, active, ind = lars_add_var(cj_, j, G, L, C, active, s, ind)
        Gi1 = sp.linalg.cho_solve((L, True), s, check_finite=False)
        A = 1. / np.sqrt(np.sum(Gi1 * s))
        w = Gi1 * A
        aj = np.dot(G[active][:, ind].T, w)
        gam = get_lars_gamma(C, cj, A, aj)
        i += 1
        betas[i, active] = betas[i-1, active] + gam * w
        C = C - gam * aj
    lambdas = lambdas[:i+1]
    betas = betas[:i+1]
    return lambdas, active, betas


def _lars_sumstats(X, y, lambdas, active, betas, s2=None, s2_method="yvar"):
    if s2 is None:
        if s2_method=="ols":
            b = np.linalg.inv(np.dot(X.T, X)).dot(np.dot(X.T, y))
            r = y - X.dot(b)
            s2 = np.sum(r**2) / r.shape[0]
        elif s2_method=="yvar":
            s2 = np.var(y)
        else:
            r = y - X.dot(betas[-1])
            s2 = np.sum(r**2) / np.sum(betas[-1]!=0)
    
    resids = y.reshape(-1, 1) - X.dot(betas.T)
    ssr = np.sum(resids**2, axis=0)
    degfree = np.sum(betas!=0, axis=1)
    n_obs = y.shape[0]
    AIC = ssr / s2 + 2 * degfree
    BIC = ssr / s2 + np.log(n_obs) * degfree
    Rsq = 1.0 - ssr / (s2 * n_obs)
    res = pd.DataFrame(np.vstack((ssr, degfree, AIC, BIC, Rsq)).T,
                       columns=["SSR", "df", "AIC", "BIC", "Rsq"])
    return res










