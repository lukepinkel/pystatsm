# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 11:17:57 2021

@author: lukepinkel
"""
import numpy as np
import scipy as sp
import scipy.linalg
import pandas as pd
from ..utilities.linalg_operations import chol_downdate



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

def chol_update(L, xtx, x, i):
    norm_xtx = np.sqrt(xtx)
    if i>0:
        r = sp.linalg.solve_triangular(L[:i, :i], x, lower=True, check_finite=False)
        rpp = np.sqrt(xtx - np.sum(r**2))
        L[i, :i] = r
        L[i, i] = rpp
    else:
        L[i, i] = norm_xtx
    return L


def lars_add_var(cj_, j, G, L, C, active, s, ind):
    Cvec = C[np.arange(C.shape[0])!=j]
    #L = _update_chol(G[ind[j], ind[j]], G[ind[j], active], L)
    L = chol_update(L, G[ind[j], ind[j]], G[ind[j], active], len(active))
    active = np.append(active, ind[j])
    s = np.append(s, np.sign(cj_))
    ind = np.delete(ind, j)
    return L, Cvec, s, active, ind

def lars_drop_var(L, betas, active, s, ind, drops, i):
    drop_ix, = np.where(drops)
    for k in drop_ix:
        L = chol_downdate(L, k)
    active_ix = active[drop_ix]
    betas[i, active_ix] = 0.0
    active = active[~drops]
    s = s[~drops]
    ind = np.append(ind, active_ix)
    return L, betas, active, s, ind
    
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


def _tmin(a, t=0):
    b = a[a>t]
    if len(b)==0:
        return np.finfo(float).max
    else:
        return np.min(b)

def get_lars_gamma(C, cj, A, aj):
    gamma1 = _tmin((cj - C) / (A - aj))
    gamma2 = _tmin((cj + C) / (A + aj))
    gamma3 = ((cj / A))
    gamma = np.min([gamma1, gamma2, gamma3])
    return gamma

def _lasso_modification(beta, active, w, gamma):
    sgn_change = -beta[active] / w
    if np.any(sgn_change>0):
        min_sgn_change = np.min([np.min(sgn_change[sgn_change>0]), gamma])
    else:
        min_sgn_change = gamma
    if min_sgn_change < gamma:
        gamma = min_sgn_change
        drops = (sgn_change == min_sgn_change)
    else:
        drops = False
    return drops, gamma


def _lars(X, y, XtX, Xty, G, C, method, n_iters, n_obs, n_var):
    betas, lambdas = np.zeros((n_var + 1, n_var)), np.zeros(n_var + 1)
    L, i, s = np.zeros((n_var, n_var)), 0, np.array([], dtype=np.float64)
    active, ind = np.array([], dtype=np.int32), np.arange(n_var)
    drops = False
    for t in range(n_iters):
        Cvec = C[ind]
        j, cj, cj_ = get_cmax(Cvec)
        lambdas[i] = cj / n_obs
        if not np.any(drops):
            L, Cvec, s, active, ind = lars_add_var(cj_, j, G, L, Cvec, active, s, ind)
        
        Gi1 = sp.linalg.cho_solve((L[:len(active), :len(active)], True), s, check_finite=False)
        A = 1. / np.sqrt(np.sum(Gi1 * s))
        w = Gi1 * A
        aj = np.dot(G[active][:, ind].T, w)
        gam = get_lars_gamma(Cvec, cj, A, aj)
        if method == "lasso":
            drops, gam = _lasso_modification(betas[i], active, w, gam)
        i += 1
        betas[i, active] = betas[i-1, active] + gam * w
        C[ind] = C[ind] - gam * aj
        if method == "lasso" and np.any(drops):
            drop_ix = np.asarray(active)[np.where(drops)]
            C[drop_ix] = Xty[drop_ix] - XtX[drop_ix].dot(betas[i])
            L, betas, active, s, ind = lars_drop_var(L, betas, active, s, ind, drops, i)
        if len(active)==(n_var):
            break
        if i>=(n_var-1):
            betas = np.append(betas, np.zeros((1, n_var)), axis=0)
            lambdas = np.append(lambdas, np.zeros((1,)))
    lambdas = lambdas[:i+1]
    betas = betas[:i+1]
    return lambdas, active, betas

def lars(X, y, method="lasso", intercept=True, normalize=False,
          standardize=False, n_iters=None):
    n_obs, n_var = X.shape
    n_iters = n_var * 10 if n_iters is None else n_iters
    X, y, XtX, Xty, G, C = handle_lars_setup(X, y, intercept, normalize, standardize)
    lambdas, active, betas = _lars(X, y, XtX, Xty, G, C, method, n_iters, n_obs, n_var)
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










