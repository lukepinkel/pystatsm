#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 23:45:29 2020

@author: lukepinkel
"""
import tqdm  # analysis:ignore
import numba # analysis:ignore
import numpy as np # analysis:ignore
import scipy as sp # analysis:ignore
import scipy.stats # analysis:ignore
import pandas as pd # analysis:ignore
import seaborn as sns # analysis:ignore
import matplotlib.pyplot as plt # analysis:ignore
from pystats.utilities.random_corr import vine_corr, multi_rand # analysis:ignore

@numba.jit(nopython=True)
def sft(x, t):
    y = np.maximum(np.abs(x) - t, 0) * np.sign(x)
    return y

@numba.jit(nopython=True)
def elnet_penalty(b, alpha, lambda_):
    p = (np.sum(b**2) * (1.0 - alpha) / 2.0 + alpha * np.sum(np.abs(b))) * lambda_
    return p


@numba.jit(nopython=True)
def elnet_loglike(X, y, b, alpha, lambda_):
    n = X.shape[0]
    r = y - X.dot(b)
    msr =  np.sum(r**2) / (2.0 * n)
    pen = elnet_penalty(b, alpha, lambda_)
    fval = msr + pen
    return msr, pen, fval
    
    
    
@numba.njit
def np_apply_along_axis(func1d, axis, arr):
    assert arr.ndim == 2
    assert axis in [0, 1]
    if axis == 0:
        result = np.empty(arr.shape[1])
        for i in range(len(result)):
            result[i] = func1d(arr[:, i])
    else:
        result = np.empty(arr.shape[0])
        for i in range(len(result)):
            result[i] = func1d(arr[i, :])
    return result

@numba.njit
def numba_mean(array, axis):
    return np_apply_along_axis(np.mean, axis, array)
  
@numba.njit
def numba_sum(array, axis):
    return np_apply_along_axis(np.sum, axis, array)

@numba.jit(nopython=True)
def eln_cd(X, y, alpha, lambda_, b, active, n_iters=1000, tol=1e-5, btol=1e-9):
    r = y - X.dot(b)
    n = y.shape[0]
    n2 = n * 2.0
    la, dn = lambda_ * alpha, (1.0 - alpha) * lambda_ + 1.0
    index = np.arange(len(b))
    msr =  np.sum(r**2) / n2
    pen = elnet_penalty(b, alpha, lambda_)
    f_old = msr + pen
    fvals = np.zeros((n_iters+1, 3))
    beta_hat = b
    for i in range(n_iters):
        active_vars = index[active]
        fvals[i] = msr, pen, f_old
        for j in active_vars:
            xb =  X[:, j] * b[j]
            yp = r + xb
            b[j] = sft(np.sum(X[:, j] * yp)/n, la) / (dn)
            if abs(b[j])>btol:
                r = yp - X[:, j] * b[j]
            else:
                r = yp
                active[j] = False
        msr = np.sum(r**2)/ n2
        pen = elnet_penalty(b, alpha, lambda_)
        f_new = msr + pen
        if (f_old - f_new)<tol:
            fvals[i+1] = msr, pen, f_new
            break
        else:
            f_old = f_new
            beta_hat = b
    return beta_hat, fvals, active, i
            

def kfold_indices(n, k):
    splits = np.array_split(np.arange(n), k)
    inds = []
    for i in range(k):
        fit_ind = np.concatenate([splits[j] for j in range(k) if j!=i])
        inds.append([fit_ind, splits[i]])
    return inds
            
def elnet(X, y, lambda_, alpha=0.99, b=None, active=None, n_iters=1000, tol=1e-9,
          intercept=True):
    if b is None:
        b = X.T.dot(y) / X.shape[0]
    if active is None:
        active = np.ones(X.shape[1], dtype=bool)
    if intercept:
        y = y - y.mean()
    beta, fvals, active, nits = eln_cd(X, y, alpha, lambda_, b, active, n_iters, tol)
    fvals = fvals[:(nits+2)]
    return beta, fvals, active, nits

def elnet_grad(b, X, y, lambda_, alpha):
    n = y.shape[0]
    r = y - X.dot(b)
    g = -1.0 / n * X.T.dot(r) + lambda_ * (1 - alpha) * b + lambda_*alpha
    return g
  
def process_cv(fval, lambdas):
    df = pd.DataFrame(fval)
    summary = pd.concat([df.mean(axis=1), df.std(axis=1)], axis=1)
    summary.columns = ['mean', 'std']
    lambda_min = lambdas[summary.idxmin()['mean']]
    return summary, lambda_min
       
def crossval_mats(X, y, n, cv):
    kfix = kfold_indices(n, cv)
    Xf, yf, Xt, yt = [], [], [], []
    for f_ix, v_ix in kfix:
        Xf.append(X[f_ix])
        yf.append(y[f_ix])
        Xt.append(X[v_ix])
        yt.append(y[v_ix])
    return Xf, yf, Xt, yt
    
def cv_glmnet(cv, X, y, alpha=0.99, lambdas=None, b=None, tol=1e-4, n_iters=1000, 
              refit=True, lmin_pct=0, lmax_pct=100, seq_rule=True, 
              warm_start=True, intercept=True):
    if b is None:
        b = X.T.dot(y) / X.shape[0]
    if (lambdas is None) or (type(lambdas) in [int, float]):
        if lambdas is None:
            nl = 150
        else:
            nl = int(lambdas)
        b0 = X.T.dot(y - y.mean()) / X.shape[0]
        lambda_min = sp.stats.scoreatpercentile(np.abs(b0), lmin_pct)
        lambda_max = sp.stats.scoreatpercentile(np.abs(b0), lmax_pct) / alpha
        lambdas = np.exp(np.linspace(np.log(lambda_max), np.log(lambda_min), nl))
    p = X.shape[1]
    betas = np.zeros((len(lambdas)+1, p))
    betas_cv = np.zeros((len(lambdas)+1, cv, p))
    fvals = np.zeros((len(lambdas), cv, 3))
    n_its = np.zeros((len(lambdas), cv))
    Xf, yf, Xt, yt = crossval_mats(X, y, X.shape[0], cv)
    progress_bar = tqdm.tqdm(total=len(lambdas)*cv)
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
            bi, _, _, n_i = elnet(Xf[k], yf[k], lambda_, alpha, beta_start.copy(), 
                             tol=tol, n_iters=n_iters, active=active,
                             intercept=intercept)
            fi = elnet_loglike(Xt[k], yt[k], bi, alpha, lambda_)
            betas_cv[i+1, k] = bi
            fvals[i, k] = fi
            n_its[i, k] = n_i
            b = bi.copy()
            progress_bar.update(1)
        if refit:
            if warm_start:
                beta_start = betas[i].copy()
            else:
                beta_start = np.random.normal(size=X.shape[1]) / X.shape[0] * 0.0
            if seq_rule:
                if i==0:
                    active = np.ones(p, dtype=bool)
                else:
                    resid = y - X.dot(betas[i-1])
                    active = np.abs(X.T.dot(resid)) > 2.0 * alpha * (lambda_ - lambdas.max())
            else:
                active = np.ones(p, dtype=bool)
            betas[i+1], _, _, _ = elnet(X, y, lambda_, alpha, beta_start.copy(),
                                     tol=tol, active=active, n_iters=n_iters,
                                     intercept=intercept)
            
    progress_bar.close()

    return betas_cv[1:], fvals, lambdas, betas[1:], n_its
                   
def plot_elnet_cv(f_path, lambdas):
    mse = pd.DataFrame(f_path[:, :, 0])
    pen = pd.DataFrame(f_path[:, :, 1])
    pll = pd.DataFrame(f_path[:, :, 2])
    mse_sumstats, lambda_min_mse = process_cv(f_path[:, :, 0], lambdas)
    pen_sumstats, lambda_min_pen = process_cv(f_path[:, :, 1], lambdas)
    pll_sumstats, lambda_min_pll = process_cv(f_path[:, :, 2], lambdas)
        
    fig, ax = plt.subplots(nrows=3)
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    
    ax[0].scatter(x=np.log(lambdas), y=mse.mean(axis=1), s=10, color='red', zorder=0)
    ax[0].errorbar(x=np.log(lambdas), y=mse.mean(axis=1), yerr=mse.std(axis=1), 
                elinewidth=1.0, fmt='none', ecolor='grey', capsize=2.0)
    ax[0].axvline(np.log(lambda_min_mse))
    xpos = np.log(lambda_min_mse)
    _, ypos = ax[0].get_ylim()
    xlb, xub = ax[0].get_xlim()
    xpos = xpos + (xub - xpos) * 0.01
    xy = xpos, ypos*0.95
    ax[0].annotate(f"{xpos:.2f}", xy=xy, xytext=xy, horizontalalignment='left')
    
    
    ax[1].scatter(x=np.log(lambdas), y=pen.mean(axis=1), s=10, color='red', zorder=0)
    ax[1].errorbar(x=np.log(lambdas), y=pen.mean(axis=1), yerr=pen.std(axis=1), 
                elinewidth=1.0, fmt='none', ecolor='grey', capsize=2.0)
    
    ax[2].scatter(x=np.log(lambdas), y=pll.mean(axis=1), s=10, color='red', zorder=0)
    ax[2].errorbar(x=np.log(lambdas), y=pll.mean(axis=1), yerr=pll.std(axis=1), 
                elinewidth=1.0, fmt='none', ecolor='grey', capsize=2.0)
    plt.subplots_adjust(left=0.3, right=0.7, top=0.96, bottom=0.05)
    ax[0].set_ylabel("MSE", rotation=0, labelpad=50)
    ax[1].set_ylabel("Penalty", rotation=0, labelpad=50)
    ax[2].set_ylabel("Penalized LL", rotation=0, labelpad=50)
    return fig, ax
    
'''

n, p, q = 1000, 500, 50
#S = vine_corr(p, eta=1.0)
S = np.eye(p)
X = multi_rand(S, n)
X/= np.sqrt(np.sum(X**2, axis=0))
X*=np.sqrt(X.shape[0])
beta = np.zeros(p)
bvals = np.random.choice([-1, -0.5, 0.5, 1.0], q, replace=True)
beta[np.random.choice(p, q, replace=False)] = bvals 
lpred = X.dot(beta)
rsq = 0.5

y = sp.stats.norm(lpred, np.sqrt((1-rsq)/rsq * lpred.var())).rvs()


alpha = 0.99
beta_path, f_path, lambdas = cv_glmnet(7, X, y, alpha=alpha)

mse_sumstats, lambda_min_mse = process_cv(f_path[:, :, 0], lambdas)
plot_elnet_cv(f_path, lambdas)

lambda_ = lambda_min_mse
beta_hat, fvals, active = elnet(X, y, lambda_, alpha,  n_iters=5000, tol=1e-12)

comp = pd.DataFrame(np.vstack((beta, beta_hat)).T)

elnet_grad(beta_hat, X, y, lambda_, alpha)[active]
lambda_ * 2 * alpha

'''