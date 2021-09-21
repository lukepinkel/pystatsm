#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 03:42:21 2020

@author: lukepinkel
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def crossval_mats(X, y, n, cv, categorical=False):
    kfix = kfold_indices(n, cv, y, categorical)
    Xf, yf, Xt, yt = [], [], [], []
    for f_ix, v_ix in kfix:
        Xf.append(X[f_ix])
        yf.append(y[f_ix])
        Xt.append(X[v_ix])
        yt.append(y[v_ix])
    return Xf, yf, Xt, yt
    
def kfold_indices(n, k, y, categorical=False):
    if categorical:
        _,  idx =np.unique(y, return_inverse=True)
        t = np.arange(y.shape[0])
        splits = list(zip(*[np.array_split(t[idx==i], k) for i in np.unique(idx)]))
        splits = [np.concatenate(x) for x in splits]
    splits = np.array_split(np.arange(n), k)
    inds = []
    for i in range(k):
        fit_ind = np.concatenate([splits[j] for j in range(k) if j!=i])
        inds.append([fit_ind, splits[i]])
    return inds


def process_cv(fval, lambdas):
    df = pd.DataFrame(fval)
    summary = pd.concat([df.mean(axis=1), df.std(axis=1) / np.sqrt(df.shape[1])], axis=1)
    summary.columns = ['mean', 'std']
    lambda_min = lambdas[summary.idxmin()['mean']]
    return summary, lambda_min
                   
def plot_elnet_cv(f_path, lambdas, bfits=None):
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
    ax[0].errorbar(x=np.log(lambdas), y=mse.mean(axis=1), yerr=mse.std(axis=1)/np.sqrt(mse.shape[1]), 
                elinewidth=1.0, fmt='none', ecolor='grey', capsize=2.0)
    ax[0].axvline(np.log(lambda_min_mse))
    xpos = np.log(lambda_min_mse)
    _, ypos = ax[0].get_ylim()
    xlb, xub = ax[0].get_xlim()
    xpos = xpos + (xub - xpos) * 0.01
    xy = xpos, ypos*0.95
    ax[0].annotate(f"{xpos:.2f}", xy=xy, xytext=xy, horizontalalignment='left')
    
    
    ax[1].scatter(x=np.log(lambdas), y=pen.mean(axis=1), s=10, color='red', zorder=0)
    ax[1].errorbar(x=np.log(lambdas), y=pen.mean(axis=1), yerr=pen.std(axis=1)/np.sqrt(pen.shape[1]), 
                elinewidth=1.0, fmt='none', ecolor='grey', capsize=2.0)
    
    ax[2].scatter(x=np.log(lambdas), y=pll.mean(axis=1), s=10, color='red', zorder=0)
    ax[2].errorbar(x=np.log(lambdas), y=pll.mean(axis=1), yerr=pll.std(axis=1)/np.sqrt(pll.shape[1]), 
                elinewidth=1.0, fmt='none', ecolor='grey', capsize=2.0)
    plt.subplots_adjust(left=0.3, right=0.7, top=0.96, bottom=0.05)
    ax[0].set_ylabel("Deviance", rotation=0, labelpad=50)
    ax[1].set_ylabel("Penalty", rotation=0, labelpad=50)
    ax[2].set_ylabel("Penalized LL", rotation=0, labelpad=50)
    if bfits is not None:
        axt = ax[0].twinx()
        axt.plot(np.log(lambdas), (bfits!=0).sum(axis=1))
    return fig, ax
    