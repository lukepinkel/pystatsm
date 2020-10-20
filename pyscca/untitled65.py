#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 19:15:09 2020

@author: lukepinkel
"""
import tqdm # analysis:ignore
import numpy as np # analysis:ignore
import scipy as sp # analysis:ignore
import pandas as pd # analysis:ignore
import seaborn as sns # analysis:ignore
import matplotlib.pyplot as plt # analysis:ignore
from pystats.pyscca.scca import SCCA, _corr, vech# analysis:ignore
from pystats.utilities.random_corr import vine_corr, multi_rand # analysis:ignore

xsp, ysp = 0.02, 0.04
n, p, q, t = 3000, 300, 200, 2
m  = t * 2
S = np.eye(m)
nx = int(xsp * p)
ny = int(ysp * q)
rinds, cinds = np.indices((m, m))

rxl, cxl = np.diag(rinds, -t), np.diag(cinds, -t)
rxu, cxu = np.diag(rinds, t), np.diag(cinds, t)
rho =  np.exp(np.linspace(np.log(0.9999), np.log(0.7), t))
S[rxl, cxl] = rho
S[rxu, cxu] = rho

w = np.zeros((p, t)) 
v = np.zeros((q, t)) 

wb = np.arange(0, nx*t+1, nx)
vb = np.arange(0, ny*t+1, ny)
for i in range(t):
    w[wb[i]:wb[i+1], i] = np.concatenate((np.ones(nx//2), -np.ones(nx-nx//2)))
    v[vb[i]:vb[i+1], i] = np.concatenate((np.ones(ny//2), -np.ones(ny-ny//2)))
    
gen_vals = (np.abs(np.concatenate((w[:, [0, 1]].reshape(-1, order='F'), 
                                   v[:, [0, 1]].reshape(-1, order='F'))))>0)*1.0
Z = multi_rand(S, n)
X = Z[:, :t].dot(w.T+np.random.normal(0.0, 0.1, size=w.T.shape)) 
Y = Z[:, t:].dot(v.T+np.random.normal(0.0, 0.1, size=v.T.shape))


X[:, nx*t:] += np.random.normal(0, 1.0, size=(n, p-nx*t))         
X[:, :nx*t] += np.random.normal(0, 0.8, size=(n, nx*t))   


X[:, nx*t:] += multi_rand(vine_corr(p-nx*t), n)/np.sqrt(2.0)
X[:, :nx*t] += multi_rand(vine_corr(nx*t), n)/np.sqrt(2.0)

Y[:, ny*t:] += np.random.normal(0, 1.0, size=(n, q-ny*t))         
Y[:, :ny*t] += np.random.normal(0, 0.8, size=(n, ny*t))   

Y[:, ny*t:] += multi_rand(vine_corr(q-ny*t), n)/np.sqrt(2.0)
Y[:, :ny*t] += multi_rand(vine_corr(ny*t), n)/np.sqrt(2.0)

X = (X - X.mean(axis=0)) / X.std(axis=0)
Y = (Y - Y.mean(axis=0)) / Y.std(axis=0)

XY  = pd.DataFrame(np.hstack((X, Y)))

R = XY.corr()

sns.heatmap(R, vmin=-1, vmax=1, center=0, cmap=plt.cm.bwr)
sns.clustermap(R, vmin=-1, vmax=1, center=0, cmap=plt.cm.bwr, method="ward")
cca_kws=dict(tol=1e-7)
model = SCCA(X, Y)
Wx, Wy, rf, rt, lambdas, optinfo = model.crossval(10, 2, lambdas=50, cca_kws=cca_kws)

x_sparsity, y_sparsity = np.mean(Wx!=0, axis=2), np.mean(Wy!=0, axis=2)
xy_sparsity = np.concatenate((x_sparsity, y_sparsity), axis=2).mean(axis=2).mean(axis=1)
rt_mu = np.mean(rt, axis=1)
rt_sd = np.std(rt, axis=1)


fig, ax = plt.subplots(nrows=3, sharex=True)
for i in range(3):
    ax[i].plot(lambdas, rt_mu[:, i])
    ax[i].axvline(lambdas[np.argmax(rt_mu[:, i])])
    ax[i].axvline(lambdas[np.argmax(rt_mu[:, i]>(rt_mu[:, i].max()*0.9))])
    ax[i].scatter(lambdas, rt_mu[:, i])
    ax[i].fill_between(lambdas, (rt_mu-1.96*rt_sd)[:, i], 
                       (rt_mu+1.96*rt_sd)[:, i], alpha=0.5)
    lb, ub = ax[i].get_ylim()
    ax[i].set_ylim(lb-0.2, ub+0.2)
    ax[i].set_xlim(lambdas.min(), lambdas.max())
    ax2 = ax[i].twinx()
    ax2.plot(lambdas, xy_sparsity)
mng = plt.get_current_fig_manager()
mng.window.showMaximized()
plt.subplots_adjust(left=0.3, right=0.7, bottom=0.025, top=0.975)


fig, ax = plt.subplots(ncols=2, nrows=2, sharex=True)
for i in range(2):
    a = x_sparsity[:, :, i]
    b = y_sparsity[:, :, i]
    ax[i, 0].plot(lambdas, a.mean(axis=1))
    ax[i, 0].fill_between(lambdas, a.min(axis=1) , a.max(axis=1),  alpha=0.5)
    
    ax[i, 1].plot(lambdas, b.mean(axis=1))
    ax[i, 1].fill_between(lambdas, b.min(axis=1), b.max(axis=1), alpha=0.5)
    
    ax[i, 0].set_xlim(lambdas.min(), lambdas.max())
    ax[i, 1].set_xlim(lambdas.min(), lambdas.max())
mng = plt.get_current_fig_manager()
mng.window.showMaximized()



lam_ = lambdas[np.argmax(rt_mu[:, i]>(rt_mu[:, i].max()*0.9))]
Wxf, Wyf, rf, optinfo = model._fit(lam_, lam_, n_comps=2, 
                                   cca_kws=dict(tol=1e-9, n_iters=5000))
Vx, Vy = model.X.dot(Wxf), model.Y.dot(Wyf)
rho = vech(_corr(Vx, Vy))
cca_kws = cca_kws=dict(cca_kws=dict(tol=1e-4,n_iters=60))
rho_dist = model.permutation_test(lam_, lam_, 1500, 2, cca_kws)

rho_dist_df = pd.DataFrame(rho_dist)

pvals = 1.0 - (np.abs(rho) > np.abs(rho_dist)).sum() / len(rho_dist)

ax = sns.PairGrid(rho_dist)
ax.map_lower(sns.regplot, scatter_kws=dict(s=0.5))
ax.map_diag(sns.distplot)
ax.map_upper(sns.kdeplot, shade_lowest=False)


Wx_boot, Wy_boot, rho_boot = model.bootstrap(lam_, lam_, 1500, 2)
dg_inds = list(zip(*np.triu_indices(4)[::-1]))
dg_inds = [i for i, (a, b) in enumerate(dg_inds) if a==b]
ax = sns.PairGrid(rho_boot)
ax.map_lower(sns.regplot, scatter_kws=dict(s=0.5))
ax.map_diag(sns.distplot)
ax.map_upper(sns.kdeplot, shade_lowest=False)

fig, ax = plt.subplots(nrows=4, sharex=True)
for i in range(4):
    sns.distplot(rho_boot.iloc[:, dg_inds[i]], ax=ax[i])
mng = plt.get_current_fig_manager()
mng.window.showMaximized()
 
    

Wx_boot_df = pd.DataFrame(Wx_boot.reshape(Wx_boot.shape[0], -1, order='F'))
Wy_boot_df = pd.DataFrame(Wy_boot.reshape(Wy_boot.shape[0], -1, order='F'))

wx_summary = Wx_boot_df.agg(['mean', 'std', 'min', 'max'])
wx_summary.columns = [f"wx{i}" for i in range(1, wx_summary.shape[1]+1)]
wy_summary = Wy_boot_df.agg(['mean', 'std', 'min', 'max'])
wy_summary.columns = [f"wy{i}" for i in range(1, wy_summary.shape[1]+1)]

w_summ = pd.concat([wx_summary, wy_summary], axis=1).T
w_summ['t'] = w_summ['mean'] / np.maximum(w_summ['std'], 1e-16)
w_summ['p_unadjusted'] = sp.stats.t(n).sf(np.abs(w_summ['t'])) * 2.0
pvals = w_summ['p_unadjusted']
w_summ['p_bonferroni'] = pvals * len(w_summ)
w_summ['p_sidak'] = 1.0 - np.power((1.0 - pvals), len(w_summ))
w_summ['p_holm'] = 1.0 - np.power((1.0 - pvals), np.arange( len(w_summ), 0, -1))
w_summ['genmod'] = gen_vals

Wx_cvmc, Wy_cvmc, res_cvmc = model.crossval_mc(lam_, lam_, 15, 12, 2)


Wx_cvmc_df = pd.DataFrame(Wx_cvmc.reshape(np.product(Wx_cvmc.shape[:2]), -1, order='F')).T
Wx_cvmc_df.index = [f'wx{i}' for i in range(Wx_cvmc_df.shape[0])]
Wy_cvmc_df = pd.DataFrame(Wy_cvmc.reshape(np.product(Wy_cvmc.shape[:2]), -1, order='F')).T
Wy_cvmc_df.index = [f'wy{i}' for i in range(Wy_cvmc_df.shape[0])]

W_cvmc_df = pd.concat([Wx_cvmc_df, Wy_cvmc_df])



W_cvmc_df['genvals'] = gen_vals

VC = (np.abs(W_cvmc_df.iloc[:, :-1])>0) * 1.0

VC_mean = pd.concat([VC.mean(axis=1), W_cvmc_df.iloc[:, -1]], axis=1)

VC_mean = VC_mean.rename(columns={0:'ProbSelect'})

VC_true_pos = ((VC==1.0) & (gen_vals[:, None]==1.0)).sum(axis=0)
VC_false_pos = ((VC==1.0) & (gen_vals[:, None]!=1.0)).sum(axis=0)
VC_false_neg = ((VC!=1.0) & (gen_vals[:, None]==1.0)).sum(axis=0)
VC_true_neg = ((VC!=1.0) & (gen_vals[:, None]!=1.0)).sum(axis=0)

VC_bpr = pd.concat([VC_true_pos, VC_false_pos, VC_false_neg, VC_true_neg], axis=1)
VC_bpr.columns = ['TP', 'FP', 'FN', 'TN']

Vxf, Vyf = X.dot(Wxf), Y.dot(Wyf)
Bx = np.linalg.inv(Vxf.T.dot(Vxf)).dot(Vxf.T.dot(X))
By = np.linalg.inv(Vyf.T.dot(Vyf)).dot(Vyf.T.dot(Y))
Xhat, Yhat = Vxf.dot(Bx), Vyf.dot(By)
Xresid, Yresid = X - Xhat, Y - Yhat

rsquared_x = (1.0 - np.sum(Xresid**2, axis=0) / (X**2).sum(axis=0))
rsquared_y = (1.0 - np.sum(Yresid**2, axis=0) / (Y**2).sum(axis=0))

np.mean(rsquared_x)
np.mean(rsquared_y)







