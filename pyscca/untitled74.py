#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 01:12:18 2020

@author: lukepinkel
"""


import tqdm # analysis:ignore
import numpy as np # analysis:ignore
import scipy as sp # analysis:ignore
import pandas as pd # analysis:ignore
import seaborn as sns # analysis:ignore
import matplotlib.pyplot as plt # analysis:ignore
from pystats.pyscca.scca3 import SCCA, _corr, vech, fast_scca_vec# analysis:ignore
from pystats.utilities.random_corr import vine_corr, multi_rand # analysis:ignore

xsp, ysp = 0.02, 0.04
n, p, q, t = 3000, 300, 200, 4
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
    
gen_vals = (np.abs(np.concatenate((w[:, [0, 1, 2, 3]].reshape(-1, order='F'), 
                                   v[:, [0, 1, 2, 3]].reshape(-1, order='F'))))>0)*1.0
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


model = SCCA(X, Y)
Wx, Wy, rf, rt, lambdas, optinfo = model.crossval(10, 4, lambdas=50)


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

lam_ = lambdas[4]
model = SCCA(X, Y)
Wxf, Wyf, rf, optinfo = model._fit_symdef_d(lam_, lam_, n_comps=4,  cca_kws=dict(tol=1e-9, n_iters=5000))
Bxf, Byf = model.orthogonalize_components(Wxf, Wyf)
Vx, Vy = model.X.dot(Wxf), model.Y.dot(Wyf)
Ux, Uy = model.X.dot(Bxf), model.Y.dot(Byf)

R1 = pd.DataFrame(np.hstack((Vx, Vy))).corr()
R2 = pd.DataFrame(np.hstack((Ux, Uy))).corr()


lambda_x, lambda_y = lam_, lam_

n_comps=4
X = model.X.copy()
Y = model.Y.copy()

wx = model.U[:, 0]
wx = wx / np.sqrt(np.sum(wx**2))
wy = model.V.T[:, 0]
wy = wy / np.sqrt(np.sum(wy**2))
cx, cy = lambda_x * np.sqrt(model.p), lambda_y * np.sqrt(model.q)
Xt, Yt = X.copy(), Y.copy()

Wx, Wy = np.zeros((n_comps, model.p)), np.zeros((n_comps, model.q))
Bx, By = np.zeros((n_comps, model.p)), np.zeros((n_comps, model.q))

Vx, Vy = np.zeros((model.n, n_comps)), np.zeros((model.n, n_comps))
n_iters=500
tol=1e-6
for i in range(n_comps):
    St = Xt.T.dot(Yt)
    wxi, wyi, dt, nc = fast_scca_vec(St, wx, wy, cx, cy, n_iters=n_iters, tol=tol)
    vxi, vyi = Xt.dot(wxi), Yt.dot(wyi)
    bxi = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(Xt)).dot(wxi)
    byi = np.linalg.inv(Y.T.dot(Y)).dot(Y.T.dot(Yt)).dot(wyi)
    Wx[i], Wy[i] = wxi, wyi
    Bx[i], By[i] = bxi, byi
    Vx[:, i], Vy[:, i] = vxi, vyi
    uX, uY = vxi.T.dot(Xt), vyi.T.dot(Yt)
    Xt = Xt - (np.outer(vxi, uX) / np.sum(vxi**2))
    Yt = Yt - (np.outer(vyi, uY) / np.sum(vyi**2))    
    





