#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 21:47:15 2020

@author: lukepinkel
"""

import tqdm # analysis:ignore
import numpy as np # analysis:ignore
import scipy as sp # analysis:ignore
import pandas as pd # analysis:ignore
import seaborn as sns # analysis:ignore
import matplotlib.pyplot as plt # analysis:ignore
from pystats.pyscca.scca import SCCA, _corr, vech,fast_scca2, fast_scca_vec # analysis:ignore
from pystats.utilities.random_corr import vine_corr, multi_rand # analysis:ignore

xsp, ysp = 0.02, 0.04
n, p, q, t = 3000, 400, 200, 4
m  = t * 2
S = np.eye(m)
nx = int(xsp * p)
ny = int(ysp * q)
rinds, cinds = np.indices((m, m))

rxl, cxl = np.diag(rinds, -t), np.diag(cinds, -t)
rxu, cxu = np.diag(rinds, t), np.diag(cinds, t)
rho =  np.exp(np.linspace(np.log(0.9999), np.log(0.90), t))[::-1]
S[rxl, cxl] = rho
S[rxu, cxu] = rho

w = np.zeros((p, t)) 
v = np.zeros((q, t)) 

wb = np.arange(0, nx*t+1, nx)
vb = np.arange(0, ny*t+1, ny)
for i in range(t):
    w[wb[i]:wb[i+1], i] = np.concatenate((np.ones(nx//2), -np.ones(nx-nx//2)))
    v[vb[i]:vb[i+1], i] = np.concatenate((np.ones(ny//2), -np.ones(ny-ny//2)))

Z = multi_rand(S, n)
X = Z[:, :t].dot(w.T+np.random.normal(0.0, 0.2, size=w.T.shape)) 
Y = Z[:, t:].dot(v.T+np.random.normal(0.0, 0.2, size=v.T.shape))


X[:, nx*t:] += np.random.normal(0, 1.0, size=(n, p-nx*t))         
X[:, :nx*t] += np.random.normal(0, 1.5, size=(n, nx*t))   


X[:, nx*t:] += multi_rand(vine_corr(p-nx*t), n)/np.sqrt(1.0)
X[:, :nx*t] += multi_rand(vine_corr(nx*t), n)/np.sqrt(1.5)

Y[:, ny*t:] += np.random.normal(0, 1.0, size=(n, q-ny*t))         
Y[:, :ny*t] += np.random.normal(0, 1.5, size=(n, ny*t))   

Y[:, ny*t:] += multi_rand(vine_corr(q-ny*t), n)/np.sqrt(1.0)
Y[:, :ny*t] += multi_rand(vine_corr(ny*t), n)/np.sqrt(1.5)

X = (X - X.mean(axis=0)) / X.std(axis=0)
Y = (Y - Y.mean(axis=0)) / Y.std(axis=0)

XY  = pd.DataFrame(np.hstack((X, Y)))


cx, cy = 0.15 * np.sqrt(p), 0.15 * np.sqrt(q)
wx, wy = np.random.normal(size=p), np.random.normal(size=q)
wx, wy  = wx/np.sum(wx**2)**0.5, wy / np.sum(wy**2)**0.5
Xt, Yt = X.copy(), Y.copy()
St = Xt.T.dot(Yt)

fast_scca_vec(St, wx, wy, cx, cy)





