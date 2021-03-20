# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 22:41:47 2020

@author: lukepinkel
"""

import numpy as np
import scipy as sp
import scipy.stats
import pandas as pd
import seaborn as sns
from pystats.utilities.random_corr import vine_corr, multi_rand
from pystats.pyscca.scca import SCCA

n_obs = 2000
nx = 200
ny = 150
nv = 5
Sigma = np.eye(nv*2)

ix1, ix2 = np.arange(nv, nv*2), np.arange(nv)

Sigma[ix1, ix2] = np.linspace(0.99, 0.90, nv)
Sigma[ix2, ix1] = np.linspace(0.99, 0.90, nv)

Lx = np.zeros((nx, nv))
Ly = np.zeros((ny, nv))

inds = np.array_split(np.arange(nx), nv), np.array_split(np.arange(ny), nv)
inds = list(zip(*inds))
for i, (ix, iy) in enumerate(inds):
    Lx[ix, i] = np.random.choice([-1.0, 1.0], len(ix))
    Ly[iy, i] = np.random.choice([-1.0, 1.0], len(iy))
    
U = multi_rand(Sigma, n_obs)
X = U[:, :nv].dot(Lx.T) + multi_rand(np.eye(nx)/5.0, n_obs)
Y = U[:, nv:].dot(Ly.T) + multi_rand(np.eye(ny)/5.0, n_obs) 

model = SCCA(X[:1000], Y[:1000])

Wx, Wy, rf, rt, lambdas, optinfo = model.crossval(10, 3, lbounds=(0.01, 0.99))
rho_means = rt.mean(axis=1)
lx, ly = lambdas[8], lambdas[8]

res = model.permutation_test(lx, ly, n_perms=2000, n_comps=3)

wx, wy, r, opt = model._fit(lx, ly, 3)

p_values = (np.abs(res)>np.abs(r)).mean()

Ux, Uy = X[1000:].dot(wx), Y[1000:].dot(wy)

Ux -= Ux.mean()
Ux /= Ux.std()
Uy -= Uy.mean()
Uy /= Uy.std()

Rxy = Ux.T.dot(Uy) / Ux.shape[0]






