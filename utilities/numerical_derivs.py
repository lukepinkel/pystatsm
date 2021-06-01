#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 14:40:17 2020

@author: lukepinkel
"""

import numpy as np
import scipy as sp

def fo_fc_fd(f, x, eps=None, args=()):
    if eps is None:
        eps = (np.finfo(float).eps)**(1.0/3.0)
    n = len(np.asarray(x))
    g, h = np.zeros(n), np.zeros(n)
    for i in range(n):
        h[i] = eps
        g[i] = (f(x+h, *args) - f(x, *args)) / eps
        h[i] = 0
    return g

def so_fc_fd(f, x, eps=None, args=()):
    if eps is None:
        eps = (np.finfo(float).eps)**(1.0/3.0)
    n = len(np.asarray(x))
    H, hi, hj = np.zeros((n, n)), np.zeros(n), np.zeros(n)
    eps2 = eps**2
    for i in range(n):
        hi[i] = eps
        for j in range(i+1):
            hj[j] = eps
            H[i, j] = (f(x+hi+hj, *args) - f(x+hi, *args) - f(x+hj, *args) + f(x, *args)) / eps2
            H[j, i] = H[i, j]
            hj[j] = 0  
        hi[i] = 0
    return H

def so_gc_fd(g, x, eps=None, args=()):
    if eps is None:
        eps = (np.finfo(float).eps)**(1.0/3.0)
    n = len(np.asarray(x))
    H, h = np.zeros((n, n)), np.zeros(n)
    gx, gxh = np.zeros((n, n)), np.zeros((n, n))
    for i in range(n):
        h[i] = eps
        gx[i] = g(x, *args)
        gxh[i] = g(x+h, *args)
        h[i] = 0
    for i in range(n):
        for j in range(i+1):
            H[i, j] = ((gxh[i, j] - gx[i, j]) + (gxh[j, i] - gx[j, i])) / (2 * eps)
            H[j, i] = H[i, j]
    return H

def fo_fc_cd(f, x, eps=None, args=()):
    if eps is None:
        eps = (np.finfo(float).eps)**(1.0/3.0)
    n = len(np.asarray(x))
    g, h = np.zeros(n), np.zeros(n)
    for i in range(n):
        h[i] = eps
        g[i] = (f(x+h, *args) - f(x - h, *args)) / (2 * eps)
        h[i] = 0
    return g


def so_fc_cd(f, x, eps=None, args=()):
    p = len(np.asarray(x))
    if eps is None:
        eps = (np.finfo(float).eps)**(1./3.)
    H = np.zeros((p, p))
    ei = np.zeros(p)
    ej = np.zeros(p)
    for i in range(p):
        for j in range(i+1):
            ei[i], ej[j] = eps, eps
            if i==j:
                dn = -f(x+2*ei, *args)+16*f(x+ei, *args)\
                    -30*f(x, *args)+16*f(x-ei, *args)-f(x-2*ei, *args)
                nm = 12*eps**2
                H[i, j] = dn/nm  
            else:
                dn = f(x+ei+ej, *args)-f(x+ei-ej, *args)-f(x-ei+ej, *args)+f(x-ei-ej, *args)
                nm = 4*eps*eps
                H[i, j] = dn/nm  
                H[j, i] = dn/nm  
            ei[i], ej[j] = 0.0, 0.0
    return H
        
def so_gc_cd(g, x, eps=None, args=()):
    if eps is None:
        eps = (np.finfo(float).eps)**(1./3.)
    n = len(np.asarray(x))
    H, h = np.zeros((n, n)), np.zeros(n)
    gxp, gxn = np.zeros((n, n)), np.zeros((n, n))
    for i in range(n):
        h[i] = eps
        gxp[i] = g(x+h, *args)
        gxn[i] = g(x-h, *args)
        h[i] = 0
    for i in range(n):
        for j in range(i+1):
            H[i, j] = ((gxp[i, j] - gxn[i, j] + gxp[j, i] - gxn[j, i])) / (4 * eps)
            H[j, i] = H[i, j]
    return H


def fd_coefficients(points, order):
    A = np.zeros((len(points), len(points)))
    A[0] = 1
    for i in range(len(points)):
        A[i] = np.asarray(points)**(i)
    b = np.zeros(len(points))
    b[order] = sp.special.factorial(order)
    c = np.linalg.inv(A).dot(b)
    return c
        
    
def finite_diff(f, x, epsilon=None, order=1, points=None):
    if points is None:
        points = np.arange(-4, 5)
    if epsilon is None:
        epsilon = (np.finfo(float).eps)**(1./3.)
    coefs = fd_coefficients(points, order)
    df = 0.0
    for c, p in list(zip(coefs, points)):
        df+=c*f(x+epsilon*p)
    df = df / (epsilon**order)
    return df
        
    
    
    
    
"""
import pandas as pd  # analysis:ignore
import seaborn as sns # analysis:ignore
import matplotlib.pyplot as plt# analysis:ignore



import numpy as np  # analysis:ignore
import scipy as sp  # analysis:ignore
import scipy.stats  # analysis:ignore
import pandas as pd  # analysis:ignore
import seaborn as sns # analysis:ignore
import scipy.sparse as sps# analysis:ignore
from pylmm.pylmm.lmm import LME2 # analysis:ignore
import matplotlib.pyplot as plt# analysis:ignore
from pylmm.tests.test_data import generate_data # analysis:ignore
from pylmm.utilities.random_corr import vine_corr, onion_corr # analysis:ignore
from pylmm.utilities.linalg_operations import invech # analysis:ignore

from pylmm.utilities.linalg_operations import (sparse_cholesky, _check_shape_nb, # analysis:ignore
                               _check_np, sparse_woodbury_inversion, _check_shape,
                               scholesky, vech)
from pylmm.pylmm.model_matrices import (make_theta, construct_model_matrices, create_gmats,# analysis:ignore
                                  lsq_estimate, get_derivmats, get_jacmats, update_gmat,
                                  get_jacmats2)



formula = "y~x1+x2-1+(1+x3|id1)"
model_dict = {}
model_dict['gcov'] = {'id1':invech(np.array([1., 0.2, 1.]))}
model_dict['ginfo'] = {'id1':dict(n_grp=150, n_per=10)}
model_dict['mu'] = np.zeros(3)
model_dict['vcov'] = vine_corr(3, 20)
model_dict['beta'] = np.array([2, -2])
model_dict['n_obs'] = 1500
df1, formula1 = generate_data(formula, model_dict, r=0.6**0.5)


model1 = LME2(formula1, df1)
model1._fit()
x = np.array([1.0, 0.1, 1.0, 5.0])
epsilon = np.finfo(float).eps


g_approximations = []
epslist = [epsilon**(x) for x in np.linspace(0.6, 0.05, 33)]

for eps in epslist:
    row = []
    for g_approx_func in [fo_fc_fd, fo_fc_cd]:
        row.append(g_approx_func(model1.loglike, x, eps=eps))
    g_approximations.append(row)
    print(eps)
        
g_true = model1.gradient(x)
H_approximations = []
h_approx_funcs = [(so_fc_fd, model1.loglike),
                  (so_gc_fd, model1.gradient),
                  (so_fc_cd, model1.loglike),
                  (so_gc_cd, model1.gradient)]

for eps in epslist:
    row = []
    for (h_approx_func, func) in h_approx_funcs:
        row.append(h_approx_func(func, x, eps=eps))
    H_approximations.append(row)
    print(eps)

H_true = model1.hessian(x)


grad_res = np.zeros((33, 2))
grad_rel = np.zeros((33, 2))

for i, row in enumerate(g_approximations):
    for j, col in enumerate(row):
        grad_res[i, j] = np.abs(col - g_true).mean()
        grad_rel[i, j] = np.linalg.norm(col - g_true) / np.linalg.norm(g_true)
        
        
hess_res = np.zeros((33, 4))
hess_rel = np.zeros((33, 4))

for i, row in enumerate(H_approximations):
    for j, col in enumerate(row):
        hess_res[i, j] = np.abs(vech(col - H_true)).mean()
        hess_rel[i, j] = np.linalg.norm(vech(col - H_true)) / np.linalg.norm(vech(H_true))
        

grad_res = pd.DataFrame(grad_res, columns=['ffd', 'fcd'], index=epslist)
grad_rel = pd.DataFrame(grad_rel, columns=['ffd', 'fcd'], index=epslist)

hess_res = pd.DataFrame(hess_res, columns=['Forward Diff Function Evals', 
                                           'Forward Diff Gradient Evals',
                                           'Central Diff Function Evals',
                                           'Central Diff Gradient Evals'],
                        index=epslist)

hess_rel = pd.DataFrame(hess_rel, columns=hess_res.columns,
                        index=epslist)
sns.set_style("darkgrid")
fig, ax = plt.subplots()
with sns.color_palette("colorblind"):
    for col in hess_rel.columns:
        ax.plot(np.log(epslist), np.log(hess_rel[col]), label=col,
                lw=2)
    fig.legend()




"""