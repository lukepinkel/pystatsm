# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 09:41:35 2021

@author: lukepinkel
"""

import numpy as np
from pystatsm.pyfa.factor_analysis import FactorAnalysis
from pystatsm.utilities.data_utils import center, scale_diag
from pystatsm.utilities.numerical_derivs import fo_fc_cd


def triang_inds(n, k=1):
    r, c = np.indices((n, n))
    rows, cols = [], []
    for i in range(1, k+1):
        rows.append(np.diag(r, -1))
        rows.append(np.diag(r,  1))
        cols.append(np.diag(c, -1))
        cols.append(np.diag(c,  1))
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    return rows, cols


def simulate_factor_model(n_obs=1000, n_vars=6, n_facs=2, psi=None, L=None, Phi=None, 
                          rng=None):
    rng = np.random.default_rng() if rng is None else rng
    psi = rng.uniform(low=0.25, high=0.75, size=n_vars) if psi is None else psi
    if L is None:
        L = np.zeros((n_vars, n_facs))
        inds = np.array_split(np.arange(n_vars), n_facs)
        for i, ix in enumerate(inds):
            L[ix, i] = np.linspace(1.0, 0.5, len(ix))#1.0 
    Phi = np.eye(n_facs) if Phi is None else Phi
    Psi = np.diag(psi)
    S = L.dot(Phi).dot(L.T) + Psi
    X = rng.multivariate_normal(mean=np.zeros(n_vars), cov=S, size=(n_obs,))
    X = center(X)
    return X, S



def test_efa():
    rng = np.random.default_rng(123)
    n_obs = 10000
    n_vars = 15
    n_facs = 3
    Phi = np.eye(n_facs)
    Phi[triang_inds(n_facs)] = 0.3
    Phi = scale_diag(Phi, np.array([3.0, 2.0, 1.0]))
    
    X, S = simulate_factor_model(n_obs=n_obs, n_vars=n_vars, n_facs=n_facs, Phi=Phi,
                                 rng=rng)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    
    model = FactorAnalysis(X, n_factors=n_facs, rotation_method="quartimax")
    model.fit()
    
    x = model.theta + 0.1
    g1 = fo_fc_cd(model.loglike, x)
    g2 = model.gradient(x)
    grad_close = np.allclose(g1, g2)
    H1 = model.hessian_approx(x)
    H2 = model.hessian(x)
    hess_close = np.allclose(H1, H2)
    
    assert(grad_close)
    assert(hess_close)
    
    Lambda = np.array([[ 0.9679300157,  0.0020941483, -0.0037152832],
                       [ 0.9808302280, -0.0009770430, -0.0012631786],
                       [ 0.9656468771,  0.0018262625, -0.0012049879],
                       [ 0.9546006929, -0.0046688628,  0.0069179692],
                       [ 0.9310045011,  0.0018245676, -0.0004898625],
                       [-0.0090650550,  0.9322148607, -0.0013228623],
                       [-0.0034623862,  0.9003171814, -0.0006441778],
                       [ 0.0023954874,  0.9219262395,  0.0013681845],
                       [ 0.0070442678,  0.8307209928, -0.0007938577],
                       [ 0.0089643198,  0.7637942355,  0.0023902527],
                       [-0.0031777987, -0.0020738436,  0.8190368257],
                       [ 0.0060579383,  0.0070006435,  0.8130428008],
                       [-0.0041559830, -0.0117483714,  0.6734727957],
                       [-0.0003143868, -0.0002077725,  0.7248903305],
                       [-0.0008859088,  0.0070824021,  0.5460441786]])
    Phi = np.array([[1.0000000000, 0.2885578, -0.0005263667],
                    [0.2885577889, 1.0000000,  0.3186286381],
                    [-0.0005263667, 0.3186286,  1.0000000000]])
    assert(np.allclose(model.L, Lambda, atol=1e-5))
    assert(np.allclose(model.Phi, Phi, atol=1e-5))

    
    
    