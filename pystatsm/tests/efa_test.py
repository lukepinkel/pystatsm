# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 09:41:35 2021

@author: lukepinkel
"""

import numpy as np
from pystatsm.pyfa.factor_analysis import FactorAnalysis
from pystatsm.utilities.data_utils import center, scale_diag
from pystatsm.utilities.numerical_derivs import fo_fc_cd, so_gc_cd


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
    H1 = so_gc_cd(model.gradient, x)
    H2 = model.hessian(x)
    hess_close = np.allclose(H1, H2)
    
    assert(grad_close)
    assert(hess_close)
    
    Lambda = np.array([[ 0.25061323,  0.0244112 ,  0.93523923],
                       [ 0.25099206,  0.0261924 ,  0.94752117],
                       [ 0.24972876,  0.02676534,  0.93296061],
                       [ 0.24063282,  0.03238229,  0.921835  ],
                       [ 0.24081752,  0.02649816,  0.89947648],
                       [ 0.87462226,  0.3126195 ,  0.02739544],
                       [ 0.84604259,  0.30270555,  0.03155507],
                       [ 0.86783792,  0.31216616,  0.03800093],
                       [ 0.78327519,  0.27939661,  0.0390121 ],
                       [ 0.72075328,  0.26007619,  0.03819407],
                       [-0.01781903,  0.81785591, -0.02360019],
                       [-0.00680052,  0.81518426, -0.01417681],
                       [-0.0244936 ,  0.66906697, -0.02128594],
                       [-0.01359771,  0.72446353, -0.01841109],
                       [-0.00359984,  0.54814374, -0.01421527]])
    Phi = np.eye(3)
    assert(np.allclose(model.L, Lambda, atol=1e-5))
    assert(np.allclose(model.Phi, Phi, atol=1e-5))

    
    
    