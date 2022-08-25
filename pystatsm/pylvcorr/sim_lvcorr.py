#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 14:48:51 2022

@author: lukepinkel
"""

import numpy as np
import scipy as sp
import scipy.special


class LVCorrSim(object):
    
    def __init__(self, corr_mat=None, x_bins=None, y_bins=None, rng=None):
        rng = np.random.default_rng(123) if rng is None else rng
        if type(x_bins) is int:
            x_bins = np.linspace(0, 1, x_bins+1, endpoint=False)[1:]
        if y_bins==False:
            kind = "polyserial"
        else:
            kind = "polychoric"
        if type(y_bins) is int:
            y_bins = np.linspace(0, 1, y_bins+1, endpoint=False)[1:]
        
        self.corr_mat = self.R = corr_mat
        self.x_bins, self.y_bins = x_bins, y_bins
        self.kind = kind
        self.rng = rng
        
    def simulate(self, n_obs=1000):
        X = self.rng.multivariate_normal(mean=np.zeros(2), cov=self.R, size=(n_obs))
        x = np.digitize(X[:, 0], np.quantile(X[:, 0], self.x_bins))
        if self.kind == "polychoric":
            y = np.digitize(X[:, 1], np.quantile(X[:, 1], self.y_bins))
        else:
            y = X[:, 1]
        return x, y



def random_thresholds(nvar, ncat=None, min_cat=None, max_cat=None, rng=None):
    rng = np.random.default_rng(123) if rng is None else rng
    if ncat is None:
        if type(min_cat) in [int, float]:
            min_cat = np.repeat(min_cat, nvar)
        if type(max_cat) in [int, float]:
            max_cat = np.repeat(max_cat, nvar)
        ncat = np.zeros(nvar, dtype=int)
        for i in range(nvar):
            ncat[i] = rng.choice(np.arange(min_cat[i]+2, max_cat[i]+2), 1)
    elif type(ncat) in [int, float]:
        ncat = np.repeat(ncat, nvar)
    quantiles = {}
    taus = {}
    for i in range(nvar):
        q = np.r_[0, rng.dirichlet(np.ones(ncat[i]-2)).cumsum()]
        q[-1] = 1.0
        quantiles[i] = q
        taus[i] = sp.special.ndtri(q)
    return quantiles,taus


class PolycoricSim(object):
    
    def __init__(self, R, taus, rng=None):
        self.rng = np.random.default_rng(123) if rng is None else rng
        self.R = R
        self.taus = taus
        self.p = R.shape[0]
        self.cat_sizes = [len(self.taus[i]) for i in range(len(self.taus))]
        
    def simulate_data(self, n_obs=1000, exact=False):
        Z = self.rng.multivariate_normal(mean=np.zeros(self.p), cov=self.R, size=n_obs)
        Y = np.zeros_like(Z)
        for i in range(self.p):
            Y[:, i] = np.digitize(Z[:, i], self.taus[i])
        return Z, Y