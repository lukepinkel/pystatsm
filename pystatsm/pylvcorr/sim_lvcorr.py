#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 14:48:51 2022

@author: lukepinkel
"""

import numpy as np



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
