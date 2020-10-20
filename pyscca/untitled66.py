#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 20:26:42 2020

@author: lukepinkel
"""

import numpy as np

def fleiss_kappa(X):
    nc = np.sum(X, axis=0)
    nr = np.sum(X, axis=1)
    n = np.sum(X)
    prob_c = nc / n
    ssq_pc = np.sum(prob_c**2)
    ssq_pr = np.sum(X * (X - 1), axis=1) / (nr * (nr - 1))
    mean_pr = np.mean(ssq_pr)
    k = (mean_pr - ssq_pc) / (1 - ssq_pc)
    return k
    