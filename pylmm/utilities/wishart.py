#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 23:01:25 2020

@author: lukepinkel
"""
import numba
import numpy as np

@numba.jit(nopython=True)
def wishart(df, V):
    n = V.shape[0]
    T = np.zeros((n, n))
    for i in range(n):
        T[i, i] = np.sqrt(np.random.chisquare(df-(i+1)+1))
        for j in range(i):
            T[i, j] = np.random.normal(0.0, 1.0)
    L = np.linalg.cholesky(V)
    A = L.dot(T)
    W = A.dot(A.T)
    return W

@numba.jit(nopython=True)
def invwishart(df, V):
    n = V.shape[0]
    T = np.zeros((n, n))
    for i in range(n):
        T[i, i] = np.sqrt(np.random.chisquare(df-(i+1)+1))
        for j in range(i):
            T[i, j] = np.random.normal(0.0, 1.0)
    L = np.linalg.cholesky(V)
    A = L.dot(T)
    W = A.dot(A.T)
    IW = np.linalg.inv(W)
    return IW

def r_invwishart(df, V):
    Vinv = np.linalg.inv(V)
    return invwishart(df, Vinv)

def r_invgamma(df, scale):
    return 1.0/np.random.gamma(df, scale=1/scale)
    