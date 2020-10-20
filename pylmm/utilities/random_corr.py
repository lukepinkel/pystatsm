#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 16 21:48:19 2020

@author: lukepinkel
"""

import numba # analysis:ignore
import numpy as np # analysis:ignore
import scipy as sp # analysis:ignore
from utilities.linalg_operations import whiten
   


def multi_rand(R, size=1000):    
    n = R.shape[0]
    X = np.random.normal(size=(size, n))
    X -= X.mean(axis=0)
    X /= X.std(axis=0)
    
    X = whiten(X)
    X -= X.mean(axis=0)
    X /= X.std(axis=0)
    
    W = np.linalg.cholesky(R)
    Y = X.dot(W.T)
    return Y


@numba.jit(nopython=True)
def vine_corr(d, betaparams=10):
    P = np.zeros((d, d))
    S = np.eye(d)
    for k in range(d-1):
        for i in range(k+1, d):
            P[k, i] = np.random.beta(betaparams, betaparams)
            P[k, i] = (P[k, i] - 0.5)*2.0
            p = P[k, i]
            for l in range(k-1, 1, -1):
                p = p * np.sqrt((1 - P[l, i]**2)*(1 - P[l, k]**2)) + P[l, i]*P[l, k]
            S[k, i] = p
            S[i, k] = p
    u, V = np.linalg.eigh(S)
    umin = np.min(u[u>0])
    u[u<0] = [umin*0.5**(float(i+1)/len(u[u<0])) for i in range(len(u[u<0]))]
    S = V.dot(np.diag(u)).dot(V.T)
    v = np.diag(S)
    v = np.diag(1/np.sqrt(v))
    S = v.dot(S).dot(v)
    return S

@numba.jit(nopython=True)
def onion_corr(d, betaparams=10):
    beta = betaparams + (d - 2) / 2
    u = np.random.beta(beta, beta)
    r12 = 2 * u  - 1
    S = np.array([[1, r12], [r12, 1]])
    I = np.array([[1.0]])
    for i in range(3, d+1):
        beta -= 0.5
        r = np.sqrt(np.random.beta((i - 1) / 2, beta))
        theta = np.random.normal(0, 1, size=(i-1, 1))
        theta/= np.linalg.norm(theta)
        w = r * theta
        c, V = np.linalg.eig(S)
        R = (V * np.sqrt(c)).dot(V.T)
        q = R.dot(w)
        S = np.concatenate((np.concatenate((S, q), axis=1),
                            np.concatenate((q.T, I), axis=1)), axis=0)
    return S
        