#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 21:10:36 2020

@author: lukepinkel
"""


import numba # analysis:ignore
import numpy as np # analysis:ignore
import pandas as pd # analysis:ignore


def logcosh(x):
    y = np.log(np.cosh(x))
    return y

def gfunc(x):
    gx = np.tanh(x)
    dg = 1 - gx**2
    return gx, dg

def symmetric_decorr(W):
    u, V = np.linalg.eigh(W.T.dot(W))
    A = W.dot(V.dot(np.diag(np.sqrt(1/u))).dot(V.T))
    return A
    
   
def cov(X, Y=None):
    if Y is None:
        Y = X
    S = X.T.dot(Y) / X.shape[0]
    return S

@numba.jit(nopython=True)
def _convergence_check(W_new, W_old, n):
    prods = np.zeros((n))
    for i in range(n):
        prods[i] = np.dot(W_new[i], W_old[i])
    return prods
    
def convergence_check(W_new, W_old, n):
    prods = _convergence_check(W_new, W_old, n)
    conv = max(abs(abs(prods) - 1))
    return conv

def _fast_ica(Xw, W_init, maxiters=500):
    W = symmetric_decorr(W_init)
    Y = Xw.dot(W)
    dW_hist = []
    fhist = []
    for i in range(maxiters):    
        Gy, dGy = gfunc(Y)
        E_XtGY, E_dGy = cov(Xw, Gy), np.diag(np.mean(dGy, axis=0))
        W_new = symmetric_decorr(E_XtGY - (E_dGy).dot(W))
        dW_hist.append(convergence_check(W_new, W, W.shape[1]))
        if dW_hist[-1]<1e-6:
            break
        W = W_new
        Y = Xw.dot(W)
        fhist.append(-logcosh(Y).sum())
    return W, Y, dW_hist, fhist


def _fast_ica2(Xw, W_init, maxiters=500):
    W = symmetric_decorr(W_init)
    Y = Xw.dot(W)
    dW_hist = []
    fhist = []
    for i in range(maxiters):    
        Gy, dGy = gfunc(Y)
        EGYYt = cov(Y, Gy)
        beta = np.diag(EGYYt)
        alpha = -1.0 / (beta + np.mean(dGy, axis=0))
        A = np.diag(alpha)
        B = np.diag(beta)
        W_new = symmetric_decorr(W + (A.dot(B+EGYYt)).dot(W))
        dW_hist.append(convergence_check(W_new, W, W.shape[1]))
        if dW_hist[-1]<1e-6:
            break
        W = W_new
        Y = Xw.dot(W)
        fhist.append(-logcosh(Y).sum())
    return W, Y, dW_hist, fhist

class ICA:
    
    def __init__(self, X, n_comps=None, algorithm='fastica'):
        self.n_obs, self.p_vars = X.shape
        if type(X) is pd.DataFrame:
            X, cols, index, is_pd = X.values, X.columns, X.index, True
        else:
            X, cols, index = X, np.arange(self.p_vars), np.arange(self.n_obs)
            is_pd = False
        
        if n_comps is None:
            n_comps = self.p_vars
        
        X = (X - np.mean(X, axis=0)) 
        self.n_comps = n_comps
        self.X, self.cols, self.index = X, cols, index
        self.S = cov(X)
        u, V = np.linalg.eigh(self.S)
        V = V[:, u.argsort()[::-1]]
        u = u[u.argsort()[::-1]]
        self.u, self.V = u, V
        self.Vs = self.V[:, :self.n_comps].dot(np.diag(np.sqrt(1/self.u[:self.n_comps])))
        self.Xw = self.X.dot(self.Vs)
        self.W_init = np.random.normal(0, 1, size=(self.n_comps, 
                                                   self.n_comps))
        self.algorithm = algorithm
        self.is_pd = is_pd
        
    def fit(self, maxiters=500):
        if self.algorithm=='fastica':
            W, Y, dW_hist, fhist = _fast_ica(self.Xw, self.W_init, maxiters)
        elif self.algorithm=='mle':
            W, Y, dW_hist, fhist = _fast_ica2(self.Xw, self.W_init, maxiters)
        self.W = self.Vs.dot(W)
        self.A = np.linalg.pinv(self.W)
        self.Y, self.dW_hist, self.fhist = Y, dW_hist, fhist
                    
        if self.is_pd:
            cnames = [f"component_{i}" for i in range(1, self.n_components+1)]
            self.A = pd.DataFrame(self.A, index=cnames, columns=self.cols)
            self.W = pd.DataFrame(self.W, index=self.cols, columns=cnames)
            self.Y = pd.DataFrame(self.Y, index=self.index, columns=cnames)
        
        