#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 15:10:53 2020

@author: lukepinkel
"""
import tqdm

import numpy as np
import scipy as sp
import scipy.signal

def _slice_at_axis(sl, axis):
    return (slice(None),) * axis + (sl,) + (...,)

def _check_n_shift(n_shift):
    res = []
    for x in n_shift:
        if hasattr(x, '__len__'):
            if len(x)==2:
                res.append(x)
            elif len(x)==1:
                x, = x
                res.append((x, x))
        else:
            res.append((x, x))
    return res
        
    

def iterable(obj):
    try:
        iter(obj)
    except Exception:
        return False
    else:
        return True
    
    
def shift1d(arr, n_shift, constant_value=0.0):
    res = np.empty_like(arr)
    if n_shift > 0:
        res[:n_shift] = constant_value
        res[n_shift:] = arr[:-n_shift]
    elif n_shift < 0:
        res[n_shift:] = constant_value
        res[:n_shift] = arr[-n_shift:]
    else:
        res[:] = arr
    return res

def shift(arr, n_shift, constant_value=0.0):
    res = np.empty_like(arr)
    if iterable(n_shift):
        n_shift = _check_n_shift(n_shift)
    for i, (nl, nr) in enumerate(n_shift):
        arr = arr[_slice_at_axis(slice(nl, None), i)]
        arr = arr[_slice_at_axis(slice(None, arr.shape[i]-nr), i)]
        
    res[:] = np.pad(arr, n_shift, mode='constant', 
                    constant_values=constant_value)
    return res
    
def conv2(A, B, mode='full'):
    C = sp.signal.convolve2d(A, B, mode='full')
    if mode=='same':
        a1 = int(np.floor(C.shape[0] / 2) - 1)
        a2 = int(np.floor(C.shape[1] / 2) - 1)
        b1, b2 = int(a1+A.shape[0]), int(a2+A.shape[1])
        C = C[a1:b1, a2:b2]
    return C
 
        

def reconstruct(W, H):
    n, k, L = W.shape
    t = int(np.product(H.shape[1:])) + 2 * L
    H = np.pad(H, ((0, 0), (L, L)))
    X_hat = np.zeros((n, t))
    for tau in range(L):
        X_hat += W[:, :, tau].dot(np.roll(H, (0, tau)))
    X_hat = X_hat[:, L:-L]
    return X_hat
    
def seqnmf_objfunc(X, W, H):
    n, k, L = W.shape
    t = int(np.product(H.shape[1:])) 
    X_hat = reconstruct(W, H)
    Sk = np.ones((1, 2*L-1))
    WtX = np.zeros((k, t))
    for i in range(L):
        X_shifted = shift(X, ((0, 0), (0, i)))
        WtX += W[:, :, i].T.dot(X_shifted)
    WtXSHt = (sp.signal.convolve2d(WtX, Sk, 'same').dot(H.T))
    WtXSHt = WtXSHt * (1 - np.eye(WtXSHt.shape[0])) # probably not very effecient
    fval = np.linalg.norm(X - X_hat)
    penalty = np.linalg.norm(WtXSHt, 1)
    return fval, penalty, WtX

def _initparams(X, k, L):
    X_pad = np.pad(X, ((0, 0), (L, L)))
    xmax = X_pad.max()
    n, t = X_pad.shape
    W_init = xmax * np.random.uniform(0, 1, size=(n, k, L))
    H_init = xmax * np.random.uniform(0, 1, size=(k, t))/np.sqrt(t/3)
    M = np.ones((n, t))
    Sk = np.ones((1, 2 * L - 1))
    _dim_dict = dict(n=n, k=k, t=t, L=L)
    return W_init, H_init, M, Sk, _dim_dict, xmax

def center_factors(W, H):
    n, k, L = W.shape
    center = np.maximum(np.floor(L / 2), 1)
    W_pad  = np.pad(W, ((0, 0), (0, 0), (L, L)))
    for i in range(k):
        tmp = np.sum(np.squeeze(W[:, i]), axis=0)
        v = np.arange(1, len(tmp)+1)
        cmass = np.maximum(np.floor(np.sum(tmp*v)/np.sum(tmp)), 1)
        W_pad[:, i] = np.roll(np.squeeze(W_pad[:, i]), int(center-cmass), 
                              axis=1)
        H[i] = np.roll(H[i], int(cmass-center), axis=1)
    W = W_pad[:, :, L:-L]
    return W



class NMFSeq:
    
    def __init__(self, X, k=10, L=100, lambda_=0.001, lambda_w=0.001, 
                 lambda_h=0.001, n_starts=10, n_iters=100, shift_factors=False):
        W_init, H_init, M, Sk, _dim_dict, xmax = _initparams(X, k, L)
        self.X = np.pad(X, ((0, 0), (L, L)))
        self.W_init = W_init
        self.H_init = H_init
        self.M = M
        self.Sk = Sk
        self._dims = _dim_dict
        self.xmax = xmax
        self.n_starts = n_starts
        self.n_iters = n_iters
        self.f_evals = np.zeros((n_iters, n_starts))
        self.penalty_evals = np.zeros((n_iters, n_starts))
        self.lambda_ = lambda_
        self.lambda_w = lambda_w
        self.lambda_h = lambda_h
        self.RI = 1 - np.eye(self._dims['k'])
        self.shift_factors = shift_factors
        self.Ws = np.zeros(W_init.shape+(n_starts,))
        self.Hs = np.zeros(H_init.shape+(n_starts,))
        
        
    def re_init_params(self):
        n, k, L = self._dims['n'], self._dims['k'], self._dims['L']
        t = self._dims['t']
        self.W_init = self.xmax * np.random.uniform(0, 1, size=(n, k, L))
        self.H_init = self.xmax * np.random.uniform(0, 1, size=(k, t))/np.sqrt(t/3)
        
    def _get_penaltyH(self, WtX, H):
        dRdH = self.lambda_ * self.RI.dot(sp.signal.convolve2d(WtX, self.Sk, 'same'))
        dHHdH = self.lambda_h * self.RI.dot(sp.signal.convolve2d(H, self.Sk, 'same'))
        dRdH += dHHdH
        return dRdH
    
    def _get_penaltyW(self, Xs, H_shiftedt, W_flat):
        dRdW = self.lambda_ * Xs.dot(H_shiftedt).dot(self.RI)
        dWWdW = self.lambda_w * W_flat.dot(self.RI)
        dRdW += dWWdW
        return dRdW
        
    def _fit(self, run):
        epsilon = np.finfo(float).eps
        varepsilon = epsilon**(1.0/3.9)
        W, H = self.W_init, self.H_init
        k, L = self._dims['k'], self._dims['L']
        t = self._dims['t']
        X_hat = reconstruct(W, H)
        self.f_evals[0, run] = np.linalg.norm(self.X - X_hat)
        for i in range(self.n_iters):
            WtX, WtX_hat = np.zeros((k, t)), np.zeros((k, t))
            for j in range(L):
                X_shifted = np.roll(self.X.copy(), -j, axis=1)
                X_hat_shifted = np.roll(X_hat, -j, axis=1)
                WtX += W[:, :, j].T.dot(X_shifted)
                WtX_hat += W[:, :, j].T.dot(X_hat_shifted)
            dRdH = self._get_penaltyH(WtX, H)
            H = H * WtX / (WtX_hat + dRdH + epsilon)
            
            if self.shift_factors:
                W, H = center_factors(W, H)
                W += varepsilon
            row_norms = np.sqrt(np.sum(H**2, axis=1))
            H = (np.diag(1.0 / (row_norms + epsilon))).dot(H)
            for j in range(L):
                W[:, :, j] = W[:, :, j].dot(np.diag(row_norms))
            X_hat = reconstruct(W, H)
            Wflat = np.sum(W, axis=2)
            Xs = sp.signal.convolve2d(self.X.copy(), self.Sk, 'same')
            for j in range(L):
                H_shiftedt = np.roll(H, j, axis=1).T
                XHt = self.X.dot(H_shiftedt)
                X_hatHt = X_hat.dot(H_shiftedt)
                dRdW = self._get_penaltyW(Xs, H_shiftedt, Wflat)
                W[:, :, j] = W[:, :, j] * XHt / (X_hatHt + dRdW + varepsilon)
            X_hat = reconstruct(W, H)
            _, penalty, _ = seqnmf_objfunc(self.X, W, H)
            self.f_evals[i, run] = np.linalg.norm(self.X - X_hat)
            self.penalty_evals[i, run] = penalty
        self.Ws[..., run] = W
        self.Hs[..., run] = H
    
    def fit(self, verbose=False):
        if verbose:
            for i in tqdm.tqdm(range(self.n_starts)):
                self._fit(i)
                self.re_init_params()
        else:
            for i in range(self.n_starts):
                self._fit(i)
                self.re_init_params()
            
                
                
#    
#W1 = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
#               [1.0, 0.0, 0.0, 0.0, 0.0],
#               [0.0, 1.0, 0.0, 0.0, 0.0],
#               [0.0, 0.0, 1.0, 0.0, 0.0],
#               [0.0, 0.0, 0.0, 0.0, 0.0],
#               [0.0, 0.0, 0.0, 0.0, 0.0],
#               [0.0, 0.0, 0.0, 0.0, 0.0],
#               [0.0, 0.0, 0.0, 0.0, 0.0],
#               [0.0, 0.0, 0.0, 0.0, 0.0],
#               [0.0, 0.0, 0.0, 0.0, 0.0],
#               [0.0, 0.0, 0.0, 0.0, 0.0],
#               [0.0, 0.0, 0.0, 0.0, 0.0]])
#    
#W2 = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
#               [0.0, 0.0, 0.0, 0.0, 0.0],
#               [0.0, 0.0, 0.0, 0.0, 0.0],
#               [0.0, 0.0, 0.0, 0.0, 0.0],
#               [0.0, 0.0, 0.0, 0.0, 0.0],
#               [0.0, 0.0, 0.0, 0.0, 0.0],
#               [0.0, 0.0, 0.0, 0.0, 0.0],
#               [0.0, 0.0, 0.0, 0.0, 0.0],
#               [0.0, 0.0, 0.0, 0.0, 0.0],
#               [0.0, 0.0, 0.0, 0.0, 1.0],
#               [0.0, 0.0, 0.0, 1.0, 0.0],
#               [0.0, 0.0, 1.0, 0.0, 0.0]])
#
#H1 = np.concatenate([np.pad(np.ones(1), (8, 7)) for i in range(20)])[:,  None].T
#    
#
#H2 = np.concatenate([np.pad(np.ones(1), (0, 15)) for i in range(20)])[:,  None].T
#
#    
#
#
#
#X = conv2(W1, H1) + conv2(W2, H2)
#
#L = 5
#k = 2
#
#model = NMFSeq(X, k=k, L=L, n_iters=200, n_starts=100,
#               lambda_=0.1, lambda_w=0.001, lambda_h=0.001)
#model.fit(verbose=1)
#
#f_evals = model.f_evals
#penalty_evals = model.penalty_evals
#
#fig, ax = plt.subplots(nrows=3)
#ax[0].plot(f_evals)
#ax[1].plot(penalty_evals)
#ax[2].plot(f_evals+penalty_evals)
#
#Ws = model.Ws
#Hs = model.Hs
#Hs = Hs[:, 7:-7]
#
#converged = ((f_evals<3)*(penalty_evals<25))[-1]
#H1s = Hs[0][:, converged]
#H2s = Hs[1][:, converged]
#W1s = Ws[:,0][:, :, converged]
#W2s = Ws[:,1][:, :, converged]
#
#W1s.mean(axis=-1)
#
#fig, ax = plt.subplots(nrows=2)
#ax[0].imshow(W1s.mean(axis=-1))
#ax[1].imshow(W2s.mean(axis=-1))
#
#
#fig, ax = plt.subplots(nrows=2)
#ax[0].plot(H1s.mean(axis=-1))
#ax[1].plot(H2s.mean(axis=-1))
#
#fig, ax = plt.subplots(nrows=2, sharex=True)
#ax[0].matshow(W1s.reshape(np.product(W1.shape[:2]), -1), aspect='auto')
#ax[1].matshow(W2s.reshape(np.product(W2.shape[:2]), -1), aspect='auto')
#
#
