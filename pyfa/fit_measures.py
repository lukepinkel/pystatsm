# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 09:28:08 2021

@author: lukepinkel
"""
import numpy as np
import scipy as sp
import scipy.stats

def measure_of_sample_adequacy(Sigma):
    V = np.diag(np.sqrt(1/np.diag(Sigma)))
    R = V.dot(Sigma).dot(V)
    Rinv = np.linalg.inv(R)
    D = np.diag(1.0/np.sqrt(np.diag(Rinv)))
    Q = D.dot(Rinv).dot(D)
    ix = np.tril_indices(Sigma.shape[0], -1)
    r = np.sum(R[ix]**2)
    q = np.sum(Q[ix]**2)
    msa = r / (r + q)
    return msa

def srmr(Sigma, S, df):
    p = S.shape[0]
    y = 0.0
    t = (p + 1.0) * p
    for i in range(p):
        for j in range(i):
            y += (Sigma[i, j]-S[i, j])**2/(S[i, i]*S[j, j])
    
    y = np.sqrt((2.0 / (t)) * y)      
    return y

def lr_test(Sigma, S, df, n):
    p = Sigma.shape[0]
    _, lndS = np.linalg.slogdet(S)
    _, lndSigma = np.linalg.slogdet(Sigma)
    Sigma_inv = np.linalg.pinv(Sigma)
    chi2 = (lndSigma + np.trace(Sigma_inv.dot(S)) - lndS - p) * n
    chi2 = np.maximum(chi2, 1e-12)
    pval = sp.stats.chi2.sf(chi2, (p + 1)*p/2)
    return chi2, pval

def gfi(Sigma, S):
    p = S.shape[0]
    tmp1 = np.linalg.pinv(Sigma).dot(S)
    tmp2 = tmp1 - np.eye(p)
    y = 1.0 - np.trace(np.dot(tmp2, tmp2)) / np.trace(np.dot(tmp1, tmp1))
    return y

def agfi(Sigma, S, df):
    p = S.shape[0]
    t = (p + 1.0) * p
    tmp1 = np.linalg.pinv(Sigma).dot(S)
    tmp2 = tmp1 - np.eye(p)
    y = 1.0 - np.trace(np.dot(tmp2, tmp2)) / np.trace(np.dot(tmp1, tmp1))
    y = 1.0 - (t / (2.0*df)) * (1.0-y)
    return y
