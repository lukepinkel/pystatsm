# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 12:54:05 2021

@author: lukepinkel
"""
import numpy as np

def diagonalize_smooth(nx, smooths):
    T, Ti = np.eye(nx), np.eye(nx)
    reparam = {}
    repmats = {}
    ixp, ixf, ixl = [], [], []
    for i, (key, val) in enumerate(smooths.items()):
        S, ix = val['S'], val['ix']
        D, U = np.linalg.eigh(S)
        r = np.argsort(D)[::-1]
        U, D = U[:, r], D[r]
        D[-1] = 1
        D = np.sqrt(1.0 / D)
        V, Vi = U.dot(np.diag(D)), U.dot(np.diag(1/D))
        T[ix, ix[:, None]] = V.T
        Ti[ix, ix[:, None]] = Vi
        reparam[key] = dict(V=V, Vi=Vi, D=D)
        ixp.append(ix[:-1])
        ixf.append(np.atleast_1d(np.asarray(ix[-1])))
        ixl.append(np.repeat(i, len(ix)-1))
    repmats['T'] = T
    repmats['Ti'] = Ti
    ixp = np.concatenate(ixp)
    ixf = np.concatenate([np.arange(min(ixp))]+ixf)
    ixl = np.concatenate(ixl)
    ixp, ixf, ixl = (ixp+1).astype(int), (ixf+1).astype(int), (ixl+1).astype(int)
    return reparam, repmats, ixp, ixf, ixl
        
