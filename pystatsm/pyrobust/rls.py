#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 23:16:43 2020

@author: lukepinkel
"""
import patsy
import numpy as np
import scipy as sp
import scipy.stats
import pandas as pd
from ..utilities.data_utils import _check_type, _check_shape
from .m_estimators import (Huber, Bisquare, Hampel, Laplace) # analysis:ignore

class RLS:
    
    def __init__(self, formula, data, method=Huber()):
        self.f = method
        self.Ydf, self.Xdf = patsy.dmatrices(formula, data, 
                                             return_type='dataframe')
        
        self.X, self.xcols, self.xix, _ = _check_type(self.Xdf)
        self.Y, self.ycols, self.yix, _ = _check_type(self.Ydf)
        self.y = _check_shape(self.Y, 1)
        self.n_obs, self.p = self.X.shape
    
    def scale(self, r):
        return self.f.estimate_scale(r)

    
    def _correction_factor(self, r):
        tmp = self.f.phi_func(r).mean()
        correction_factor = 1.0 + self.p / self.n_obs * (1 - tmp) / tmp
        return correction_factor
    
    def _optimize_rho(self, n_iters=200, tol=1e-10):
        w = np.ones((self.n_obs,))
        b_old = np.ones(self.p) * 1e16
        for i in range(n_iters):
            Xw = self.X * w.reshape(-1, 1)
            b = np.linalg.solve(Xw.T.dot(self.X), Xw.T.dot(self.y))
            r = self.y - self.X.dot(b)
            s = self.scale(r)
            u = r / s
            w = self.f.weights(u).reshape(-1, 1)
            db = np.sum(np.abs(b - b_old))
            if db < tol:
                break
            b_old = b
        return b, w, u, r, s

        
    def fit(self, n_iters=200, tol=1e-10):
        beta, w, u, r, s = self._optimize_rho(n_iters, tol)
        dfe = self.n_obs - self.p
        XtX = np.dot(self.X.T, self.X)
        
        G1 = np.linalg.inv(XtX)
        G2 = np.linalg.inv(np.dot((self.X * w.reshape(-1, 1)).T, self.X))
        G3 = G2.dot(XtX).dot(G2)
        
        k = self._correction_factor(r)
        scale = self.scale(r)
        
        num = np.sum(self.f.psi_func(u)**2) / dfe * scale**2
        den = (self.f.phi_func(u)).sum() / self.n_obs
        
        v2 = num / den * k
        v1 = v2 / den * k
        v3 = num / k
        
        H1 = v1 * G1
        H2 = v2 * G2
        H3 = v3 * G3
        
        se1 = np.sqrt(np.diag(H1))
        se2 = np.sqrt(np.diag(H2))
        se3 = np.sqrt(np.diag(H3))
       
        res = pd.DataFrame(np.vstack([beta, se1, se2, se3]).T)
        res.columns = ['beta', 'SE1', 'SE2', 'SE3']
        res.index = self.xcols
        res['t1'] = res['beta'] / res['SE1']
        res['t2'] = res['beta'] / res['SE2']
        res['t3'] = res['beta'] / res['SE3']
        
        res['p1'] = sp.stats.t.sf(np.abs(res['t1']), dfe)
        res['p2'] = sp.stats.t.sf(np.abs(res['t2']), dfe)
        res['p3'] = sp.stats.t.sf(np.abs(res['t3']), dfe)
        
        
        yhat = self.X.dot(beta)
        mu_h = np.median(self.y)
        sst = self.f.rho_func((self.y - mu_h) / scale).sum()
        sse = self.f.rho_func(u).sum()
        ssr = sst - sse
        
        r2 = ssr / sse
        
        dev = 2*scale**2 * sse
        
        self.res = res
        self._G1, self._G2, self._G3 = G1, G3, G3
        self._v1, self._v2, self._v3 = v1, v2, v3
        self.H1, self.H2, self.H3 = H1, H2, H3
        self.se1, self.se2, self.se3 = se1, se2, se3
        self.beta = beta
        self.u = u
        self.w = w
        self.r = r
        self.yhat = yhat
        self.s2 = scale
        self.sse, self.ssr, self.sst = sse, ssr, sst
        self.r2 =r2
        self.mu_h = mu_h
        self.deviance = dev
        
        
        
        