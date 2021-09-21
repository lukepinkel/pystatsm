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
from ..utilities.data_utils import _check_type
from ..utilities.linalg_operations import _check_shape
from .m_estimators import (Huber, Bisquare, Hampel, Laplace) # analysis:ignore

class RLS:
    
    def __init__(self, formula, data, method=Huber()):
        self.f = method
        self.Ydf, self.Xdf = patsy.dmatrices(formula, data, 
                                             return_type='dataframe')
        
        self.X, self.xcols, self.xix, _ = _check_type(self.Xdf)
        self.Y, self.ycols, self.yix, _ = _check_type(self.Ydf)
        self.n_obs, self.p = self.X.shape
    
    def scale(self, r):
        return self.f.estimate_scale(r)

    
    def _correction_factor(self, r):
        tmp = self.f.phi_func(r).mean()
        correction_factor = 1.0 + self.p / self.n_obs * (1 - tmp) / tmp
        return correction_factor
    
    
        
    def fit(self, n_iters=200, tol=1e-10):
        X, Y = self.X, self.Y
        n, p = self.n_obs, self.p
        w = np.ones((self.n_obs, 1))
        b0 = np.zeros(p)
        dhist = []
        for i in range(n_iters):
            if w.ndim==1:
                w = w.reshape(w.shape[0], 1)
            Xw = X * w
            XtWX_inv = np.linalg.pinv(np.dot(Xw.T, X))
            beta = XtWX_inv.dot(np.dot(Xw.T, Y))
            resids = _check_shape(Y, 1) - _check_shape(X.dot(beta), 1)
            s = self.scale(resids)
            u = resids / s
            w = self.f.weights(u)
            
            db = np.sum(np.abs(beta-b0))
            dhist.append(db)
            if db < tol:
                break
            b0 = beta
        dfe = n - p
        XtX = np.dot(X.T, X)
        
        G1 = np.linalg.inv(XtX)
        G2 = np.linalg.inv(np.dot((X * _check_shape(w, 2)).T, X))
        G3 = G2.dot(XtX).dot(G2)
        
        k = self._correction_factor(resids)
        scale = self.scale(resids)
        
        num = np.sum(self.f.psi_func(u)**2) / dfe * scale**2
        den = (self.f.phi_func(u)).sum() / n
        
        v2 = num / den * k
        v1 = v2 / den * k
        v3 = num / k
        
        H1 = v1 * G1
        H2 = v2 * G2
        H3 = v3 * G3
        
        se1 = np.sqrt(np.diag(H1))[:, None]
        se2 = np.sqrt(np.diag(H2))[:, None]
        se3 = np.sqrt(np.diag(H3))[:, None]
       
        res = pd.DataFrame(np.hstack([beta, se1, se2, se3]))
        res.columns = ['beta', 'SE1', 'SE2', 'SE3']
        res.index = self.xcols
        res['t1'] = res['beta'] / res['SE1']
        res['t2'] = res['beta'] / res['SE2']
        res['t3'] = res['beta'] / res['SE3']
        
        res['p1'] = sp.stats.t.sf(np.abs(res['t1']), n-p)
        res['p2'] = sp.stats.t.sf(np.abs(res['t2']), n-p)
        res['p3'] = sp.stats.t.sf(np.abs(res['t3']), n-p)
        
        
        yhat = _check_shape(X.dot(beta), 1)
        mu_h = np.median(Y)
        sst = self.f.rho_func((Y - mu_h) / scale).sum()
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
        self.resids = resids
        self.dhist = dhist
        self.yhat = yhat
        self.s2 = scale
        self.sse, self.ssr, self.sst = sse, ssr, sst
        self.r2 =r2
        self.mu_h = mu_h
        self.deviance = dev
        
        
        
        