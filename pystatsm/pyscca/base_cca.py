#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 17:06:36 2022

@author: lukepinkel
"""

import numpy as np
import pandas as pd
from ..utilities.linalg_operations import eighs, cproject
from ..utilities.data_utils import _csd, _handle_pandas

        

class CCA(object):
    
    @staticmethod
    def _eig_isq(S):
        u, V = eighs(S)
        u_mask = u > 1e-16
        u_isq = u.copy()
        u_isq[u_mask] = 1.0 / np.sqrt(u_isq[u_mask])
        S_isq = (V * u_isq).dot(V.T)
        return u, u_mask, u_isq, V, S_isq
    
    
    @staticmethod
    def _cca_svd(Sxy, Sxx_isq, Syy_isq):
        U, rhos, Vt = np.linalg.svd(Sxx_isq.dot(Sxy).dot(Syy_isq), full_matrices=False)
        V = Vt.T
        Wx, Wy = Sxx_isq.dot(U), Syy_isq.dot(V)
        return Wx, Wy, rhos
    
    def __init__(self, X=None, Y=None):
        self._handle_full_data(X, Y)
        self._decompose_mats()
        
    def _handle_full_data(self, X, Y):
        _X, xcols, xinds = _handle_pandas(X, "x")
        _Y, ycols, yinds = _handle_pandas(Y, "y")
        X, _X_mean, _X_std = _csd(X, True)
        Y, _Y_mean, _Y_std = _csd(Y, True)
        n, p = X.shape
        q = Y.shape[1]
        Sxx = np.dot(X.T, X) / n
        Syy = np.dot(Y.T, Y) / n
        Sxy = np.dot(X.T, Y) / n
        self.X, self._X, self._X_mean, self._X_std = X, _X, _X_mean, _X_std
        self.Y, self._Y, self._Y_mean, self._Y_std = Y, _Y, _Y_mean, _Y_std
        self.xcols, self.xinds = xcols, xinds
        self.ycols, self.yinds = ycols, yinds
        self.n_xvars = self.p = p
        self.n_yvars = self.q = q
        self.n_obs = self.n = n
        self.x_corr = self.Sxx = Sxx
        self.y_corr = self.Syy = Syy
        self.xy_corr = self.Sxy = Sxy

        
    def _decompose_mats(self, Sxx=None, Syy=None):
        Sxx = self.Sxx if Sxx is None else Sxx
        Syy = self.Syy if Syy is None else Syy
        
        self.ux, _, _, self.Vx, self.Sxx_isq = self._eig_isq(Sxx)
        self.uy, _, _, self.Vy, self.Syy_isq = self._eig_isq(Syy)
    
    def predict(self, X, Y):
        Zx, Zy = np.dot(X, self.Wx), np.dot(Y, self.Wy)
        return Zx, Zy
        
    def _fit(self, n_comps=None, compute_projections=True, compute_canvars=True):
        n_comps = self.p if n_comps is None else n_comps
        Wx, Wy, rhos = self._cca_svd(self.Sxy, self.Sxx_isq, self.Syy_isq)
        Wx, Wy = Wx[:, :n_comps], Wy[:, :n_comps]
        Lx, Ly = self.Sxx.dot(Wx), self.Syy.dot(Wy)
        u_cols = [f"u{i+1}" for i in range(n_comps)]
        v_cols = [f"v{i+1}" for i in range(n_comps)]
        self.Wx, self.Wy, self.Lx, self.Ly = Wx, Wy, Lx, Ly
        self.x_coefs = pd.DataFrame(Wx, index=self.xcols, columns=u_cols)
        self.y_coefs = pd.DataFrame(Wy, index=self.ycols, columns=v_cols)
        self.x_loadings = pd.DataFrame(Lx, index=self.xcols, columns=u_cols)
        self.y_loadings = pd.DataFrame(Ly, index=self.ycols, columns=v_cols)
        self.rhos = rhos
        
        if compute_projections:
            self.PWx, self.PWy = cproject(self.Wx), cproject(self.Wy)
            self.PLx, self.PLy = cproject(self.Lx), cproject(self.Ly)
        
        if compute_canvars:
            self.Zx, self.Zy = self.predict(self.X, self.Y)
            self.x_canvar = pd.DataFrame(self.Zx, index=self.xinds, columns=u_cols)
            self.y_canvar = pd.DataFrame(self.Zy, index=self.yinds, columns=v_cols)
            self.Z = np.hstack((self.Zx, self.Zy))
            self.canvar = pd.concat([self.x_canvar, self.y_canvar], axis=1)
            

            
    
    