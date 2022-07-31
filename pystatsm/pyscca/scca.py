#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 02:38:52 2020

@author: lukepinkel
"""


import tqdm# analysis:ignore
import numba # analysis:ignore
import numpy as np # analysis:ignore
import pandas as pd # analysis:ignore
from ..utilities.linalg_operations import vech # analysis:ignore

 
def make_lambdas(xlams, xlmin, xlmax):
    '''
    Parameters
    ----------
    xlams : int
        Number of lambdas
    
    xlmin : float
        lambda min
        
    xlmax : float
        lambda max

    Returns
    -------
    xlams : array_like
        Lambda range
        
    '''
    xlb = np.log(np.maximum(xlmin, 1e-16))
    xub = np.log(np.minimum(xlmax, 1.0-1e-16))
    xlams = np.exp(np.linspace(xlb, xub, xlams))      
    return xlams

def kfold_indices(n, k):
    '''
    Parameters
    ----------
    n : int
        Number of observations
    
    k : int
        Number of splits

    Returns
    -------
    inds : list
        List of indices
    '''
    splits = np.array_split(np.arange(n), k)
    inds = []
    for i in range(k):
        fit_ind = np.concatenate([splits[j] for j in range(k) if j!=i])
        inds.append([fit_ind, splits[i]])
    return inds
   
def crossval_mats(X, y, n, cv, center=True, standardize=True):
    '''
    Parameters
    ----------
    X : array_like
        Data matrix
    
    y : array_like
        Data matrix
        
    n : int
        number of observations
        
    cv : int
        Number of cross validation folds

    Returns
    -------
    Xf : array_like
        Design matrix for model fitting
    
    yf : array_like
        Dependent variables for model fitting
    
    Xt : array_like
        Design matrix for model testing
        
    yt : array_like
        Dependent variables for model testing
        
    '''
    kfix = kfold_indices(n, cv)
    Xf, yf, Xt, yt = [], [], [], []
    for f_ix, v_ix in kfix:
        Xfi, yfi, Xti, yti = X[f_ix], y[f_ix], X[v_ix], y[v_ix]
        
        if center: 
            Xfi -= Xfi.mean(axis=0)
            yfi -= yfi.mean(axis=0)
            Xti -= Xti.mean(axis=0)
            yti -= yti.mean(axis=0)
        if standardize: 
            Xfi /= Xfi.std(axis=0)
            yfi /= yfi.std(axis=0)
            Xti /= Xti.std(axis=0)
            yti /= yti.std(axis=0) 
        Xf.append(Xfi)
        yf.append(yfi)
        Xt.append(Xti)
        yt.append(yti)
    return Xf, yf, Xt, yt
    

 
def _corr(X, Y):
    '''
    Parameters
    ----------
    X : array_like
        n x p matrix of observations
    
    Y : array_like
        n x q matrix of observations

    Returns
    -------
    R : array_like
        p x q matrix of correlations
    '''
    X -= X.mean(axis=0)
    X /= X.std(axis=0)
    Y -= Y.mean(axis=0)
    Y /= Y.std(axis=0)
    R = X.T.dot(Y) / X.shape[0]
    return R

@numba.jit(nopython=True)
def l2_penalty(x):
    '''
    L2 Norm
    Parameters
    ----------
    x : array_like

    Returns
    -------
    y : float
        max(sqrt(sum(x**2)), 0.05)
    '''
    y = max(np.sum(x**2)**(0.5), 0.05)
    return y

@numba.jit(nopython=True)
def l1_constraint(x):
    '''
    L1 Norm
    Parameters
    ----------
    x : array_like
    Returns
    -------
    y : array_like
        sum(abs(x / L2Norm(x)))
    '''
    xn = l2_penalty(x)
    y = np.sum(np.abs(x / xn))
    return y

@numba.jit(nopython=True)
def sft(x, t):
    '''
    Soft Thresholding
    Parameters
    ----------
    x : array_like
        Values to be subject to soft thresholding
    t : array_like
        Threshold
    Returns
    -------
    y : array_like
        the input x subject to soft thresholding max(abs(x) - t, 0) * sign(x) 
    '''
    y = np.maximum(np.abs(x) - t, 0.0) * np.sign(x)
    return y
    

@numba.jit(nopython=True)
def threshold_search(v, c, n_iters=10_000, d=1e-6, tol=1e-7):
    '''
    Threshold Search
    Parameters
    ----------
    v : float
        The vector that after soft thresholding at t should have an L1 norm 
        less than c
    
    c : float
        The threshold
    
    n_iters : int
        Number of binary search iterations
    
    d : float
        Initial decrement
        
    tol : float
        Binary search tolerance

    Returns
    -------
    y : float
        Final threshold
        
    '''
    s = l2_penalty(v)
    if s==0 or np.sum(np.abs(v / s))<=c:
        return 0.0
    else:
        ls, rs = 0.0, np.max(np.abs(v)) - d
        for i in range(n_iters):
            vt = sft(v, (ls + rs) / 2.0)
            if l1_constraint(vt)<c:
                rs = (ls + rs) / 2.0
            else:
                ls = (ls + rs) / 2.0
            if (rs - ls) < tol:
                return (rs + ls) / 2.0
    return  (rs + ls) / 2.0

@numba.jit(nopython=True)  
def _scca_vec(S, wx, wy, cx, cy, n_iters=500, tol=1e-6):
    '''
    sCCA Algorithm for a single component
    Parameters
    ----------
    S : array_like
        Cross product matrix X'Y of size (p x q)
    
    wx : array_like
        Vector (of length p) of starting weights corresponding to X 
        
    wy : array_like
        Vector (of length q) of starting weights corresponding to Y
        
    cx : float
        wx regularization term
    
    cy : float
        wy regularization term
        
    n_iters : int
        Number of binary search iterations
    
    tol : float
        Convergence tolerance

    Returns
    -------
    wx : array_like
        Vector (of length p) of weights corresponding to X 
        
    wy : array_like
        Vector (of length q) of weights corresponding to Y
    
    dwx : float
        change in wx 
    
    i : int
        Number of iterations till convergence.
        
    '''
    for i in range(n_iters):
        wx_new = S.dot(wy)
        x_thresh = threshold_search(wx_new, cx)
        wx_new = sft(wx_new, x_thresh)
        wx_new = wx_new / l2_penalty(wx_new)
        dwx =  np.sum(np.abs(wx_new - wx))
        wx = wx_new
        
        wy_new = wx.dot(S)
        y_thresh = threshold_search(wy_new, cy)
        wy_new = sft(wy_new, y_thresh)
        wy = wy_new / l2_penalty(wy_new)
        
        if dwx < tol:
            break        
    return wx, wy, dwx, i
    
@numba.jit(nopython=True)  
def _scca(X, Y, wx, wy, cx, cy, n_comps, n_iters=500, tol=1e-6):
    '''
    sCCA Algorithm for multiple components component
    Parameters
    ----------
    X : array_like
        (n x p) matrix of observations
        
    Y : array_like
        (n x q) matrix of observations
    
    wx : array_like
        Vector (of length p) of starting weights corresponding to X 
        
    wy : array_like
        Vector (of length q) of starting weights corresponding to Y
        
    cx : float
        wx regularization term
    
    cy : float
        wy regularization term
        
    n_comps : int
        Number of components to extract
        
    n_iters : int
        Number of binary search iterations
    
    tol : float
        Convergence tolerance

    Returns
    -------
    Wx : array_like
        Matrix (p x n_comps) of weights corresponding to X 
        
    wy : array_like
         Matrix (q x n_comps) of weights corresponding to Y
    
    rhos : array_like
        
    
    optinfo : dict
        Optimization information
        
    '''
    S = X.T.dot(Y)
    Xt, Yt = X.copy(), Y.copy()
    Wx = np.zeros((n_comps, S.shape[0]))
    Wy = np.zeros((n_comps, S.shape[1]))
    optinfo = np.zeros((n_comps, 2))
    for i in range(n_comps):
        St = Xt.T.dot(Yt)
        wxi, wyi, dt, nc = _scca_vec(St, wx, wy, cx, cy, n_iters=n_iters, tol=tol)
        Wx[i], Wy[i] = wxi, wyi
        optinfo[i, 0] = dt
        optinfo[i, 1] = nc
        vx, vy = Xt.dot(wxi), Yt.dot(wyi)
        uX, uY = vx.T.dot(Xt), vy.T.dot(Yt)
        Xt = Xt - (np.outer(vx, uX) / np.sum(vx**2))
        Yt = Yt - (np.outer(vy, uY) / np.sum(vy**2))
    return Wx, Wy, optinfo
        
       
      
class SCCA:

    def __init__(self, X, Y):
        '''
        Sparse Canonical Correlation Analysis
    
        Parameters
        ----------
        X : array_like
            (n x p) matrix of data
        
        Y : array_like
            (n x q) matrix of data
    
        '''
        if type(X) is pd.DataFrame:
            X, xcols, xind, x_is_pd = X.values, X.columns, X.index, True
        else:
            X, xcols, xind, x_is_pd = X, None, None, False
        
        if type(Y) is pd.DataFrame:
            Y, ycols, yind, y_is_pd = Y.values, Y.columns, Y.index, True
        else:
            Y, ycols, yind, y_is_pd = Y, None, None, False
        
        self.xcols, self.xind, self.x_is_pd =  xcols, xind, x_is_pd
        self.ycols, self.yind, self.y_is_pd =  ycols, yind, y_is_pd
        
        self.X = (X - X.mean(axis=0)) / X.std(axis=0)
        self.Y = (Y - Y.mean(axis=0)) / Y.std(axis=0)
        self.Sxinv = np.linalg.inv(self.X.T.dot(self.X)).dot(self.X.T)
        self.Syinv = np.linalg.inv(self.Y.T.dot(self.Y)).dot(self.Y.T)
        self.n, self.p, self.q = X.shape[0], X.shape[1], Y.shape[1]
        self.S, self.Sx, self.Sy = X.T.dot(Y), X.T.dot(X), Y.T.dot(Y)
        self.U,  self.d, self.V = np.linalg.svd(self.S)
        wx_init = self.U[:, 0]
        wy_init =  self.V.T[:, 0]
        self.wx_init = wx_init / np.sqrt(np.sum(wx_init**2))
        self.wy_init = wy_init / np.sqrt(np.sum(wy_init**2))
        
    def _fit_symdef_d(self, lambda_x, lambda_y, n_comps=None, X=None, Y=None, wx_init=None,
             wy_init=None, pdout=False, cca_kws={}):
        if n_comps is None:
            n_comps = np.minimum(self.p, self.q)
        if X is None:
            X = self.X.copy()
        if Y is None:
            Y = self.Y.copy()
        if wx_init is None:
            wx_init = self.U[:, 0]
            wx_init = wx_init / np.sqrt(np.sum(wx_init**2))
        if wy_init is None:
            wy_init =  self.V.T[:, 0]
            wy_init = wy_init / np.sqrt(np.sum(wy_init**2))
        cx, cy = lambda_x * np.sqrt(self.p), lambda_y * np.sqrt(self.q)
        Wx, Wy, optinfo = _scca(X, Y, wx_init, wy_init, cx, cy, n_comps, 
                                          **cca_kws)
        
        if pdout:
            if self.x_is_pd:
                Wx = pd.DataFrame(Wx, index=['V%i'%i for i in range(1, n_comps+1)],
                                             columns=self.xcols)
            if self.y_is_pd:
                Wy = pd.DataFrame(Wy, index=['V%i'%i for i in range(1, n_comps+1)],
                                             columns=self.ycols)

        return Wx.T, Wy.T, optinfo
    
    def _fit(self, lambda_x, lambda_y, n_comps=None, X=None, Y=None, wx_init=None,
             wy_init=None, pdout=False, ortho=True, cca_kws={}):
        if X is None:
            X = self.X.copy()
        if Y is None:
            Y = self.Y.copy()
        Wx, Wy, optinfo = self._fit_symdef_d(lambda_x, lambda_y, n_comps, X,
                                             Y, wx_init, wy_init, pdout
                                             , cca_kws)
        if ortho:
            Wx, Wy = self.orthogonalize_components(Wx, Wy, X=X, Y=Y)
        rho = vech(_corr(X.dot(Wx), Y.dot(Wy)))
        return Wx, Wy, rho, optinfo
    
    def orthogonalize_components(self, Wx, Wy, X=None, Y=None):
        if X is None:
            X = self.X.copy()
        if Y is None:
            Y = self.Y.copy()
        Qx = np.linalg.inv(X.T.dot(X)).dot(X.T)
        Qy = np.linalg.inv(Y.T.dot(Y)).dot(Y.T)
        Xt, Yt = X.copy(), Y.copy()
        t = Wx.shape[1]
        Bx, By = np.zeros_like(Wx), np.zeros_like(Wy)
        for i in range(t):
            wxi, wyi = Wx[:, i], Wy[:, i]
            vx, vy = Xt.dot(wxi), Yt.dot(wyi)
            wxi = Qx.dot(Xt).dot(wxi)
            wyi = Qy.dot(Yt).dot(wyi)
            wxi[np.abs(wxi)<1e-12] = 0.0
            wyi[np.abs(wyi)<1e-12] = 0.0
            Bx[:, i], By[:, i] = wxi, wyi
            uX, uY = vx.T.dot(Xt), vy.T.dot(Yt)
            Xt = Xt - (np.outer(vx, uX) / np.sum(vx**2))
            Yt = Yt - (np.outer(vy, uY) / np.sum(vy**2))
        return Bx, By  
       
        
    def crossval(self, ncv, n_comps=None, lambdas=None, lbounds=None,
                 X=None, Y=None, ortho=True, cca_kws={}):
        
        if lbounds is None:
            lbounds = 0.1, 0.8
            
        if type(lambdas) in [int, float]:
            lambdas = np.linspace(lbounds[0], lbounds[1], int(lambdas))
        elif lambdas is None:
            lambdas = np.linspace(lbounds[0], lbounds[1], 20)
        
        if X is None:
            X = self.X.copy()
        
        if Y is None:
            Y = self.Y.copy()
        
        if n_comps is None:
            n_comps = 1
        
        nl = len(lambdas)
        Xf, Yf, Xt, Yt = crossval_mats(X, Y, X.shape[0], ncv)
        progress_bar = tqdm.tqdm(total=len(lambdas)*int(ncv))
        Wx = np.zeros((nl, ncv, self.p, n_comps))
        Wy = np.zeros((nl, ncv, self.q, n_comps))
        rf = np.zeros((nl, ncv, int(n_comps * (n_comps + 1) / 2)))
        rt = np.zeros((nl, ncv, int(n_comps * (n_comps + 1) / 2)))
        optinfo = np.zeros((nl, ncv, n_comps, 2))
        for i, a in enumerate(lambdas):
            for j in range(ncv):
                wx, wy, r, opt = self._fit(a, a, n_comps, Xf[j], Yf[j], ortho=ortho,
                                        cca_kws=cca_kws)
                optinfo[i, j] = opt
                Wx[i, j] , Wy[i, j]  =  wx, wy
                rf[i, j], rt[i, j] = r, vech(_corr(Xt[j].dot(wx), Yt[j].dot(wy))) 
                progress_bar.update(1)
        progress_bar.close()
        return Wx, Wy, rf, rt, lambdas, optinfo
    
    def permutation_test(self, lambda_x, lambda_y, n_perms=1500, n_comps=None, 
                         ortho=True, cca_kws={}):
        if n_comps is None:
            n_comps = 1
        
        progress_bar = tqdm.tqdm(total=n_perms)
        res = np.zeros((n_perms, int(n_comps*(n_comps+1)/2)))
        wx = np.random.normal(0.0, 1.0, size=self.p)
        wy = np.random.normal(0.0, 1.0, size=self.q)
        wx = wx /  np.sqrt(np.sum(wx**2))
        wy = wy / np.sqrt(np.sum(wx**2))
        for i in range(n_perms):
            ixx = np.random.permutation(self.n)
            ixy = np.random.permutation(self.n)
            X_p, Y_p = self.X[ixx], self.Y[ixy]
            wx, wy, r, _ = self._fit(lambda_x, lambda_y, n_comps, 
                                     X_p, Y_p, wx, wy, pdout=False, ortho=ortho,
                                     cca_kws=cca_kws)
            res[i] = r
            
            wx = np.random.normal(0.0, 1.0, size=self.p)
            wy = np.random.normal(0.0, 1.0, size=self.q)
            wx = wx /  np.sqrt(np.sum(wx**2))
            wy = wy / np.sqrt(np.sum(wx**2))
               
           
            progress_bar.update(1)
        progress_bar.close()
        return pd.DataFrame(res)
            
    def bootstrap(self, lambda_x, lambda_y, n_boot=1500, n_comps=None, 
                  ortho=True, cca_kws={}):
        if n_comps is None:
            n_comps = 1
        
        progress_bar = tqdm.tqdm(total=n_boot)
        res = np.zeros((n_boot, int(n_comps*(n_comps+1)/2)))
        Wx = np.zeros((n_boot, self.p, n_comps))
        Wy = np.zeros((n_boot, self.q, n_comps))
        for i in range(n_boot):
            ix = np.random.choice(self.n, self.n, replace=True)
            X_p, Y_p = self.X[ix], self.Y[ix]
            Wx[i], Wy[i], r, _ = self._fit(lambda_x, lambda_y, n_comps, 
                                                    X_p, Y_p, ortho=ortho,
                                                    cca_kws=cca_kws)
            res[i] = r
            progress_bar.update(1)
        progress_bar.close()
        return Wx, Wy, pd.DataFrame(res)
    
    def crossval_mc(self, lambda_x, lambda_y, n_samples=1500, n_cv=7, 
                    n_comps=None, cca_kws={}):
        if n_comps is None:
            n_comps = 1
        progress_bar = tqdm.tqdm(total=n_samples*n_cv)
        res = np.zeros((n_samples, n_cv, int(n_comps*(n_comps+1)/2)))
        Wx = np.zeros((n_samples, n_cv, self.p, n_comps))
        Wy = np.zeros((n_samples, n_cv, self.q, n_comps))
        for i in range(n_samples):
            ix = np.random.choice(self.n, self.n, replace=True)
            X_p, Y_p = self.X[ix], self.Y[ix]
            Xf, Yf, Xt, Yt = crossval_mats(X_p, Y_p, self.X.shape[0], n_cv)
            for j in range(n_cv):
                Wx[i, j], Wy[i, j], r, _ = self._fit(lambda_x, lambda_y, n_comps, 
                                                     Xf[j], Yf[j], cca_kws=cca_kws)
                progress_bar.update(1)
                res[i, j] = vech(_corr(Xt[j].dot(Wx[i, j]), Yt[j].dot(Wy[i, j])))
           
        progress_bar.close()
        return Wx, Wy, res
    
    def fit(self, ncv, n_comps=None, cv_kws=None, fit_kws=None):
        cv_kws = {} if cv_kws is None else cv_kws
        fit_kws = {} if fit_kws is None else fit_kws

        self.Wxcv, self.Wycv, self.rf, self.rt, self.lambdas, optinfo = self.crossval(ncv,
                                                                                      n_comps,
                                                                                      **cv_kws)
        self.lmax = np.argmax(self.rt.mean(axis=1)[:, [0, 2]].mean(axis=1))
        self.lam = self.lambdas[self.lmax]
        self.Wx, self.Wy, self.rho, self.optinfo = self._fit(self.lam, self.lam,
                                                             n_comps,
                                                             **fit_kws)
        
