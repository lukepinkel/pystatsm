#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 14:40:17 2020

@author: lukepinkel
"""
import tqdm
import numba
import numpy as np
import scipy as sp
import scipy.special
from .indexing_utils import ndindex

def fo_fc_fd(f, x, eps=None, args=()):
    if eps is None:
        eps = (np.finfo(float).eps)**(1.0/3.0)
    n = len(np.asarray(x))
    g, h = np.zeros(n), np.zeros(n)
    for i in range(n):
        h[i] = eps
        g[i] = (f(x+h, *args) - f(x, *args)) / eps
        h[i] = 0
    return g


def so_fc_fd(f, x, eps=None, args=()):
    if eps is None:
        eps = (np.finfo(float).eps)**(1.0/3.0)
    n = len(np.asarray(x))
    H, hi, hj = np.zeros((n, n)), np.zeros(n), np.zeros(n)
    eps2 = eps**2
    for i in range(n):
        hi[i] = eps
        for j in range(i+1):
            hj[j] = eps
            H[i, j] = (f(x+hi+hj, *args) - f(x+hi, *args) - f(x+hj, *args) + f(x, *args)) / eps2
            H[j, i] = H[i, j]
            hj[j] = 0  
        hi[i] = 0
    return H


def so_gc_fd(g, x, eps=None, args=()):
    if eps is None:
        eps = (np.finfo(float).eps)**(1.0/3.0)
    n = len(np.asarray(x))
    H, h = np.zeros((n, n)), np.zeros(n)
    gx, gxh = np.zeros((n, n)), np.zeros((n, n))
    for i in range(n):
        h[i] = eps
        gx[i] = g(x, *args)
        gxh[i] = g(x+h, *args)
        h[i] = 0
    for i in range(n):
        for j in range(i+1):
            H[i, j] = ((gxh[i, j] - gx[i, j]) + (gxh[j, i] - gx[j, i])) / (2 * eps)
            H[j, i] = H[i, j]
    return H


def fo_fc_cd(f, x, eps=None, args=()):
    if eps is None:
        eps = (np.finfo(float).eps)**(1.0/3.0)
    n = len(np.asarray(x))
    g, h = np.zeros(n), np.zeros(n)
    for i in range(n):
        h[i] = eps
        g[i] = (f(x+h, *args) - f(x - h, *args)) / (2 * eps)
        h[i] = 0
    return g


def so_fc_cd(f, x, eps=None, args=()):
    p = len(np.asarray(x))
    if eps is None:
        eps = (np.finfo(float).eps)**(1./3.)
    H = np.zeros((p, p))
    ei = np.zeros(p)
    ej = np.zeros(p)
    for i in range(p):
        for j in range(i+1):
            ei[i], ej[j] = eps, eps
            if i==j:
                dn = -f(x+2*ei, *args)+16*f(x+ei, *args)\
                    -30*f(x, *args)+16*f(x-ei, *args)-f(x-2*ei, *args)
                nm = 12*eps**2
                H[i, j] = dn/nm  
            else:
                dn = f(x+ei+ej, *args)-f(x+ei-ej, *args)-f(x-ei+ej, *args)+f(x-ei-ej, *args)
                nm = 4*eps*eps
                H[i, j] = dn/nm  
                H[j, i] = dn/nm  
            ei[i], ej[j] = 0.0, 0.0
    return H
        

def so_gc_cd(g, x, eps=None, args=()):
    if eps is None:
        eps = (np.finfo(float).eps)**(1./3.)
    n = len(np.asarray(x))
    H, h = np.zeros((n, n)), np.zeros(n)
    gxp, gxn = np.zeros((n, n)), np.zeros((n, n))
    for i in range(n):
        h[i] = eps
        gxp[i] = g(x+h, *args)
        gxn[i] = g(x-h, *args)
        h[i] = 0
    for i in range(n):
        for j in range(i+1):
            H[i, j] = ((gxp[i, j] - gxn[i, j] + gxp[j, i] - gxn[j, i])) / (4 * eps)
            H[j, i] = H[i, j]
    return H


@numba.jit(nopython=True)
def _fd_coefs(W, x, x0=0, n=1):
    m = len(x)
    c1, c4 = 1.0, x[0] - x0
    W[0,0] = 1.
    for i in range(1, m):
        r = min(i, n)
        c2, c5, c4 = 1.0, c4, x[i] - x0
        for j in range(i):
            c3 = x[i] - x[j]
            c2 = c2 * c3
            if j==i-1:
                for k in range(r, 0, -1):
                    W[i, k] = c1 * (k * W[i-1, k-1] - c5 * W[i-1, k]) / c2
                W[i, 0] = -c1 * c5 * W[i-1, 0] / c2
            for k in range(r, 0, -1):
                W[j, k] = (c4 * W[j, k] - k * W[j, k-1]) / c3
            W[j, 0] = c4 * W[j, 0] / c3
        c1 = c2
    return W

  
def fd_coefs(x, x0=0, n=1, last_col=True):
    m = len(x)
    W = np.zeros((m, n + 1))
    W = _fd_coefs(W, x, x0, n)
    res = W[: ,-1] if last_col else W   
    return res
          

# def _finite_diff(y, x, n=1, m=2):
#     num_x = len(x)
#     du = np.zeros_like(y)
#     mm = n // 2 + m
#     size = 2 * mm + 2  
#     x1, x2 = x[:size], x[-size:]
#     y1, y2 = y[:size], y[-size:]
#     for i in range(mm):
#         du[i] = np.dot(fd_coefs(x1, x0=x[i], n=n),y1)
#         du[-i - 1] = np.dot(fd_coefs(x2, x0=x[-i - 1], n=n), y2)
#     for i in range(mm, num_x - mm):
#         j, k = i - mm, i + mm + 1
#         du[i] = np.dot(fd_coefs(x[j:k], x0=x[i], n=n), y[j:k])
#     return du



def finite_diff(f, x, order=1, m=6, n=16):
    du = np.zeros_like(f(x))
    stencil_points = np.linspace(-n, n, 2*m+2*order+1)
    h = 1 / n
    coefs = fd_coefs(stencil_points, x0=0, n=order)
    for i in range(len(stencil_points)):
        du += coefs[i]*f(x + h * stencil_points[i]) / h**order
    return du



def grad_approx(f, x, eps=1e-4, tol=None, d=1e-4, nr=6, v=2):
    tol = np.finfo(float).eps**(1/3) if tol is None else tol
    h = np.abs(d * x) + eps * (np.abs(x) < tol)
    n = len(x)
    u = np.zeros_like(h)
    A = np.zeros((nr, n))
    for i in range(nr):
        for j in range(n):
            u[j] = h[j]
            A[i, j] = (f(x + u) - f(x - u)) / (2.0 * h[j])
            u[j] = 0.0
        h /= v
    for i in range(nr-1):
        t = 4**(i+1)
        A = (A[1:(nr-i)]*t - A[:(nr-i-1)]) / (t-1.0)
    return A




def jac_approx(F, X, eps=1e-4, tol=None, d=1e-4, args=(), progress_bar=False, mask=None):
    def Func(X, args=()):
        return np.atleast_1d(F(X, *args))
    X = np.asarray(X)
    Y = np.asarray(Func(X, *args))
    tol = np.finfo(float).eps**(1/3) if tol is None else tol
    H = np.zeros_like(X)
    J = np.zeros(Y.shape+X.shape)
    if progress_bar:
        pbar = tqdm.tqdm(total=np.prod(Y.shape)*np.prod(X.shape))
        
    if mask is None:
        use_mask = False  # If mask is not passed, don't use it
    else:
        use_mask = True  # If mask is passed, use it

    for ii in np.ndindex(Y.shape):
        for jj in np.ndindex(X.shape):
            if not use_mask or mask[ii]:
                H[jj] = np.abs(d * X[jj]) + eps * (np.abs(X[jj]) < tol)
                FpH = Func(X+H, *args)[ii]
                FmH = Func(X-H, *args)[ii]
                J[ii+jj] = (FpH - FmH) / (2.0 * H[jj])
                H[jj] = 0.0
            if progress_bar:
                pbar.update(1)
    if progress_bar:
        pbar.close()
    return J

def jac_approx2(f, x, eps=1e-4, tol=None, d=1e-4, nr=6, v=2):
    tol = np.finfo(float).eps**(1/3) if tol is None else tol
    h = np.abs(d * x) + eps * (np.abs(x) < tol)
    n = len(x)
    p = len(f(x))
    u = np.zeros_like(h)
    A = np.zeros((nr, n, p))
    for i in range(nr):
        for j in range(n):
            u[j] = h[j]
            A[i, j] = (f(x + u) - f(x - u)) / (2.0 * h[j])
            u[j] = 0.0
        h /= v
    for i in range(nr-1):
        t = 4**(i+1)
        A = (A[1:(nr-i)]*t - A[:(nr-i-1)]) / (t-1.0)
    return A

def jac_approx3(F, X, eps=1e-4, tol=None, d=1e-4, args=(), progress_bar=False):
    def Func(X, args=()):
        return np.atleast_1d(F(X, *args))
    
    X = np.asarray(X)
    Y = np.asarray(Func(X, *args))
    
    tol = np.finfo(float).eps**(1/3) if tol is None else tol
    H = np.zeros_like(X)
    J = np.zeros(Y.shape+X.shape)
    
    if progress_bar:
        pbar = tqdm.tqdm(total=np.prod(X.shape))
        
    for jj in np.ndindex(X.shape):
        H[jj] = (np.abs(d * X[jj]) + eps * (np.abs(X[jj]) < tol))
        J[(Ellipsis,)+jj] =  Func(X+H * 1j, *args).imag / H[jj]
        H[jj] = 0.0
        if progress_bar:
            pbar.update(1)
    if progress_bar:
        pbar.close()
    return J


def jac_approx4(F, X, eps=1e-4, tol=None, d=1e-4, args=(), progress_bar=False):
    def Func(X, args=()):
        return np.atleast_1d(F(X, *args))
    X = np.asarray(X)
    Y = np.asarray(Func(X, *args))
    tol = np.finfo(float).eps**(1/3) if tol is None else tol
    H = np.zeros_like(X)
    J = np.zeros(Y.shape+X.shape)
    if progress_bar:
        pbar = tqdm.tqdm(total=np.prod(X.shape))
    for jj in np.ndindex(X.shape):
        H[jj] = np.abs(d * X[jj]) + eps * (np.abs(X[jj]) < tol)
        FpH = Func(X+H, *args)
        FmH = Func(X-H, *args)
        J[(Ellipsis,)+jj] = (FpH - FmH) / (2.0 * H[jj])
        H[jj] = 0.0
        if progress_bar:
            pbar.update(1)
    if progress_bar:
        pbar.close()
    return J


def hess_approx(F, X, eps=1e-4, tol=None, d=1e-4, args=(), progress_bar=False,  mask=None):
    def Func(X, args=()):
        return np.atleast_1d(F(X, *args))
    X = np.asarray(X)
    Y = np.asarray(Func(X, *args))
    tol = np.finfo(float).eps**(1/3) if tol is None else tol
    Hjj, Hkk = np.zeros_like(X), np.zeros_like(X)
    J = np.zeros(Y.shape+X.shape+X.shape)
        
    if mask is None:
        use_mask = False  # If mask is not passed, don't use it
    else:
        use_mask = True  # If mask is passed, use it
        
    if progress_bar:
        if use_mask:
            total = int(np.sum(mask))
        else:
            total = np.prod(Y.shape)*np.prod(X.shape)*np.prod(X.shape)
        pbar = tqdm.tqdm(total=total, smoothing=1e-4)

    
    for ii in ndindex(Y.shape):
        for jj in ndindex(X.shape):
            for kk in ndindex(X.shape):
                if not use_mask or mask[ii+jj+kk]:
                    Hjj[jj] = np.abs(d * X[jj]) + eps * (np.abs(X[jj]) < tol)
                    Hkk[kk] = np.abs(d * X[kk]) + eps * (np.abs(X[kk]) < tol)
                    if jj == kk:
                        t1 =  -1.0 * Func(X+2*Hjj, *args)[ii]
                        t2 =  16.0 * Func(X+1*Hjj, *args)[ii]
                        t3 = -30.0 * Func(X      , *args)[ii] 
                        t4 =  16.0 * Func(X-1*Hjj, *args)[ii]
                        t5 =  -1.0 * Func(X-2*Hjj, *args)[ii]
                        num = t1 + t2 + t3 + t4 + t5
                        J[ii+jj+kk] = num / (12 * Hjj[jj]**2)
                    else:
                        t1 = Func(X+Hjj+Hkk, *args)[ii] + Func(X-Hjj-Hkk, *args)[ii]
                        t2 = Func(X+Hjj-Hkk, *args)[ii] + Func(X-Hjj+Hkk, *args)[ii]
                        num = t1 - t2
                        J[ii+jj+kk] = num / (4 * Hjj[jj] * Hkk[kk])
                    Hjj[jj] = 0.0
                    Hkk[kk] = 0.0
                if progress_bar:
                    pbar.update(1)
    if progress_bar:
        pbar.close()
    return J



def _hess_approx(f, x, a_eps=1e-4, r_eps=1e-4, xtol=None, nr=6, s=2):
    xtol = np.finfo(float).eps**(1/3) if xtol is None else xtol
    d = np.abs(r_eps * x) + a_eps * (np.abs(x) < xtol)
    y = f(x)
    u = np.zeros_like(d)
    v = np.zeros_like(d)
    nx, ny = len(x), len(y)
    D = np.zeros((ny, int(nx * (nx + 3) // 2)))
    Da = np.zeros((ny, nr))
    Hd = np.zeros((ny, nx))
    Ha = np.zeros((ny, nr))
    
    for i in range(nx):
        h = d.copy()
        u[i] = 1.0
        for j in range(nr):
            fp, fn = f(x+h*u), f(x-h*u)
            Da[:, j] = (fp - fn) / (2 * h[i])
            Ha[:, j] = (fp + fn - 2.0 * y) / (h[i]**2)
            h /= s
        for j in range(nr-1):
            for k in range(nr-j-1):
                t = 4**(j+1)
                Da[:, k] = (Da[:, k+1] * t - Da[:, k]) / (t - 1.0)
                Ha[:, k] = (Ha[:, k+1] * t - Ha[:, k]) / (t - 1.0)
        D[:, i] = Da[:, 0]
        Hd[:, i] = Ha[:,0]
        u[i] = 0.0
    
    c = nx-1
    for i in range(nx):
        for j in range(i+1):
            c += 1
            if (i==j):
                D[:, c] = Hd[:, i]
            else:
                h = d.copy()
                u[i] = 1.0
                v[j] = 1.0
                for m in range(nr):
                    fp = f(x + u*h + v*h)
                    fn = f(x - u*h - v*h)
                    fii = Hd[:, i] * h[i]**2
                    fjj = Hd[:, j] * h[j]**2
                    Da[:, m] = (fp - 2.0 * y + fn - fii - fjj) / (2.0 * h[i] * h[j])
                    h /= s
                for m in range(nr-1):
                    for k in range(nr-m-1):
                        t = 4**(m+1)
                        Da[:, k] = (Da[:, k+1]*t - Da[:, k]) / (t - 1.0)
                D[:, c] = Da[:, 0]
                u[i] = v[j] = 0.0
    return D
   
 
def hess_approx2(f, x, a_eps=1e-4, r_eps=1e-4, xtol=None, nr=6, s=2):
    D = _hess_approx(f, x, a_eps, r_eps, xtol, nr, s)
    y = f(x)
    nx, ny = len(x), len(y)
    H = np.zeros((ny, nx, nx))
    k = nx - 1
    for i in range(nx):
        for j in range(i+1):
            k+=1
            H[:, i, j] = H[:, j, i] = D[:, k]
    return H
  

def fd_coefficients(points, order):
    A = np.zeros((len(points), len(points)))
    A[0] = 1
    for i in range(len(points)):
        A[i] = np.asarray(points)**(i)
    b = np.zeros(len(points))
    b[order] = sp.special.factorial(order)
    c = np.linalg.inv(A).dot(b)
    return c
        
    
def fd_derivative(f, x, epsilon=None, order=1, points=None):
    if points is None:
        points = np.arange(-4, 5)
    if epsilon is None:
        epsilon = (np.finfo(float).eps)**(1./3.)
    coefs = fd_coefficients(points, order)
    df = 0.0
    for c, p in list(zip(coefs, points)):
        df+=c*f(x+epsilon*p)
    df = df / (epsilon**order)
    return df
   

def fo_fc_cs(f, x, delta=None, args=()):
    if delta is None:
        delta = (np.finfo(float).eps)**(1.0/3.0)
    n = len(np.asarray(x))
    g, h = np.zeros(n), np.zeros(n)
    for i in range(n):
        h[i] = delta
        g[i] = (f(x+h*1j, *args)).imag / delta
        h[i] = 0
    return g


def jac_cs(f, x, delta=None, args=()):
    if delta is None:
        delta = (np.finfo(float).eps)**(1.0/3.0)
    n, m = len(np.asarray(x)), len(f(x))
    J, h =  np.zeros((m, n)), np.zeros(n)
    for i in range(n):
        h[i] = delta
        J[...,i] = (f(x+h*1j, *args)).imag / delta
        h[i] = 0
    return J


def hess_cs(f, x, delta=None):
    n = len(x)
    if delta is None:
        delta = (np.finfo(float).eps)**(1.0/3.0)
    delta = delta * np.maximum(np.abs(x), 0.1)
    y = f(x)
    if len(np.shape(y))==0:
        H = np.zeros((n, n))
    else:
        m = len(y)
        H = np.zeros((m,n, n))
    I = np.eye(n)
    pbar = tqdm.tqdm(total=(n*(n+1)//2), smoothing=1e-6)
    for j in range(n):
        for i in range(j, n):
            yp = f(x + 1j * I[i] * delta[i] + I[j] * delta[j])
            yn = f(x + 1j * I[i] * delta[i] - I[j] * delta[j])
            H[...,i, j] = H[...,j, i] = (yp - yn).imag / (2 * delta[i] * delta[j])
            pbar.update(1)
    pbar.close()
    return H

