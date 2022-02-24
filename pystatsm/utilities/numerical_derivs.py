#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 14:40:17 2020

@author: lukepinkel
"""
import numba
import numpy as np
import scipy as sp
import scipy.special

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
          

def finite_diff(y, x, n=1, m=2):
    num_x = len(x)
    du = np.zeros_like(y)
    mm = n // 2 + m
    size = 2 * mm + 2  
    x1, x2 = x[:size], x[-size:]
    y1, y2 = y[:size], y[-size:]
    for i in range(mm):
        du[i] = np.dot(fd_coefs(x1, x0=x[i], n=n),y1)
        du[-i - 1] = np.dot(fd_coefs(x2, x0=x[-i - 1], n=n), y2)
    for i in range(mm, num_x - mm):
        j, k = i - mm, i + mm + 1
        du[i] = np.dot(fd_coefs(x[j:k], x0=x[i], n=n), y[j:k])
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


def jac_approx(f, x, eps=1e-4, tol=None, d=1e-4, nr=6, v=2):
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



def _hess_approx(f, x, a_eps=1e-4, r_eps=1e-4, xtol=None, nr=6, s=2):
    xtol = np.finfo(np.float).eps**(1/3) if xtol is None else xtol
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
   
 
def hess_approx(f, x, a_eps=1e-4, r_eps=1e-4, xtol=None, nr=6, s=2):
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
    

 #def fd_coefficients(points, order):
 #    A = np.zeros((len(points), len(points)))
 #    A[0] = 1
 #    for i in range(len(points)):
 #        A[i] = np.asarray(points)**(i)
 #    b = np.zeros(len(points))
 #    b[order] = sp.special.factorial(order)
 #    c = np.linalg.inv(A).dot(b)
 #    return c
         
     
 #def finite_diff(f, x, epsilon=None, order=1, points=None):
 #    if points is None:
 #        points = np.arange(-4, 5)
 #    if epsilon is None:
 #        epsilon = (np.finfo(float).eps)**(1./3.)
 #    coefs = fd_coefficients(points, order)
 #    df = 0.0
 #    for c, p in list(zip(coefs, points)):
 #        df+=c*f(x+epsilon*p)
 #    df = df / (epsilon**order)
 #    return df
    