#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 23:09:48 2023

@author: lukepinkel
"""

import numba
import numpy as np

@numba.jit(nopython=True)
def vecl_inds_forwards(r, s, m):
    """
    Returns the index of the (r, s)-th element in the lower triangle of a
    square matrix of size (m x m) that has been half vectorized, excluding
    the diagonal.
    
    Parameters
    ----------
    r : int
        The row index of the element in the original matrix.
    s : int
        The column index of the element in the original matrix.
    m : int
        The number of rows (and columns) of the square matrix.
        
    Returns
    -------
    int
        The index in the half vectorized form corresponding to the
        (r, s)-th element in the original matrix, excluding the diagonal elements.
    """
    # Index i is computed by summing the total number of elements in the matrix
    #above the current row (s * m), subtracting the triangular number 
    #associated with the current column (s * (s + 1) // 2), adding the current
    #row index (r), and subtracting s to account for the exclusion of the 
    #diagonal elements
    i = r + s * m - (s * (s + 1) // 2) - s  -1
    return i


@numba.jit(nopython=True)
def lhv_size_to_mat_size(lhv_size):
    mat_size = int((np.sqrt(8 * lhv_size + 1) + 1) // 2)
    return mat_size


@numba.jit(nopython=True)
def rvs(x):
    n = lhv_size_to_mat_size(len(x))
    y = np.zeros_like(x)
    for i in range(1, n):
        k = vecl_inds_forwards(i, 0, n)
        y[k] = x[k]
        s = x[k]**2
        for j in range(1, i):
            k = vecl_inds_forwards(i, j, n)
            t = 1.0 - s
            u = np.sqrt(t)
            y[k] = x[k] * u
            v = y[k]**2
            s = s + v
    return y


@numba.jit(nopython=True)
def jac_rvs(x):
    m = len(x)
    n = lhv_size_to_mat_size(m)
    J = np.zeros((m, m))
    ds_dx = np.zeros(m)
    y = rvs(x)
    for i in range(1, n):
        k1 = vecl_inds_forwards(i, 0, n)
        J[k1, k1] = 1.0
        s = x[k1] ** 2
        ds_dx = ds_dx * 0.0
        ds_dx[k1] = 2 * x[k1]
        for j in range(1, i):
            k2 = vecl_inds_forwards(i, j, n)
            t = 1 - s
            u = np.sqrt(t)
            dt_dx = - ds_dx
            du_dx = 0.5 * (1 / u) * dt_dx
            J[k2] = x[k2] * du_dx
            J[k2, k2] = J[k2, k2] + u
            v = y[k2] ** 2
            dv_dx = 2 * y[k2] * J[k2]
            s = s + v
            ds_dx = ds_dx + dv_dx
    return J


@numba.jit(nopython=True)
def hess_rvs(x):
    m = len(x)
    n = lhv_size_to_mat_size(m)
    J = np.zeros((m, m))
    H = np.zeros((m, m, m))
    y = rvs(x)
    ds_dx = np.zeros(m)
    d2s_dx2 = np.zeros((m, m))
    for i in range(1, n):
        k1 = vecl_inds_forwards(i, 0, n)
        J[k1, k1] = 1
        s = x[k1] ** 2
        ds_dx = ds_dx * 0.0
        d2s_dx2 = d2s_dx2 * 0.0
        ds_dx[k1] = 2 * x[k1]
        d2s_dx2[k1, k1] = 2.0
        for j in range(1, i):
            k2 = vecl_inds_forwards(i, j, n)
            t = 1 - s
            u = np.sqrt(t)
            w = 1 / u
            du_dt = 1 / (2.0  * u)
            dw_du = (-1.0 / u**2)
            dw_dt = dw_du * du_dt
            dt_dx = - ds_dx
            dw_dx = dw_dt * dt_dx
            d2t_dx2 = - d2s_dx2
            du_dx = 0.5 * w * dt_dx
            d2u_dx2 = 0.5 * np.outer(dw_dx, dt_dx) + 0.5 * w * d2t_dx2
            J[k2] = x[k2] * du_dx
            J[k2, k2] = J[k2, k2] + u
            
            H[k2] =  x[k2] * d2u_dx2
            H[k2, k2] = H[k2, k2] + du_dx
            H[k2, :, k2] = H[k2, :, k2] + du_dx
            H[k2, k2, k2] = H[k2, k2, k2] + du_dx[k2] * 2.0
            v = y[k2] ** 2
            dv_dx = 2 * y[k2] * J[k2]
            d2v_dx2 = 2 * np.outer(J[k2], J[k2]) + 2 * y[k2] * H[k2]
            s = s+v
            ds_dx = ds_dx + dv_dx
            d2s_dx2 = d2s_dx2 + d2v_dx2
    return H

@numba.jit(nopython=True)
def fwd(y):
    n = lhv_size_to_mat_size(len(y))
    x = np.zeros_like(y)
    for i in range(1, n):
        k = vecl_inds_forwards(i, 0, n)
        x[k] = y[k]
        s = y[k]**2
        for j in range(1, i):
            k = vecl_inds_forwards(i, j, n)
            x[k] = y[k] / np.sqrt(1.0 - s)
            s = s + y[k]**2
    return x

@numba.jit(nopython=True)
def jac_fwd(y):
    m = len(y)
    n = lhv_size_to_mat_size(m)
    J = np.zeros((m, m))
    ds_dy = np.zeros(m)
    for i in range(1, n):
        k1 = vecl_inds_forwards(i, 0, n)
        J[k1, k1] = 1
        s = y[k1] ** 2
        ds_dy = ds_dy * 0.0
        ds_dy[k1] = 2 * y[k1]
        for j in range(1, i):
            k2 = vecl_inds_forwards(i, j, n)
            t = 1 - s
            u = np.sqrt(t)
            dt_dy = -ds_dy
            du_dy = 0.5 * (1 / u) * dt_dy
            J[k2] = - y[k2] * du_dy / u**2 
            J[k2, k2] = J[k2, k2] + 1 / u
            v = y[k2] ** 2
            s = s + v
            ds_dy[k2] = ds_dy[k2] + 2 * y[k2]
    return J

@numba.jit(nopython=True)
def hess_fwd(y):
    m = len(y)
    n = lhv_size_to_mat_size(m)
    J = np.zeros((m, m))
    H = np.zeros((m, m, m))
    for i in range(1, n):
        k1 = vecl_inds_forwards(i, 0, n)
        J[k1, k1] = 1
        s = y[k1] ** 2
        ds_dy = np.zeros(m)
        d2s_dy2 = np.zeros((m, m))
        
        ds_dy[k1] = 2 * y[k1]
        d2s_dy2[k1, k1] = 2.0
        for j in range(1, i):
            #y--> s --> t --> u --> w 
            #
            k2 = vecl_inds_forwards(i, j, n)
            t = 1 - s               # scalar 
            u = np.sqrt(t)          # scalar
            w = 1 / u               # scalar
            a = w**2                # scalar
            b = -y[k2] * a          # scalar
            
            du_dt = 1 / (2.0  * u)  # scalar
            dt_ds = -1.0            # scalar
            dt_dy = dt_ds * ds_dy   # scalar
            du_dy = du_dt * dt_dy   # vector (m,)
            dw_du = (-1.0 / u**2)   # scalar
            dw_dt = dw_du * du_dt   # scalar
            dw_dy = dw_dt * dt_dy   # vector (m,)
            da_dy = 2 * w * dw_dy   # vector (m,)
            db_dy = -y[k2] * da_dy  # vector (m,)
            db_dy[k2] = db_dy[k2] - a
            
            d2t_ds2 = 0.0 #analysis:ignore #scalar
            d2u_dt2 = -1.0 / (4 * u**3) #scalar
            d2t_dy2 = dt_ds * d2s_dy2 # d2t_ds2 * ds_dy + dt_ds * d2s_dy2 #array (m,m)
            d2u_dy2 = d2u_dt2 * np.outer(dt_dy, dt_dy) + du_dt * d2t_dy2
            
          
            J[k2] = du_dy * b
            J[k2, k2] = J[k2, k2] + w
            
            H[k2] = d2u_dy2 * b + np.outer(du_dy, db_dy)
            H[k2, k2] =  H[k2, k2] + dw_dy
            v = y[k2] ** 2
            s = s + v
            ds_dy[k2] =  ds_dy[k2] +  2 * y[k2] 
            d2s_dy2[k2, k2] = d2s_dy2[k2, k2] + 2.0
    return H
