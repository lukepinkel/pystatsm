#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  2 03:48:59 2020

@author: lukepinkel
"""

import scipy as sp           # analysis:ignore
import numpy as np           # analysis:ignore


def difference_mat(k, order=2):
    Dk = np.diff(np.eye(k), order, axis=0)
    return Dk
    

def equispaced_knots(x, degree, ndx):
    xl = np.min(x)
    xr = np.max(x)
    dx = (xr - xl) / ndx
    order = degree + 1
    lb = xl - order * dx
    ub = xr + order * dx
    knots = np.arange(xl - order * dx, xr + order * dx, dx)
    return knots, order, lb, ub

def bspline(x, knots, degree, deriv=0):
    if len(knots)<=(degree+1):
        raise ValueError("Number of knots must be greater than order")
    order = degree + 1
    q = len(knots) - order
    u = np.zeros(q)
    B = np.zeros((len(x), q))
    for i in range(q):
        u[i] = 1
        tck = (knots, u, degree)
        B[:, i] = sp.interpolate.splev(x, tck, der=deriv)
        u[i] = 0
    return B[:, 1:]

def bspline_des(x, degree=3, ndx=20, deriv=0):
    knots, order, _, _ = equispaced_knots(x, degree, ndx)
    B = bspline(x, knots, degree, deriv)
    return B