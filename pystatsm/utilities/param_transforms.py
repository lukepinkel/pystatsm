# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 23:32:16 2023

@author: lukepinkel
"""
import numpy as np
import scipy as sp
from . import indexing_utils

class OrderedTransform(object):
    
    @staticmethod
    def _rvs(y, q):
        x = y.copy()
        x[1:q] = np.cumsum(np.exp(y[1:q]))+x[0]
        return x

    @staticmethod
    def _fwd(x, q):
        y = x.copy()
        y[1:q] = np.log(np.diff(x[:q]))
        return y
    
    
    @staticmethod
    def _jac_rvs(y, q):
        dx_dy = np.zeros((q, q))
        inds = indexing_utils.tril_indices(q)
        vals = np.repeat(np.r_[1, np.exp(y[1:q])], np.arange(q, 0, -1))
        dx_dy[inds] = vals
        # dx_dy[:, 0] = 1.0
        # for i in range(1, q):
        #     dx_dy[i:, i] = np.exp(y[i])
        return dx_dy

    @staticmethod
    def _hess_rvs(y, q):
        d2x_dy2 = np.zeros((q, q, q))
        z = np.exp(y)
        for i in range(1, q):
            for j in range(1, i+1):
                d2x_dy2[i, j, j] = z[j]
        return d2x_dy2
    

    @staticmethod
    def _jac_fwd(x, q):
        dy_dx = np.zeros((q, q))
        dy_dx[0, 0] = 1
        ii = np.arange(1, q)
        dy_dx[ii, ii] = 1 / (x[ii] - x[ii-1])
        dy_dx[ii, ii-1] =  -1 / (x[ii] - x[ii - 1])
        # for i in range(1, q):
        #     dy_dx[i, i] = 1 / (x[i] - x[i - 1])
        #     dy_dx[i, i - 1] = -1 / (x[i] - x[i - 1])
        return dy_dx
    
    @staticmethod
    def _hess_fwd(x, q):
        d2y_dx2 = np.zeros((q, q, q))
        for i in range(1, q):
            d2y_dx2[i, i, i] = -1 / (x[i] - x[i - 1])**2
            d2y_dx2[i, i - 1, i - 1] = -1 / (x[i] - x[i - 1])**2
            d2y_dx2[i, i, i - 1] = 1 / (x[i] - x[i - 1])**2
            d2y_dx2[i, i - 1, i] = d2y_dx2[i, i, i - 1]  # Use symmetry
        return d2y_dx2



