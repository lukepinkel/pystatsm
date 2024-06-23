#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 14:11:35 2024

@author: lukepinkel
"""
import pandas as pd
import numpy as np
import scipy as sp
from ..utilities.cs_kron import coo_to_csc, sparse_kron, sparse_dense_kron


def make_zinds(n_ob, n_rv, cols):
    z_rows = np.repeat(np.arange(n_ob), n_rv)
    z_cols = np.repeat(cols * n_rv, n_rv)  + np.tile(np.arange(n_rv), n_ob)
    return z_rows, z_cols

def make_ginds_bdiag(n_rv, n_lv):
    gdim = n_rv * n_lv
    g_cols = np.repeat(np.arange(gdim), n_rv)
    g_rows = np.repeat(np.arange(n_lv)*n_rv, n_rv * n_rv) + np.tile(np.arange(n_rv), gdim)
    return g_rows, g_cols

def make_remod_mat(arr1, arr2, return_array=False):
    cols, u = pd.factorize(arr1, sort=False)
    n_lv = len(u)
    n_ob, n_rv = arr2.shape
    z_rows, z_cols = make_zinds(n_ob, n_rv, cols)
    z_data = arr2.reshape(-1, order='C')
    z_size = (n_ob, n_lv * n_rv)
    Z = coo_to_csc(z_rows, z_cols, z_data, z_size, return_array=True)
    return Z

def make_recov_mat(n_rv, n_lv, a_cov=None):
    if a_cov is None:
        a_cov = sp.sparse.eye(n_lv, format="csc") 
        bdiag = True
    else:
        bdiag = False
    G0 = np.eye(n_rv)
    g_data = np.tile(G0.reshape(-1, order='F'), n_lv)
    if bdiag:
        g_rows, g_cols = make_ginds_bdiag(n_rv, n_lv)
        G = coo_to_csc(g_rows, g_cols, g_data, (n_rv*n_lv, n_rv*n_lv), return_array=True)
    else:
        G = sparse_dense_kron(a_cov, G0)
    return G, bdiag
    
    