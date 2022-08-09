#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 01:22:10 2022

@author: lukepinkel
"""

import numpy as np

def diag_indices(n, k=0):
    """

    Parameters
    ----------
    n : int
        size of array.
    k : int, optional
        diagonal offset, k>0 is above diagonal, k<0 is below. The default is 0.

    Returns
    -------
    r : array
        row indices.
    c : array
        column indices.

    """
    r, c = np.diag_indices(n)
    if k < 0:
        r, c = r[-k:], c[:k]
    elif k > 0:
        r, c = r[:-k], c[k:]
    return r, c

def vecl_inds(n):
    i, j = np.indices((n, n))
    i, j = i.flatten(), j.flatten()
    ix = j>i
    return ix

def tril_indices(n, k=0, m=None, order='F'):
    if order == 'F':
        inds = np.triu_indices(n=n, k=k, m=m)[::-1]
    elif order == 'C':
        inds = np.tril_indices(n=n, k=k, m=m)
    return inds

def triu_indices(n, k=0, m=None, order='F'):
    if order == 'F':
        inds = np.tril_indices(n=n, k=k, m=m)[::-1]
    elif order == 'C':
        inds = np.triu_indices(n=n, k=k, m=m)
    return inds
    
    


 