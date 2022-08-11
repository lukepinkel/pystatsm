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
    """

    Parameters
    ----------
    n : int
        Size of matrix for which to return lower half vectorization indices.

    Returns
    -------
    ix : array of boolean
        Boolean indicator array to select elements from a vectorized matrix
        corresponding to the lower half.

    """
    i, j = np.indices((n, n))
    i, j = i.flatten(), j.flatten()
    ix = j>i
    return ix

def tril_indices(n, k=0, m=None, order='F'):
    """

    Parameters
    ----------
    n : int
        number of rows in array
    k : int, optional
        Diagonal offset, negative for below and positive for above. The default is 0.
    m : int, optional
        number of columns in array. The default is None, taken to be n
    order : str, optional
        Whether to return fortran or C ordered indices. The default is 'F'.

    Returns
    -------
    inds : tuple of arrays
        lower triangular indices.

    """
    if order == 'F':
        k = int(k * -1)
        inds = np.triu_indices(n=n, k=k, m=m)[::-1]
    elif order == 'C':
        inds = np.tril_indices(n=n, k=k, m=m)
    return inds

def triu_indices(n, k=0, m=None, order='F'):
    """

    Parameters
    ----------
    n : int
        number of rows in array
    k : int, optional
        Diagonal offset, negative for below and positive for above. The default is 0.
    m : int, optional
        number of columns in array. The default is None, taken to be n
    order : str, optional
        Whether to return fortran or C ordered indices. The default is 'F'.

    Returns
    -------
    inds : tuple of arrays
        upper triangular indices.

    """
    if order == 'F':
        k = int(k * -1)
        inds = np.tril_indices(n=n, k=k, m=m)[::-1]
    elif order == 'C':
        inds = np.triu_indices(n=n, k=k, m=m)
    return inds
 
   
    


 