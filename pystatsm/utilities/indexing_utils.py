#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 01:22:10 2022

@author: lukepinkel
"""

import numpy as np
import scipy as sp

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

def inv_tril_indices(n, k=0, m=None, order='F', indexing=0):
    arr = np.zeros((n, n), dtype=int)
    ij = tril_indices(n, k, m, order)
    arr[ij] = np.arange(indexing, indexing+len(ij[0]))
    arr = arr + arr.T
    return arr

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
 
   
def flat_mgrid(*xi, **kwargs):
    g = np.meshgrid(*xi, **kwargs)
    f = [i.flatten() for i in g]
    return f


class ndindex:

    def __init__(self, *shape, order='F'):
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        x = np.lib.stride_tricks.as_strided(np.core.numeric.zeros(1), 
                                            shape=shape,
                                            strides=np.core.numeric.zeros_like(shape))
        self._it = np.core.numeric.nditer(x, flags=['multi_index', 'zerosize_ok'],
                              order=order)

    def __iter__(self):
        return self


    def __next__(self):
        next(self._it)
        return self._it.multi_index
    
    

def get_lower_indices(*args):
    res = 0
    for i, x in enumerate(args):
        i1 = i + 1
        den = int(sp.special.factorial(i1))
        res = res + int(np.product([x + k for k in range(i1)], dtype=int) / den)
    return res

def rmq(x, n):
    return x % n, x // n

