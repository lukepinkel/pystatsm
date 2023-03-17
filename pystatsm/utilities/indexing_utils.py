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
    """

    Parameters
    ----------
    x : int
    n : int
    Returns
    -------
    r : int
        Remainder
    q : int
        Quotient.
        
    Returns (x % n, x // n ) = (r, q) where x=r+qn
    """
    r, q =  x % n, x // n
    return r, q

def kronecker_indices_forward(i, j, p, q):
    """
    Parameters
    ----------
    i : int
        First Index of Kronecker Product
    j : int
        Second Index of Kronecker Product
    p : int
        First Dimension of Second Matrix, B, in Kronecker Product.
    q : int
        Second Dimension of Second Matrix, B, in Kronecker Product.
    Returns
    -------
    A_ind : tuple
        (r, s) = (i // p, j // q)
    B_ind : tuple
        (t, u) = (i % p, j % q)

    If C is the Kronecker product of A and B then 
    C_{i, j} = A_{r, s}B_{t, u} = A_{i // p, j // q}B_{i % p, j % q}
    """
    r, s = i // p, j // q
    t, u = i % p, j % q
    A_ind = r, s
    B_ind = t, u
    return A_ind, B_ind



def kronecker_indices_reverse(r, s, t, u, p, q):
    """

    Parameters
    ----------
    r : int
        First Index of First Matrix in  Kronecker Product.
    s : int
        Second Index of First Matrix in  Kronecker Product.
    t : TYPE
        First Index of Second Matrix in  Kronecker Product.
    u : TYPE
        Second Index of Second Matrix in  Kronecker Product.
    p : TYPE
        First Dimension of Second Matrix, B, in Kronecker Product.
    q : TYPE
        Second Dimension of Second Matrix, B, in Kronecker Product.

    Returns
    -------
    i : int
        First Index of Kronecker Product.
    j : int
        Second Index of Kronecker Product.

    """
    i = p * r + t
    j = q * s + u
    return i, j



def commutation_matrix_indices(m, n):
    """
    If X is mn then X_{rs} is vec(X)_{sm+r} and therefore (X^{T})_{sr} which is
    vec(X^{T})_{rn+s} implying Kmn=K has as row rn+s the sm+r row of Imn=I
    The ith row of Kmn=K is the (i%n)m+i//n th row of Imn=I i.e. if 
    i = rn + s
    then i % n = s and i // n = r and the element sm + r, sm + r is 1
    """
    ii = np.arange(m * n, dtype=int)
    i = (ii % n) * m + ii // n
    return i





