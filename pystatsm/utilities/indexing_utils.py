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

def vec_inds_reverse(i, m):
    """

    Parameters
    ----------
    i : int
    m : int
    Returns
    -------
    r : int
        Remainder
    s : int
        Quotient.
        
    Returns (i % m, i // m ) = (r, s) where i = sm+r
    """
    r, s =  i % m, i // m
    return r, s

def vec_inds_forwards(r, s, m):
    i = s * m + r
    return i

    
def largest_triangular(n):
    k = int(np.floor((-1 + np.sqrt(8 * n + 1)) / 2))
    return k

def vech_inds_reverse(i, n):
    q = int(n * (n + 1) / 2)                #q = n * (n + 1) / 2 - the number of elements in the vectorization of the lower half of the matrix, so i ranges over 0,....,q-1
    r = q - i - 1                           #distance of i from bottom of vector
    s = largest_triangular(r)               #index of largest triangular number less than r, i.e. less than the distance from the bottom of the vector
    t = int(s * (s + 1)//2)                 #t is the s-th triangular number
    p = r - t                               #row index counting from bottom
    j = n - p - 1                           #row index
    k = n - s - 1                           #column index      
    return j, k

def vech_inds_forwards(r, s, m):
    i = r + s * m - int(s / 2 * (s + 1))
    return i

    


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
    return ii, i





