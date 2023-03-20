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
        Index in vector
    m : int
        Number of rows in matrix
    Returns
    -------
    r : int
        Row index corresponding to the vector index i
    s : int
        Column index corresponding to the vector index i

    Returns (i % m, i // m ) = (r, s) where i = sm+r
    """
    r, s =  i % m, i // m
    return r, s

def vec_inds_forwards(r, s, m):
    """
    Parameters
    ----------
    r : int
        Row index
    s : int
        Column index
    m : int
        Number of rows oin matrix

    Returns
    -------
    i : int
        Vextor index corresponding to the the index (r, s) in a matrix with
        m rows
    """
    i = s * m + r
    return i


def largest_triangular(n):
    """
    Returns the index of the largest triangular number that is less than or equal to n.

    Parameters
    ----------
    n : int
        The upper bound of the triangular numbers to consider.

    Returns
    -------
    int
        The index of the largest triangular number k such that k <= n, where the
        first triangular number is at index 1.
    """
    k = int(np.floor((-1 + np.sqrt(8 * n + 1)) / 2))
    return k

def vech_inds_reverse(i, n):
    """
    Returns the row and column indices of the lower triangular matrix element
    corresponding to the i-th element of the half vectorization.

    Parameters
    ----------
    i : int
        Index in the half vectorization
    n : int
        The number of rows (and columns) of the square matrix.

    Returns
    -------
    j : int
        Row index
    k : int
        Column index

    The tuple (j, k) represents the row and column indices in the lower
    triangular part of a matrix corresponding to the i-th element of the
    half vectorization.

    """
    q = int(n * (n + 1) / 2)                #q = n * (n + 1) / 2 - the number of elements in the vectorization of the lower half of the matrix, so i ranges over 0,....,q-1
    r = q - i - 1                           #distance of i from bottom of vector
    s = largest_triangular(r)               #index of largest triangular number less than r, i.e. less than the distance from the bottom of the vector
    t = int(s * (s + 1)//2)                 #t is the s-th triangular number
    p = r - t                               #row index counting from bottom
    j = n - p - 1                           #row index
    k = n - s - 1                           #column index
    return j, k

def vech_inds_forwards(r, s, m):
    """
    Returns the index of the (r, s)-th element in the lower triangle of a
    square matrix of size (m x m) that has been half vectorized.

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
        (r, s)-th element in the original matrix.

    """
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
    t : int
        First Index of Second Matrix in  Kronecker Product.
    u : int
        Second Index of Second Matrix in  Kronecker Product.
    p : int
        First Dimension of Second Matrix, B, in Kronecker Product.
    q : int
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
    Returns the indices needed to construct the commutation matrix Kmn.
    Kmn is the mn x mn matrix such that the product of the vectorized form
    of any m x n matrix X with Kmn is equal to the vectorized form of X^T.

    Parameters
    ----------
    m : int
        The number of rows of the matrix X.
    n : int
        The number of columns of the matrix X.

    Returns
    -------
    ii : numpy.ndarray
        An array of integers representing the indices of the mn x mn identity matrix
        as well as the row indices of the corresponding commutation matrix
    i : numpy.ndarray
        An array of integers representing the columns indices of the commutation
        matrix Kmn.

    Notes
    -------
    If X is mn then X_{rs} is vec(X)_{sm+r} and therefore (X^{T})_{sr} which is
    vec(X^{T})_{rn+s} implying Kmn=K has as row rn+s the sm+r row of Imn=I
    The ith row of Kmn=K is the (i%n)m+i//n th row of Imn=I i.e. if
    i = rn + s
    then i % n = s and i // n = r and the element sm + r, sm + r is 1
    """
    ii = np.arange(m * n, dtype=int)
    i = (ii % n) * m + ii // n
    return ii, i

def duplication_matrix_indices(n):
    """
    Compute the indices of the non-zero elements of the duplication matrix
    for half vectorizing an n by n symmetric matrix.

    Parameters:
    -----------
    n : int
        The size of the symmetric matrix.
    Returns:
    --------
    r : array_like
        The row indices of the non-zero elements of the duplication matrix.
    c : array_like
        The column indices of the non-zero elements of the duplication matrix.
    
    
    For duplication matrix corresponding to symmetric matrix of size n for 
    n>i>=j>=0 the jn+i and in+j rows are give by the i+jn-j(j+1)/2 row of
    the n(n+1)/2 sized identity matrix
    """
    i, j = tril_indices(n)
    r1 = j * n + i
    r2 = i * n + j
    c1 = c2 = i + j * n - (j * (j + 1) // 2).astype(int)
    r, c = np.concatenate([r1, r2]), np.concatenate([c1, c2])
    return r, c

def elimination_matrix_indices(n):
    """
    Compute the indices of the non-zero elements of the elimination matrix
    for n by n matrices
    Parameters
    ----------
    n : int
        The size of the matrix

    Returns
    -------
    r : array_like
        The row indices of the non-zero elements of the elimination matrix.
    c : array_like
        The column indices of the non-zero elements of the elimination matrix.

    """
    i, j = tril_indices(n)
    c = j * n + i
    r = np.arange(int(n * (n + 1) // 2))
    return r, c

