# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 15:24:55 2021

@author: lukepinkel
"""
import timeit
import numpy as np
import scipy as sp
import scipy.linalg
from pystatsm.utilities import linalg_operations as linalg 


def test_chol_downdate():
    LA = np.array([[ 5.0,  0.0,  0.0,  0.0],
                   [ 3.0,  4.0,  0.0,  0.0],
                   [-2.0,  1.0,  3.0,  0.0],
                   [ 1.0, -1.0,  0.5,  2.0]])
    
    
    A = np.dot(LA, LA.T)
    B = A[np.ix_([0, 1, 3], [0, 1, 3])]
    
    LB1 = np.linalg.cholesky(B)
    LB2 = linalg.chol_downdate(LA.copy(), 2)[:-1, :-1]
    
    assert(np.allclose(LB1, LB2))
    
def test_chol_update():
    LA = np.array([[ 5.0,  0.0,  0.0,  0.0],
                   [ 3.0,  4.0,  0.0,  0.0],
                   [-2.0,  1.0,  3.0,  0.0],
                   [ 1.0, -1.0,  0.5,  2.0]])
    
    
    A = np.dot(LA, LA.T)
    B = A[np.ix_([0, 1, 3], [0, 1, 3])]
    
    LB = np.linalg.cholesky(B)
    LA1 = LA[:-1, :-1]
    LA2 = linalg.add_chol_row(A, LB.copy(), 2)
    assert(np.allclose(LA1, LA2))
    
    
def test_toeplitz_chol():
    A = 0.5**sp.linalg.toeplitz(np.arange(100))
    L1 = linalg.toeplitz_cholesky_lower_nb(100, A)
    L2 = np.linalg.cholesky(A)
    assert(np.allclose(L1, L2))
    
    #t1 = timeit.timeit("linalg.toeplitz_cholesky_lower_nb(100, A)", number=10000, globals=globals())
    #t2 = timeit.timeit("np.linalg.cholesky(A)", number=10000, globals=globals())
    
    #assert(t1<t2)
    
def test_vec():
    a1 = np.arange(8)
    A = linalg.invec(a1, 4, 2)
    a2 = linalg.vec(A)
    assert(np.allclose(a1, a2))
    
    a1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    A = linalg.invech(a1)
    a2 = linalg.vech(A)
    
    assert(np.allclose(A, A.T))
    assert(np.allclose(a1, a2))
    
    a1 = np.array([1.0, 2.0, 3.0])
    A = linalg.invecl(a1)
    a2 = linalg.vecl(A)
    
    assert(np.allclose(A, A.T))
    assert(np.allclose(a1, a2))
    
    L1 = np.array([[ 5.0,  0.0,  0.0,  0.0],
                   [ 3.0,  4.0,  0.0,  0.0],
                   [-2.0,  1.0,  3.0,  0.0],
                   [ 1.0, -1.0,  0.5,  2.0]])
    
    lv = linalg.vech(L1)
    L2 = linalg.invech_chol(lv)
    
    assert(np.allclose(L1, L2))





    
