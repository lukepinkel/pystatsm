# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 15:24:55 2021

@author: lukepinkel
"""
import timeit
import numpy as np
import scipy as sp
import scipy.stats
import scipy.linalg
from pystatsm.utilities import linalg_operations as linalg 
from pystatsm.utilities.random import exact_rmvnorm



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
    
def test_vec_alt():
    n = 17
    m = 11
    p = 29
    q = 19

    x2 = np.arange(n*m).reshape(n, m)
    y2 = linalg._vec(x2)

    x3 = np.arange(n*m*p).reshape(p, n, m)
    y3 = linalg._vec(x3)

    x4 = np.arange(n*m*p*q).reshape(p, q, n, m)
    y4 = linalg._vec(x4)


    assert(np.allclose(linalg._invec(linalg._vec(x2), n, m), x2))
    assert(np.allclose(linalg._vec(linalg._invec(y2, n, m)), y2))

    assert(np.allclose(linalg._invec(linalg._vec(x3), n, m), x3))
    assert(np.allclose(linalg._vec(linalg._invec(y3, n, m)), y3))

    assert(np.allclose(linalg._invec(linalg._vec(x4), n, m), x4))
    assert(np.allclose(linalg._vec(linalg._invec(y4, n, m)), y4))
        
    n = 7
    m = 2
    p = 3
    q = 5
    
    a2 = np.arange(m*n).reshape(m, n)
    a3 = np.arange(m*n*p).reshape(p, m, n)
    a4 = np.arange(m*n*p*q).reshape(p, q, m, n)
    
    
    x2 = np.einsum("kl,km->lm", a2, a2)
    y2 = linalg._vech(x2)
    
    x3 = np.einsum("jkl,jkm->jlm", a3, a3)
    y3 = linalg._vech(x3)
    
    x4 = np.einsum("ijkl,ijkm->ijlm", a4, a4)
    y4 = linalg._vech(x4)
    
    assert(np.allclose(linalg._invech(linalg._vech(x2)), x2))
    assert(np.allclose(linalg._vech(linalg._invech(y2)), y2))
    
    assert(np.allclose(linalg._invech(linalg._vech(x3)), x3))
    assert(np.allclose(linalg._vech(linalg._invech(y3)), y3))
    
    assert(np.allclose(linalg._invech(linalg._vech(x4)), x4))
    assert(np.allclose(linalg._vech(linalg._invech(y4)), y4))
    
    x2[..., np.diag_indices(n)[0], np.diag_indices(n)[1]] = 1
    y2 =  linalg._vecl(x2)
    x3[..., np.diag_indices(n)[0], np.diag_indices(n)[1]] = 1
    y3 =  linalg._vecl(x3)
    x4[..., np.diag_indices(n)[0], np.diag_indices(n)[1]] = 1
    y4 =  linalg._vecl(x4)
    
    assert(np.allclose(linalg._invecl(linalg._vecl(x2)), x2))
    assert(np.allclose(linalg._vecl(linalg._invecl(y2)), y2))
    
    assert(np.allclose(linalg._invecl(linalg._vecl(x3)), x3))
    assert(np.allclose(linalg._vecl(linalg._invecl(y3)), y3))
    
    assert(np.allclose(linalg._invecl(linalg._vecl(x4)), x4))
    assert(np.allclose(linalg._vecl(linalg._invecl(y4)), y4))







def test_wls():
    rng = np.random.default_rng(123)
    eigvals = np.arange(1, 6)**4 * 1.0
    eigvals /= np.sum(eigvals) / len(eigvals)
    R = sp.stats.random_correlation.rvs(eigvals, random_state=rng)
    X = exact_rmvnorm(R, n=1000, seed=123)
    
    w_pos = rng.uniform(0.01, 10, size=(1000))
    w_mix = rng.uniform(-5, 5, size=(1000))
    
    y = rng.normal(0, 1, size=1000)
    
    b_pos = np.linalg.solve(np.dot(X.T * w_pos, X), np.dot(X.T *w_pos, y))
    b_pos_chol = linalg.wls_chol(X, y, w_pos)
    b_pos_qr = linalg.wls_qr(X, y, w_pos)
    b_pos_n = linalg.nwls(X, y, w_pos)
    
    assert(np.allclose(b_pos, b_pos_chol))
    assert(np.allclose(b_pos, b_pos_qr))
    assert(np.allclose(b_pos, b_pos_n))


    b_neg = np.linalg.solve(np.dot(X.T * w_mix, X), np.dot(X.T * w_mix, y))
    b_neg_chol = linalg.nwls(X, y, w_mix)
    
    assert(np.allclose(b_neg, b_neg_chol))

    
