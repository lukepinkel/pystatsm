#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 17:32:36 2023

@author: lukepinkel
"""

import numpy as np
from pystatsm.utilities import special_mats as spmats
from pystatsm.utilities import indexing_utils
import itertools


def test_commutation_matrix_indices():
    sizes = list(itertools.product(range(1, 18), range(1, 18)))
    for m, n in sizes:
        K1 = spmats.kmat(m, n).A
        K2 = np.zeros((m*n, m*n))
        K2[indexing_utils.commutation_matrix_indices(m, n)]=1
        assert(np.allclose(K1, K2))


def test_duplication_matrix_indices():
    sizes = np.arange(20)
    for n in sizes:
        D1 = spmats.dmat(n).A
        D2 = np.zeros_like(D1)
        D2[indexing_utils.duplication_matrix_indices(n)]=1
        assert(np.allclose(D1, D2))

def test_elimination_matrix_indices():
    sizes = np.arange(20)
    for n in sizes:
        L1 = spmats.lmat(n).A
        L2 = np.zeros_like(L1)
        L2[indexing_utils.elimination_matrix_indices(n)]=1
        assert(np.allclose(L1, L2))


def test_vec_indices():
    m, n = 9, 6
    ii = np.arange(m * n)
    r, s = indexing_utils.vec_inds_reverse(ii, m)
    assert(np.all(indexing_utils.vec_inds_forwards(r, s, m)==ii))
    
    
    
    r, s = np.indices((m, n))
    r, s = r.reshape(-1, order='F'), s.reshape(-1, order='F')
    r1, s1 = indexing_utils.vec_inds_reverse(indexing_utils.vec_inds_forwards(r, s, m), m)
    assert(np.all(r1==r))
    assert(np.all(s1==s))
    
    
    ii = np.arange(m * n)
    ii1 = indexing_utils.vec_inds_forwards(*indexing_utils.vec_inds_reverse(ii, m), m)
    assert(np.all(ii==ii1))

def test_vech_indices(): 
    n = 6
    ii = np.arange(n * (n + 1) // 2)
    r, s = indexing_utils.vech_inds_reverse(ii, n)
    assert(np.all(indexing_utils.vech_inds_forwards(r, s, n)==ii))
    

    r, s = indexing_utils.tril_indices(n)
    r1, s1 = indexing_utils.vech_inds_reverse(indexing_utils.vech_inds_forwards(r, s, n), n)
    assert(np.all(r1==r))
    assert(np.all(s1==s))
    
    ii = np.arange(n * (n + 1) // 2)
    ii1 = indexing_utils.vech_inds_forwards(*indexing_utils.vech_inds_reverse(ii, n), n)
    assert(np.all(ii==ii1))
    
    

def test_ascending_indices():
    for ii in indexing_utils.ascending_indices((4, 4, 4, 4)):
        m = indexing_utils.ascending_indices_forward(ii)
        ii_rev = indexing_utils.ascending_indices_reversed(m, 4)
        assert(np.allclose(np.array(ii), np.array(ii_rev)))
    


