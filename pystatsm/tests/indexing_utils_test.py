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
    for ii in indexing_utils.ascending_indices((5, 5, 5, 5)):
        m = indexing_utils.ascending_indices_forward(ii)
        ii_rev = indexing_utils.ascending_indices_reversed(m, 4 ,5)
        assert(np.allclose(np.array(ii), np.array(ii_rev)))
    
def check_indexing(shape, rvs, fwd, order, verbose=False):
    """
    Test function to verify that forward and reverse index transformations are consistent.
    
    Arguments:
    shape -- the shape of the array to be indexed
    rvs -- the reverse index transformation function
    fwd -- the forward index transformation function
    order -- the order of the array ('C' for row-major, 'F' for column-major)
    
    Returns:
    A boolean value indicating whether all tests passed (True) or not (False).
    """
    all_tests_passed = True
    
    # Iterate over all possible indices in the array
    for i, ind in enumerate(indexing_utils.ndindex(tuple(shape), order=order)):
        # Convert index to array for easier comparison
        ind = np.array(ind)

        # Test reverse transformation
        rev_ind = rvs(i, shape)
        if not np.array_equal(ind, rev_ind):
            if verbose:
                print(f"Reverse transformation failed for index {i}. Expected {ind}, got {rev_ind}.")
            all_tests_passed = False

        # Test forward transformation
        fwd_i = fwd(ind, shape)
        if fwd_i != i:
            if verbose:
                print(f"Forward transformation failed for index {ind}. Expected {i}, got {fwd_i}.")
            all_tests_passed = False

        # Test that forward and reverse transformations are consistent
        if not np.array_equal(ind, rvs(fwd_i, shape)):
            if verbose:
                print(f"Inconsistent transformations for index {ind}.")
            all_tests_passed = False
            
    return all_tests_passed
    

def check_edge_cases(rvs, fwd):
    # Test single-dimensional shape
    all_tests_passed = True
    shape = np.array([5])
    all_tests_passed = all_tests_passed & (fwd(np.array([3]), shape) == 3)
    all_tests_passed = all_tests_passed & (np.array_equal(rvs(3, shape), np.array([3])))
    
    # Test shape with dimension size 1
    shape = np.array([1, 1, 1, 1])
    all_tests_passed = all_tests_passed & (fwd(np.array([0, 0, 0, 0]), shape) == 0)
    all_tests_passed = all_tests_passed & (np.array_equal(rvs(0, shape), np.zeros(shape.shape, dtype=int)))
    return all_tests_passed
    

def test_indexing():
    
    shape = np.array([11, 2, 5, 7, 13])
    assert(check_indexing(shape, indexing_utils.lex_index_reverse, indexing_utils.lex_index_forward, 'C'))
    assert(check_indexing(shape, indexing_utils.colex_index_reverse, indexing_utils.colex_index_forward, 'F'))
    assert(check_edge_cases(indexing_utils.lex_index_reverse, indexing_utils.lex_index_forward))
    assert(check_edge_cases(indexing_utils.colex_index_reverse, indexing_utils.colex_index_forward))

    
    shape = np.array([1, 1])
    assert(check_indexing(shape, indexing_utils.lex_index_reverse, indexing_utils.lex_index_forward, 'C'))
    assert(check_indexing(shape, indexing_utils.colex_index_reverse, indexing_utils.colex_index_forward, 'F'))
    



