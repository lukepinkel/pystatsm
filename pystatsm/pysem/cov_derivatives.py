#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 00:31:43 2023

@author: lukepinkel
"""

import numba
import numpy as np
import scipy as sp
from ..utilities.linalg_operations import _vech_nb
from ..utilities.func_utils import sizes_to_ind_arrs, triangular_number
from ..utilities.indexing_utils import (vec_inds_reverse, vech_inds_reverse, 
                                        vech_inds_forwards, tril_indices)



class CovarianceDerivatives(object):
    nonzero_matrix_second_derivs = [(0, 0), (1, 0), (2, 0), (1, 1), (2, 1)]
    def __init__(self, p, q, nf, nt, ptf_ind, ttf_ind, n_matrices=4):
        self.p = p
        self.q = q
        self.nf = nf
        self.nt = nt
        self.ptf_ind = ptf_ind
        self.ttf_ind = ttf_ind
        self.n_matrices = 4
        self.calculate_matrix_size_vars()
        self.calculate_relative_indices_and_dimensions()
        self.make_derivative_matriecs()
    
    def calculate_matrix_size_vars(self):
        self.p2 = triangular_number(self.p)
        self.q2 = triangular_number(self.q)
        self.pq = self.p * self.q
        self.qq = self.q * self.q
        self.ns = self.pq + self.qq + self.q2 + self.p2
        self.ns2 = triangular_number(self.ns)
        self.nf2 = triangular_number(self.nf)
        self.p_or_q = max(self.p, self.q)
        self.matrix_sizes = np.array([self.pq, self.qq, self.q2, self.p2])
        self.matrix_index_offset = np.r_[0, np.cumsum(self.matrix_sizes)]

    def calculate_relative_indices_and_dimensions(self):
        matrix_sizes = [self.pq, self.qq, self.q2, self.p2]
        matrix_dims = np.array([[self.p, self.q], 
                                [self.q, self.q],
                                [self.q, self.q],
                                [self.p, self.p]], dtype=int)

        # Determine indices in the parameter vector corresponding to each matrix type
        parameter_indices = sizes_to_ind_arrs(matrix_sizes)
    
        # Intersect parameter indices with indices of free parameters
        free_parameter_indices = {}
        for i in range(self.n_matrices):
            free_parameter_indices[i] = np.intersect1d(parameter_indices[i], self.ptf_ind)
    
        # Get the relative indices for each matrix type
        relative_indices = self.get_relative_index(free_parameter_indices)
    
        # Store the location in the vector of free parameters corresponding to each matrix type
        free_parameter_locations = sizes_to_ind_arrs([len(free_parameter_indices[i]) for i in range(self.n_matrices)])
    
        # Initialize array to store matrix type for each free parameter
        matrix_type = np.zeros(self.nf, dtype=int)
    
        # Initialize array to store row and column indices for each free parameter
        matrix_indices = np.zeros((self.nf, 2), dtype=int)
    
        # For each matrix type...
        for i in range(self.n_matrices):
            # Assign matrix type to the corresponding free parameters
            matrix_type[free_parameter_locations[i]] = i
    
            # If the matrix type is L or B...
            if i < 2:
                row, col = vec_inds_reverse(relative_indices[i], matrix_dims[i, 0])
            # If the matrix type is F or P...
            elif i >= 2:
                row, col = vech_inds_reverse(relative_indices[i], matrix_dims[i, 0])
            else:
                continue
    
            # Assign row and column indices to the corresponding free parameters
            matrix_indices[free_parameter_locations[i], 0] = row
            matrix_indices[free_parameter_locations[i], 1] = col
    
        self.matrix_type = matrix_type
        self.matrix_indices = matrix_indices    
        self.matrix_dims = matrix_dims
        self.parameter_indices = parameter_indices
        self.free_parameter_indices = free_parameter_indices
        self.relative_free_parameter_indices = relative_indices
        self.free_parameter_locations = free_parameter_locations

    
    def make_derivative_matriecs(self):
        second_deriv_pairs = tril_indices(self.nf)
        matrix_type_j = self.matrix_type[second_deriv_pairs[0]]
        matrix_type_k = self.matrix_type[second_deriv_pairs[1]]
        num_unique_pairs = self.nf2
        matrix_pair_types = np.zeros(num_unique_pairs, dtype=int)
        for i, (mat_j, mat_k) in enumerate(self.nonzero_matrix_second_derivs, 1):
            mask_j = (matrix_type_j == mat_j)
            mask_k = (matrix_type_k == mat_k)
            matrix_pair_types[mask_j & mask_k] = i
        
        matrix_derivatives = np.zeros((self.nf, self.p_or_q, self.p_or_q))
        row_size, col_size = np.zeros((2, self.nf), dtype=int)
        for i in range(self.nf):
            row, col = self.matrix_indices[i]
            matrix = self.matrix_type[i]
            row_size[i], col_size[i] = self.matrix_dims[matrix]
            matrix_derivatives[i, row, col] = 1.0
            if matrix > 1:
                matrix_derivatives[i, col, row] = 1.0
        s, r = np.triu_indices(self.p, k=0)
        self._vech_inds = r+s*self.p
        self.dA = self.matrix_derivatives = matrix_derivatives
        self.dS = np.zeros((self.p2, self.nf))
        self.d2S = np.zeros((self.p2, self.nf, self.nf))
        self.d2_inds = np.vstack(second_deriv_pairs).T
        self.d2_kind = matrix_pair_types
        self.r, self.c = row_size, col_size
        self.J_theta = sp.sparse.csc_array(
            (np.ones(self.nf), (np.arange(self.nf), self.ttf_ind)), 
             shape=(self.nf, self.nt))
        s, r = np.triu_indices(self.p, k=0)
        self._vech_inds = r+s*self.p
        
        self.cov_first_deriv_kws = dict(dA=self.dA, r=self.r, c=self.c, deriv_type=self.matrix_type,
                                        n=self.nf, vech_inds=self._vech_inds)
        
        self.cov_second_deriv_kws = dict(dA=self.dA, r=self.r, c=self.c, deriv_type=self.d2_kind,
                                        n=self.nf2, vech_inds=self._vech_inds,
                                        d2_inds=self.d2_inds)
        self.f_first_deriv_kws = dict(dA=self.dA, r=self.r, c=self.c, deriv_type=self.matrix_type, 
                                      n=self.nf,  vech_inds=self._vech_inds)
        
        
        
            
    def get_relative_index(self, absolute_mat_inds):
        offset = self.matrix_index_offset
        rel_mat_inds = {}
        for i in range(4):
            rel_mat_inds[i] = absolute_mat_inds[i] - offset[i]
        return rel_mat_inds
        
@numba.jit(nopython=True)
def _dsigma(dS, L, B, F, dA, r, c, deriv_type, n, vech_inds):
    LB = L.dot(B)
    BF = B.dot(F)
    LBt = LB.T
    BFBt = BF.dot(B.T)
    LBFBt = L.dot(BFBt)
    for i in range(n):
        kind = deriv_type[i]
        J = dA[i, :r[i], :c[i]]
        if kind == 0:
            J1 = LBFBt.dot(J.T)
            tmp = (J1 + J1.T)
        elif kind == 1:
            J1 = J.dot(BF)
            tmp =LB.dot(J1+J1.T).dot(LBt)
        elif kind==2:
            J1 = LB.dot(J).dot(LBt)
            tmp = J1
        elif kind ==3:
            tmp = J
        dS[:, i] += tmp.T.flatten()[vech_inds]
    return dS

@numba.jit(nopython=True)
def _d2sigma(d2S, L, B, F, dA, r, c, d2_inds, deriv_type, n, vech_inds):
    LB, BF = L.dot(B), B.dot(F)
    BFBt = BF.dot(B.T)
    for ij in range(n):
        i, j = d2_inds[ij]
        kind = deriv_type[ij]
        Ji = dA[i, :r[i], :c[i]]
        Jj = dA[j, :r[j], :c[j]]
        if kind == 1:
            tmp = (Ji.dot(BFBt).dot(Jj.T) + Jj.dot(BFBt).dot(Ji.T))
        elif kind == 2:
            BJj = B.T.dot(Jj.T)
            JiBF = Ji.dot(BF)
            C = JiBF + JiBF.T
            D = LB.dot(C).dot(BJj)
            tmp = D + D.T
        elif kind == 3:
            JjB = Jj.dot(B)
            C = JjB.dot(Ji).dot(LB.T)
            tmp = C + C.T
        elif kind == 4:
            C1 = Ji.dot(BF)
            C1 = C1 + C1.T
            C2, C3 = Ji.dot(B), Jj.dot(B)
            t1 = C3.dot(C1)
            t3 = C2.dot(C3.dot(F))
            t4 = BF.T.dot(C3.T).dot(C2.T)
            tmp = LB.dot(t1 + t1.T + t3 + t4).dot(LB.T)
            tmp = tmp
        elif kind == 5:
            C = Jj.dot(B).dot(Ji)
            tmp  = LB.dot(C+C.T).dot(LB.T)
        else:
            continue
        d2S[:, i, j] += tmp.T.flatten()[vech_inds]
        d2S[:, j, i] = d2S[:, i, j]
    return d2S
    

@numba.jit(nopython=True)
def _dloglike(g, L, B, F, vecVRV, dA, r, c, deriv_type, n, vech_inds):
    LB = L.dot(B)
    BF = B.dot(F)
    LBt = LB.T
    BFBt = BF.dot(B.T)
    LBFBt = L.dot(BFBt)
    for i in range(n):
        kind = deriv_type[i]
        J = dA[i, :r[i], :c[i]]
        if kind == 0:
            J1 = LBFBt.dot(J.T)
            tmp = (J1 + J1.T)
        elif kind == 1:
            J1 = J.dot(BF)
            tmp = LB.dot(J1+J1.T).dot(LBt)
        elif kind==2:
            J1 = LB.dot(J).dot(LBt)
            tmp = J1
        elif kind ==3:
            tmp = J
        g[i] += -np.dot(vecVRV, tmp.flatten())
    return g

@numba.jit(nopython=True)
def _d2loglike(H, L, B, F, Sinv, S, vecVRV, vecV, dA, r, c, first_deriv_type, 
               second_deriv_type, n, vech_inds):
    LB, BF = L.dot(B), B.dot(F)
    LBt = LB.T
    BFt = BF.T
    Bt = B.T
    BFBt = BF.dot(Bt)
    LB = L.dot(B)
    BF = B.dot(F)
    BFBt = BF.dot(B.T)
    LBFBt = L.dot(BFBt)
    vecVRV2 = vecVRV + vecV / 2
    ij = 0
    for j in range(n):
        kindj = first_deriv_type[j]
        Jj = dA[j, :r[j], :c[j]]
        if kindj == 0:
            J1 = LBFBt.dot(Jj.T)
            D1Sj = (J1 + J1.T)
        elif kindj == 1:
            J1 = Jj.dot(BF)
            D1Sj = LB.dot(J1+J1.T).dot(LBt)
        elif kindj==2:
            J1 = LB.dot(Jj).dot(LBt)
            D1Sj = J1
        elif kindj ==3:
            D1Sj = Jj

        for i in range(j, n):
            kindi = first_deriv_type[i]
            kindij = second_deriv_type[ij]
            ij += 1
            Ji = dA[i, :r[i], :c[i]]
            
            if kindi == 0:
                J1 = LBFBt.dot(Ji.T)
                D1Si = (J1 + J1.T)
            elif kindi == 1:
                J1 = Ji.dot(BF)
                D1Si = LB.dot(J1+J1.T).dot(LBt)
            elif kindi==2:
                J1 = LB.dot(Ji).dot(LBt)
                D1Si = J1
            elif kindi ==3:
                D1Si = Ji
                
            SiSj = D1Si.dot(Sinv).dot(D1Sj)
            H[i, j] += 2*np.dot(vecVRV2, SiSj.flatten()) 
            H[j, i] = H[i, j]
            if kindij == 1:
                D2Sij = (Ji.dot(BFBt).dot(Jj.T) + Jj.dot(BFBt).dot(Ji.T))
            elif kindij == 2:
                BJj = Bt.dot(Jj.T)
                JiBF = Ji.dot(BF)
                C = JiBF + JiBF.T
                D = LB.dot(C).dot(BJj)
                D2Sij = D + D.T
            elif kindij == 3:
                JjB = Jj.dot(B)
                C = JjB.dot(Ji).dot(LBt)
                D2Sij = C + C.T
            elif kindij == 4:
                C1 = Ji.dot(BF)
                C1 = C1 + C1.T
                C2, C3 = Ji.dot(B), Jj.dot(B)
                t1 = C3.dot(C1)
                t3 = C2.dot(C3.dot(F))
                t4 = BFt.dot(C3.T).dot(C2.T)
                tmp = LB.dot(t1 + t1.T + t3 + t4).dot(LBt)
                D2Sij = tmp
            elif kindij == 5:
                C = Jj.dot(B).dot(Ji)
                D2Sij  = LB.dot(C+C.T).dot(LBt)
            
            if kindij>0:
                H[i, j] +=  -np.dot(vecVRV, D2Sij.flatten())
                H[j, i] = H[i, j]
    return H