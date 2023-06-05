#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 00:31:43 2023

@author: lukepinkel
"""

import numba
import numpy as np
from ..utilities.linalg_operations import _vech_nb

@numba.jit(nopython=True)
def _dsigma(dS, L, B, F, dA, r, c, mtype, ns):
    LB = L.dot(B)
    BFBt = B.dot(F).dot(B.T)
    LBFBt = L.dot(BFBt)
    for i in range(ns):
        kind = mtype[i]
        J = np.ascontiguousarray(dA[i, :r[i], :c[i]])
        if kind == 0:
            J1 = LBFBt.dot(J.T)
            tmp = _vech_nb((J1 + J1.T))
        elif kind == 1:
            J1 = J.dot(B.dot(F))
            tmp =_vech_nb(LB.dot(J1+J1.T).dot(LB.T))
        elif kind==2:
            J1 = LB.dot(J).dot(LB.T)
            tmp = _vech_nb(J1)
        elif kind ==3:
            tmp = _vech_nb(J)
        dS[:, i] += tmp
    return dS

@numba.jit(nopython=True)
def _d2sigma(d2S, L, B, F, dA, r, c, ltr_inds, htype, ns2):
    LB, BF = L.dot(B), B.dot(F)
    BFBt = BF.dot(B.T)
    for ij in range(ns2):
        i, j = ltr_inds[ij]
        kind = htype[ij]
        Ji = np.ascontiguousarray(dA[i, :r[i], :c[i]])
        Jj = np.ascontiguousarray(dA[j, :r[j], :c[j]])
        if kind == 1:
            tmp = (Ji.dot(BFBt).dot(Jj.T) + Jj.dot(BFBt).dot(Ji.T))
            tmp = _vech_nb(tmp)
        elif kind == 2:
            BJj = B.T.dot(Jj.T)
            JiBF = Ji.dot(BF)
            C = JiBF + JiBF.T
            D = LB.dot(C).dot(BJj)
            tmp = _vech_nb(D + D.T)
        elif kind == 3:
            JjB = Jj.dot(B)
            C = JjB.dot(Ji).dot(LB.T)
            tmp = _vech_nb(C + C.T)
        elif kind == 4:
            C1 = Ji.dot(BF)
            C1 = C1 + C1.T
            C2, C3 = Ji.dot(B), Jj.dot(B)
            t1 = C3.dot(C1)
            t3 = C2.dot(C3.dot(F))
            t4 = BF.T.dot(C3.T).dot(C2.T)
            tmp = LB.dot(t1 + t1.T + t3 + t4).dot(LB.T)
            tmp = _vech_nb(tmp)
        elif kind == 5:
            C = Jj.dot(B).dot(Ji)
            tmp  = _vech_nb(LB.dot(C+C.T).dot(LB.T))
        else:
            continue
        d2S[:, i, j] += tmp
        d2S[:, j, i] = d2S[:, i, j]
    return d2S
    