#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 01:53:02 2023

@author: lukepinkel
"""

import numba
import numpy as np

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
def _dsigma_mu(dSm, L, B, F, b, a, dA, m_size, m_type, vind, n, p2):
    LB, BF = L.dot(B), B.dot(F)
    Bbt = B.dot(b.T)
    LBt, BFBt = LB.T, BF.dot(B.T)
    LBFBt = L.dot(BFBt)
    for i in range(n):
        kind = m_type[i]
        J = dA[i, :m_size[i, 0], :m_size[i, 1]]
        if kind == 0:
            J1 = LBFBt.dot(J.T)
            dSi = (J1 + J1.T)
            dmi = J.dot(Bbt)
            dSm[:p2, i] += dSi.T.flatten()[vind]
            dSm[p2: ,i] += dmi.T
        elif kind == 1:
            J1 = J.dot(BF)
            dSi =LB.dot(J1+J1.T).dot(LBt)
            dmi = LB.dot(J).dot(Bbt)
            dSm[:p2, i] += dSi.T.flatten()[vind]
            dSm[p2: ,i] += dmi
        elif kind==2:
            J1 = LB.dot(J).dot(LBt)
            dSi = J1
            dSm[:p2, i] += dSi.T.flatten()[vind]
        elif kind ==3:
            dSi = J
            dSm[:p2, i] += dSi.T.flatten()[vind]
        elif kind == 4:
            dmi = J[0]
            dSm[p2: ,i] += dmi
        elif kind == 5:
            dmi = LB.dot(J[0].T)
            dSm[p2:,i] += dmi
    return dSm      


@numba.jit(nopython=True)
def _d2sigma_mu(d2Sm, L, B, F, b, a, dA, m_size, m_type, d2_inds, vind, n, p2):
    LB, BF = L.dot(B), B.dot(F)
    BFBt = BF.dot(B.T)
    Bb = B.dot(b)
    for ij in range(n):
        i, j = d2_inds[ij]
        kind = m_type[ij]
        Ji =dA[i, :m_size[i, 0], :m_size[i, 1]]
        Jj =dA[j, :m_size[j, 0], :m_size[j, 1]]
        if kind == 1:
            dSij = (Ji.dot(BFBt).dot(Jj.T) + Jj.dot(BFBt).dot(Ji.T))  #0, 0 L ,L
            d2Sm[:p2, i, j] += dSij.T.flatten()[vind]
            d2Sm[:p2, j, i] = d2Sm[:p2, i, j]
        elif kind == 2:                         
            BJj = B.T.dot(Jj.T)                                      #1,0, B, L
            JiBF = Ji.dot(BF)
            C = JiBF + JiBF.T
            D = LB.dot(C).dot(BJj)
            dSij = D + D.T
            d2Sm[:p2, i, j] += dSij.T.flatten()[vind]
            d2Sm[:p2, j, i] = d2Sm[:p2, i, j]
            dmij = Jj.dot(B).dot(Ji).dot(Bb)
            d2Sm[p2:, i, j] += dmij.flatten()
        elif kind == 3:
            JjB = Jj.dot(B)                                          #2,0, F, L
            C = JjB.dot(Ji).dot(LB.T)
            dSij = C + C.T
            d2Sm[:p2, i, j] += dSij.T.flatten()[vind]
            d2Sm[:p2, j, i] = d2Sm[:p2, i, j]
        elif kind == 4:
            C1 = Ji.dot(BF)                                          #1,1, B, B
            C1 = C1 + C1.T
            C2, C3 = Ji.dot(B), Jj.dot(B)
            t1 = C3.dot(C1)
            t3 = C2.dot(C3.dot(F))
            t4 = BF.T.dot(C3.T).dot(C2.T)
            dSij = LB.dot(t1 + t1.T + t3 + t4).dot(LB.T)
            d2Sm[:p2, i, j] += dSij.T.flatten()[vind]
            d2Sm[:p2, j, i] = d2Sm[:p2, i, j]
            
            dmij = LB.dot(Ji).dot(B).dot(Jj).dot(Bb)
            d2Sm[p2:, i, j] += dmij.flatten()
        elif kind == 5:
            C = Jj.dot(B).dot(Ji)                                    #2,1, F, B
            dSij  = LB.dot(C+C.T).dot(LB.T)
            d2Sm[:p2, i, j] += dSij.T.flatten()[vind]
            d2Sm[:p2, j, i] = d2Sm[:p2, i, j]
        elif kind == 6:
            dmij = (Ji.dot(B)).dot(Jj.T)[0]                                  #5, 0, b, L
            d2Sm[p2:, i, j] += dmij.flatten()
        elif kind == 7:
            dmij = LB.dot(Jj).dot(B).dot(Ji.T)                          #5, 1, b, B
            d2Sm[p2:, i, j] += dmij.flatten()
    return d2Sm

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
def _dloglike_mu(g, L, B, F, b, a, vecVRV, rtV, dA, m_size, m_type, vind, n, p2):
    LB, BF = L.dot(B), B.dot(F)
    Bbt = B.dot(b.T)
    LBt, BFBt = LB.T, BF.dot(B.T)
    LBFBt = L.dot(BFBt)
    for i in range(n):
        kind = m_type[i]
        J = dA[i, :m_size[i, 0], :m_size[i, 1]]
        if kind == 0:
            J1 = LBFBt.dot(J.T)
            dSi = (J1 + J1.T)
            dmi = J.dot(Bbt)
            g[i] += -np.dot(vecVRV, dSi.flatten())
            g[i] += -2.0 * np.dot(dmi.T, rtV)
        elif kind == 1:
            J1 = J.dot(BF)
            dSi =LB.dot(J1+J1.T).dot(LBt)
            dmi = LB.dot(J).dot(Bbt)
            g[i] += -np.dot(vecVRV, dSi.flatten())
            g[i] += -2.0 * np.dot(dmi.T, rtV)
        elif kind==2:
            J1 = LB.dot(J).dot(LBt)
            dSi = J1
            g[i] += -np.dot(vecVRV, dSi.flatten())
        elif kind ==3:
            dSi = J
            g[i] += -np.dot(vecVRV, dSi.flatten())
        elif kind == 4:
            dmi = J[0]
            g[i] += -2.0 * np.dot(dmi.T, rtV)
        elif kind == 5:
            dmi = LB.dot(J[0].T)
            g[i] += -2.0 * np.dot(dmi.T, rtV)
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

@numba.jit(nopython=True)
def _d2loglike_mu(H, L, B, F, Sinv, S, vecVRV, vecV, dA, r, c, first_deriv_type, 
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