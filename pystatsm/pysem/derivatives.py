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
            d2Sm[p2:, j, i] = d2Sm[p2:, i, j]
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
            t4 = F.T.dot(C3.T).dot(C2.T)
            dSij = LB.dot(t1 + t1.T + t3 + t4).dot(LB.T)
            d2Sm[:p2, i, j] += dSij.T.flatten()[vind]
            d2Sm[:p2, j, i] = d2Sm[:p2, i, j]
            dmij = LB.dot(Ji).dot(B).dot(Jj).dot(Bb)
            d2Sm[p2:, i, j] += dmij.flatten()
            d2Sm[p2:, j, i] = d2Sm[p2:, i, j]
        elif kind == 5:
            C = Jj.dot(B).dot(Ji)                                    #2,1, F, B
            dSij  = LB.dot(C+C.T).dot(LB.T)
            d2Sm[:p2, i, j] += dSij.T.flatten()[vind]
            d2Sm[:p2, j, i] = d2Sm[:p2, i, j]
        elif kind == 6:
            dmij = Jj.dot(B.dot(Ji[0]))                                  #5, 0, b, L
            d2Sm[p2:, i, j] += dmij.flatten()
            d2Sm[p2:, j, i] = d2Sm[p2:, i, j]
        elif kind == 7:
            dmij = LB.dot(Jj).dot(B).dot(Ji.T)                          #5, 1, b, B
            d2Sm[p2:, i, j] += dmij.flatten()
            d2Sm[p2:, j, i] = d2Sm[p2:, i, j]
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
            t4 = F.T.dot(C3.T).dot(C2.T)
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
            g[i] += -(2.0 * np.dot(dmi.T, rtV) + rtV.dot(dSi.dot(rtV)))
        elif kind == 1:
            J1 = J.dot(BF)
            dSi =LB.dot(J1+J1.T).dot(LBt)
            dmi = LB.dot(J).dot(Bbt)
            g[i] += -np.dot(vecVRV, dSi.flatten())
            g[i] += -(2.0 * np.dot(dmi.T, rtV) + rtV.dot(dSi.dot(rtV)))
        elif kind==2:
            J1 = LB.dot(J).dot(LBt)
            dSi = J1
            g[i] += -np.dot(vecVRV, dSi.flatten())
            g[i] += -(rtV.dot(dSi.dot(rtV)))
        elif kind ==3:
            dSi = J
            g[i] += -np.dot(vecVRV, dSi.flatten())
            g[i] += -(rtV.dot(dSi.dot(rtV)))
        elif kind == 4:
            dmi = J[0]
            g[i] += -2.0 * np.dot(dmi.T, rtV)
        elif kind == 5:
            dmi = LB.dot(J[0].T)
            g[i] += -2.0 * np.dot(dmi.T, rtV)
    return g      


@numba.jit(nopython=True)
def compute_dsigma_dmu(LB, BF, LBt, LBFBt, Bbt, j, dA, m_type, m_size, sigma_0, mu_0):
    kind = m_type[j]
    Jj = dA[j, :m_size[j ,0], :m_size[j, 1]]
    if kind == 0: # L
        J1 = LBFBt.dot(Jj.T) 
        sigma_j = (J1 + J1.T)
        mu_j = Jj.dot(Bbt) 
    elif kind == 1: #B
        J1 = Jj.dot(BF)
        sigma_j = LB.dot(J1+J1.T).dot(LBt)
        mu_j = LB.dot(Jj).dot(Bbt)
    elif kind==2: #F
        J1 = LB.dot(Jj).dot(LBt)
        sigma_j = J1
        mu_j = mu_0
    elif kind ==3: #P
        sigma_j = Jj
        mu_j = mu_0
    elif kind == 4:
        sigma_j = sigma_0
        mu_j = Jj[0]
    elif kind == 5:
        sigma_j = sigma_0
        mu_j = LB.dot(Jj[0].T)
    return sigma_j, mu_j


@numba.jit(nopython=True)
def compute_d2sigma_mu(L, B, F, LB, LBt, BF, BFBt, Bb, ij, dA, d2_inds, m_type, m_size, mu_0, sigma_0):
    i, j = d2_inds[ij]
    kind = m_type[ij]
    Ji =dA[i, :m_size[i, 0], :m_size[i, 1]]
    Jj =dA[j, :m_size[j, 0], :m_size[j, 1]]
    if kind == 1:
        dSij = (Ji.dot(BFBt).dot(Jj.T) + Jj.dot(BFBt).dot(Ji.T))  #0, 0 L ,L
        dmij = mu_0
    elif kind == 2:                         
        BJj = B.T.dot(Jj.T)                                      #1,0, B, L
        JiBF = Ji.dot(BF)
        C = JiBF + JiBF.T
        D = LB.dot(C).dot(BJj)
        dSij = D + D.T
        dmij = Jj.dot(B).dot(Ji).dot(Bb)
    elif kind == 3:
        JjB = Jj.dot(B)                                          #2,0, F, L
        C = JjB.dot(Ji).dot(LBt)
        dSij = C + C.T
        dmij = mu_0
    elif kind == 4:
        C1 = Ji.dot(BF)                                          #1,1, B, B
        C1 = C1 + C1.T
        C2, C3 = Ji.dot(B), Jj.dot(B)
        t1 = C3.dot(C1)
        t3 = C2.dot(C3.dot(F))
        t4 = F.T.dot(C3.T).dot(C2.T)
        dSij = LB.dot(t1 + t1.T + t3 + t4).dot(LBt)
        dmij = LB.dot(Ji).dot(B).dot(Jj).dot(Bb)
    elif kind == 5:
        C = Jj.dot(B).dot(Ji)                                    #2,1, F, B
        dSij  = LB.dot(C+C.T).dot(LBt)
        dmij = mu_0
    elif kind == 6:
        dmij = Jj.dot(B.dot(Ji[0])).flatten()                            #5, 0, b, L
        dSij = sigma_0
    elif kind == 7:
        dmij = LB.dot(Jj).dot(B).dot(Ji.T).flatten()                          #5, 1, b, 
        dSij = sigma_0
    else:
        dSij = sigma_0
        dmij= mu_0
    return dSij, dmij

@numba.jit(nopython=True)
def _d2loglike_mu1(H, d1Sm, L, B, F, P, a, b, Sinv, S, d, vecVRV, vecV, dA, m_size,
                d2_inds,  first_deriv_type, second_deriv_type, n, vech_inds):
    LB, BF = L.dot(B), B.dot(F)
    LBt = LB.T
    Bt = B.T
    BFBt = BF.dot(Bt)
    Bbt = B.dot(b.T)
    BFBt = BF.dot(Bt)
    LBFBt = L.dot(BFBt)
    vecVRV2 = vecVRV + vecV / 2
    mu_0 = np.zeros(a.shape)
    sigma_0 = np.zeros(S.shape)
    dtV = np.dot(d.T, Sinv)
    ij = 0
    for j in range(n):
        sigma_j, mu_j = compute_dsigma_dmu(LB, BF, LBt, LBFBt, Bbt, j, dA, first_deriv_type, m_size, sigma_0, mu_0)
        d1Sm[j, :, :-1] = sigma_j
        d1Sm[j, :, -1] = mu_j
    for j in range(n):
        sigma_j, mu_j  = d1Sm[j, :, :-1], d1Sm[j, :, -1]
        for i in range(j, n):
            sigma_i, mu_i  = d1Sm[i, :, :-1], d1Sm[i, :, -1]
            sigma_ij, mu_ij = compute_d2sigma_mu(L, B, F, LB, LBt, BF, BFBt, Bbt,
                                                 ij, dA, d2_inds, second_deriv_type, m_size,
                                                 mu_0, sigma_0)
            i, j = d2_inds[ij]
            kindij = second_deriv_type[ij]
            kindi = first_deriv_type[i]
            kindj = first_deriv_type[j]
            # #kindi in {2, 3} then mu_i is zero
            # #kindi in {4, 5} then sigma_i is zero
            # #kindij in {1, 3, 5,} then mu_ij is zero
            # #kindij in {6, 7} then sigma_ij is zero
            # if (kindi != 4 and kindi != 5) and (kindj != 4 and kindj != 5): 
            #     SiSj = sigma_i.dot(Sinv).dot(sigma_j)
            #     H[i, j] += 2 * np.dot(vecVRV2, SiSj.flatten()) 
            #     #t1 = 2 * np.dot(vecVRV2, SiSj.flatten()) 
            # if kindij != 6 and kindij != 7:
            #     H[i, j] +=  -np.dot(vecVRV, sigma_ij.flatten())
            #     #t2 =  -np.dot(vecVRV, sigma_ij.flatten())
            # if (kindi != 2 and kindi != 3) and (kindj != 2 and kindj != 3): 
            #     H[i, j] += 2 * np.dot(mu_j, Sinv.dot(mu_i))
            #     #t3 = 2 * np.dot(mu_j, Sinv.dot(mu_i)) 
            # if kindij != 1 and kindij != 3 and kindij != 5:
            #     H[i, j] -= 2 * dtV.dot(mu_ij)
            #     #t3 -= 2 * dtV.dot(mu_ij)

            # if (kindi != 4 and kindi != 5) and (kindj != 2 and kindj != 3): 
            #     H[i, j] +=  dtV.dot(sigma_j).dot(Sinv.dot(mu_i))
            #    # t4 = dtV.dot(sigma_j).dot(Sinv.dot(mu_i))
            
            # if (kindi != 2 and kindi != 3) and (kindj != 4 and kindj != 5): 
            #     H[i, j] += dtV.dot(sigma_i).dot(Sinv.dot(mu_j))
            #     #t5 = dtV.dot(sigma_i).dot(Sinv.dot(mu_j))
            if (kindi != 4) and (kindi != 5):
                if (kindj != 2) and (kindj != 3):
                    t6 = 2 * dtV.dot(sigma_i).dot(Sinv.dot(mu_j))
                else:
                    t6 = 0.0
                if (kindj != 4) and (kindj !=5 ):
                    SiSj = sigma_i.dot(Sinv).dot(sigma_j)
                    t1 =  2 * np.dot(vecVRV2, SiSj.flatten()) 
                else:
                    t1 = 0.0
            else:
                t1, t6 = 0.0, 0.0
            if (kindij != 6) and (kindij != 7):
                t2 = -np.dot(vecVRV, sigma_ij.flatten())
            else:
                t2 = 0.0
            if (kindi != 2) and (kindi != 3): 
                t3 = 2 * np.dot(mu_j, Sinv.dot(mu_i))
            else:
                t3 = 0.0
            if (kindij != 1) and (kindij != 3) and (kindij != 5):
                t4 =-2 * dtV.dot(mu_ij)
            else:
                t4 = 0.0
            if (kindj != 4) and (kindj != 5):
                if (kindi != 2) and (kindi != 3):
                    t5 = 2 * dtV.dot(sigma_j).dot(Sinv.dot(mu_i))
                else:
                    t5 = 0.0
                if (kindi != 4) and (kindi !=5):
                    t7 = 2 * dtV.dot(sigma_j).dot(Sinv).dot(sigma_i.dot(dtV))
                else:
                    t7 = 0.0
            else:
                t5, t7 = 0.0, 0.0
            if (kindij != 6) and (kindij != 7):
                t8 = -dtV.dot(sigma_ij).dot(dtV)
            else:
                t8 = 0.0
            H[i, j] += t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8
            H[j, i] =  H[i, j]
            ij += 1
    return H
        
@numba.jit(nopython=True)
def _d2loglike_mu2(H, d1Sm, L, B, F, P, a, b, Sinv, S, d, vecVRV, vecV, dA, m_size,
                d2_inds,  first_deriv_type, second_deriv_type, n, vech_inds):
    LB, BF = L.dot(B), B.dot(F)
    LBt = LB.T
    Bt = B.T
    BFBt = BF.dot(Bt)
    Bbt = B.dot(b.T)
    BFBt = BF.dot(Bt)
    LBFBt = L.dot(BFBt)
    vecVRV2 = vecVRV + vecV / 2
    mu_0 = np.zeros(a.shape)
    sigma_0 = np.zeros(S.shape)
    dtV = np.dot(d.T, Sinv)
    ij = 0
    for j in range(n):
        sigma_j, mu_j = compute_dsigma_dmu(LB, BF, LBt, LBFBt, Bbt, j, dA, first_deriv_type, m_size, sigma_0, mu_0)
        d1Sm[j, :, :-1] = sigma_j
        d1Sm[j, :, -1] = mu_j
    for j in range(n):
        sigma_j, mu_j  = d1Sm[j, :, :-1], d1Sm[j, :, -1]
        for i in range(j, n):
            sigma_i, mu_i  = d1Sm[i, :, :-1], d1Sm[i, :, -1]
            sigma_ij, mu_ij = compute_d2sigma_mu(L, B, F, LB, LBt, BF, BFBt, Bbt,
                                                 ij, dA, d2_inds, second_deriv_type, m_size,
                                                 mu_0, sigma_0)
            i, j = d2_inds[ij]
            kindij = second_deriv_type[ij]
            kindi = first_deriv_type[i]
            kindj = first_deriv_type[j]
            non_zero_sigma_i = (kindi != 4) and (kindi != 5)
            non_zero_sigma_j = (kindj != 4) and (kindj != 5)
            non_zero_mu_i = (kindi != 2) and (kindi != 3)
            non_zero_mu_j = (kindj != 2) and (kindj != 3)
            non_zero_sigma_ij = (kindij != 6) and (kindij != 7)
            non_zero_mu_ij = (kindij != 1) and (kindij != 3) and (kindij != 5)
            

            
            if non_zero_sigma_i:
                if non_zero_mu_j:
                    t6 = 2 * dtV.dot(sigma_i).dot(Sinv.dot(mu_j))
                else:
                    t6 = 0.0
                if non_zero_sigma_j:
                    SiSj = sigma_i.dot(Sinv).dot(sigma_j)
                    t1 =  2 * np.dot(vecVRV2, SiSj.flatten()) 
                else:
                    t1 = 0.0
            else:
                t1, t6 = 0.0, 0.0
            if non_zero_sigma_ij:
                t2 = -np.dot(vecVRV, sigma_ij.flatten())
            else:
                t2 = 0.0
            if non_zero_mu_i and non_zero_mu_j: 
                t3 = 2 * np.dot(mu_j, Sinv.dot(mu_i))
            else:
                t3 = 0.0
            if non_zero_mu_ij:
                t4 =-2 * dtV.dot(mu_ij)
            else:
                t4 = 0.0
            if non_zero_sigma_j:
                if non_zero_mu_i:
                    t5 = 2 * dtV.dot(sigma_j).dot(Sinv.dot(mu_i))
                else:
                    t5 = 0.0
                if non_zero_sigma_i:
                    t7 = 2 * dtV.dot(sigma_j).dot(Sinv).dot(sigma_i.dot(dtV))
                else:
                    t7 = 0.0
            else:
                t5, t7 = 0.0, 0.0
            if non_zero_sigma_ij:
                t8 = -dtV.dot(sigma_ij).dot(dtV)
            else:
                t8 = 0.0
            H[i, j] += t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8
            H[j, i] =  H[i, j]
            ij += 1
    return H
        
        
    
@numba.jit(nopython=True)
def _d2loglike_mu0(H, d1Sm, L, B, F, P, a, b, Sinv, S, d, vecVRV, vecV, dA, m_size,
                d2_inds,  first_deriv_type, second_deriv_type, n, vech_inds):
    LB, BF = L.dot(B), B.dot(F)
    LBt = LB.T
    Bt = B.T
    BFBt = BF.dot(Bt)
    Bbt = B.dot(b.T)
    BFBt = BF.dot(Bt)
    LBFBt = L.dot(BFBt)
    vecVRV2 = vecVRV + vecV / 2
    mu_0 = np.zeros(a.shape)
    sigma_0 = np.zeros(S.shape)
    dtV = np.dot(d.T, Sinv)
    ij = 0
    for j in range(n):
        sigma_j, mu_j = compute_dsigma_dmu(LB, BF, LBt, LBFBt, Bbt, j, dA, first_deriv_type, m_size, sigma_0, mu_0)
        d1Sm[j, :, :-1] = sigma_j
        d1Sm[j, :, -1] = mu_j
    for j in range(n):
        sigma_j, mu_j  = d1Sm[j, :, :-1], d1Sm[j, :, -1]
        for i in range(j, n):
            sigma_i, mu_i  = d1Sm[i, :, :-1], d1Sm[i, :, -1]
            sigma_ij, mu_ij = compute_d2sigma_mu(L, B, F, LB, LBt, BF, BFBt, Bbt,
                                                 ij, dA, d2_inds, second_deriv_type, m_size,
                                                 mu_0, sigma_0)
            i, j = d2_inds[ij]
            kindij = second_deriv_type[ij]
            kindi = first_deriv_type[i]
            kindj = first_deriv_type[j]
            SiSj = sigma_i.dot(Sinv).dot(sigma_j)
            t1 = 2 * np.dot(vecVRV2, SiSj.flatten()) 
            t2 =  -np.dot(vecVRV, sigma_ij.flatten())
            t3 = 2 * np.dot(mu_j, Sinv.dot(mu_i)) - 2 * dtV.dot(mu_ij)
            t4 = 2*dtV.dot(sigma_j).dot(Sinv.dot(mu_i))
            t5 = 2*dtV.dot(sigma_i).dot(Sinv.dot(mu_j))
            t6 = 2 * dtV.dot(sigma_j).dot(Sinv).dot(sigma_i.dot(dtV))- dtV.dot(sigma_ij).dot(dtV)
            H[i, j] += t1 + t2 + t3 + t4 + t5 + t6
            H[j, i] =  H[i, j]
            ij += 1
    return H
        

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