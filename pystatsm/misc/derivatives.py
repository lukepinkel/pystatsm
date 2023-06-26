#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 01:53:02 2023

@author: lukepinkel
"""

import numba
import numpy as np

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
def _dloglike_mu(g, L, B, F, b, a, VRV, rtV, dA, m_size, m_type, n, p2):
    LB, BF = L.dot(B), B.dot(F)
    Bbt = B.dot(b.T)
    LBt, BFBt = LB.T, BF.dot(B.T)
    LB, BF = L.dot(B), B.dot(F)
    Bbt = B.dot(b.T)
    LBt, BFBt = LB.T, BF.dot(B.T)
    vecBFBt = BFBt.flatten()
    vecVRV = VRV.flatten()
    VRVL = VRV.dot(L)
    rtVL = rtV.dot(L)
    for i in range(n):
        kind = m_type[i]
        J = dA[i, :m_size[i, 0], :m_size[i, 1]]
        if kind == 0:
            dmi = J.dot(Bbt)
            g[i] += -2*np.dot((J.T.dot(VRVL)).flatten(), vecBFBt)
            g[i] += -(2.0 * np.dot(dmi.T, rtV) + 2*np.dot(rtVL, BFBt.dot(rtV.dot(J))))
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
        elif kind == 3:
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
def _d2loglike_mu(H, d1Sm, L, B, F, P, a, b, VRV, rtV, V, dA, m_size,
                   d2_inds,  first_deriv_type, second_deriv_type, n, vech_inds):
    LB, BF = L.dot(B), B.dot(F)
    LBt = LB.T
    Bt = B.T
    BFBt = BF.dot(Bt)
    Bbt = B.dot(b.T)
    BFBt = BF.dot(Bt)
    LBFBt = L.dot(BFBt)
    vecVRV = VRV.flatten()
    vecVRV2 = vecVRV + V.flatten() / 2
    mu_0 = np.zeros(a.shape)
    sigma_0 = np.zeros(V.shape)
    ij = 0
    for j in range(n):
        kind = first_deriv_type[j]
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
        d1Sm[j, :, :-1] = sigma_j
        d1Sm[j, :, -1] = mu_j

    for j in range(n):
        sigma_j, mu_j  = d1Sm[j, :, :-1], d1Sm[j, :, -1]
        for i in range(j, n):
            sigma_i, mu_i  = d1Sm[i, :, :-1], d1Sm[i, :, -1]
            i, j = d2_inds[ij]
            kindij = second_deriv_type[ij]
            kindi = first_deriv_type[i]
            kindj = first_deriv_type[j]
            Ji =dA[i, :m_size[i, 0], :m_size[i, 1]]
            Jj =dA[j, :m_size[j, 0], :m_size[j, 1]]
            if kindij == 1:
                JiBFBtJjt = Ji.dot(BFBt).dot(Jj.T)
                sigma_ij = (JiBFBtJjt + JiBFBtJjt.T)  #0, 0 L ,L
                mu_ij = mu_0
            elif kindij == 2:
                BJj = B.T.dot(Jj.T)                                      #1,0, B, L
                JiBF = Ji.dot(BF)
                C = JiBF + JiBF.T
                D = LB.dot(C).dot(BJj)
                sigma_ij = D + D.T
                mu_ij = Jj.dot(B).dot(Ji).dot(Bbt)
            elif kindij == 3:
                JjB = Jj.dot(B)                                          #2,0, F, L
                C = JjB.dot(Ji).dot(LBt)
                sigma_ij = C + C.T
                mu_ij = mu_0
            elif kindij == 4:
                C1 = Ji.dot(BF)                                          #1,1, B, B
                C1 = C1 + C1.T
                JiB, JjB = Ji.dot(B), Jj.dot(B)
                t1 = JjB.dot(C1)
                t3 = JiB.dot(JjB.dot(F))
                t4 = F.T.dot(JjB.T).dot(JiB.T)
                sigma_ij = LB.dot(t1 + t1.T + t3 + t4).dot(LBt)
                mu_ij = (LB.dot(JjB.dot(Ji) + JiB.dot(Jj)).dot(Bbt)).flatten() #LB.dot(Ji).dot(B).dot(Jj).dot(Bbt)
            elif kindij == 5:
                C = Jj.dot(B).dot(Ji)                                    #2,1, F, B
                sigma_ij  = LB.dot(C+C.T).dot(LBt)
                mu_ij = mu_0
            elif kindij == 6:
                mu_ij = Jj.dot(B.dot(Ji[0])).flatten()                            #5, 0, b, L
                sigma_ij = sigma_0
            elif kindij == 7:
                mu_ij = LB.dot(Jj).dot(B).dot(Ji.T).flatten()                          #5, 1, b,
                sigma_ij = sigma_0
            else:
                sigma_ij = sigma_0
                mu_ij= mu_0

            if (kindi != 4) and (kindi != 5):
                if (kindj != 2) and (kindj != 3):
                    t6 = 2 * rtV.dot(sigma_i).dot(V.dot(mu_j))
                else:
                    t6 = 0.0
                if (kindj != 4) and (kindj !=5 ):
                    SiSj = sigma_i.dot(V).dot(sigma_j)
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
                t3 = 2 * np.dot(mu_j, V.dot(mu_i))
            else:
                t3 = 0.0
            if (kindij != 1) and (kindij != 3) and (kindij != 5):
                t4 =-2 * rtV.dot(mu_ij)
            else:
                t4 = 0.0
            if (kindj != 4) and (kindj != 5):
                if (kindi != 2) and (kindi != 3):
                    t5 = 2 * rtV.dot(sigma_j).dot(V.dot(mu_i))
                else:
                    t5 = 0.0
                if (kindi != 4) and (kindi !=5):
                    t7 = 2 * rtV.dot(sigma_j).dot(V).dot(sigma_i.dot(rtV))
                else:
                    t7 = 0.0
            else:
                t5, t7 = 0.0, 0.0
            if (kindij != 6) and (kindij != 7):
                t8 = -rtV.dot(sigma_ij).dot(rtV)
            else:
                t8 = 0.0
            H[i, j] += t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8
            H[j, i] =  H[i, j]
            ij += 1
    return H
