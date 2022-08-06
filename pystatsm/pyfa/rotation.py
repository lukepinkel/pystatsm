#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 22:04:25 2020

@author: lukepinkel
"""

import numpy as np
import pandas as pd
from ..utilities.linalg_operations import vec, invec, vecl, vdg
from ..utilities.special_mats import kmat, nmat, lmat


class RotationMethod(object):
    
    def __init__(self, A, rotation_type="ortho"):
        self.p, self.m = A.shape
        self.A = A
        self.rotation_type = rotation_type
        self.Kmp = kmat(self.m, self.p)
        if self.rotation_type == "oblique":
            self.lix = vec(np.eye(self.m)!=1)
            self.cix = vec(np.tril(np.ones(self.m), -1)!=0)
        elif self.rotation_type == "ortho":
            self.lix = vec(np.tril(np.ones((self.m, self.m)), -1)!=0)
    
    def rotate(self, T):
        if self.rotation_type == "ortho":
            L = np.dot(self.A, T)
        else:
            L = np.dot(self.A, np.linalg.inv(T.T))
        return L

    
    def _d_rotate_oblique(self, T):
        A = self.A
        B = np.linalg.inv(T.T)
        L = np.dot(A, B)
        J = -self.Kmp.dot(np.kron(L, B.T))#J = np.kron(B.T, L)
        return J
    
    def _d_rotate_ortho(self, T):
        J = np.kron(np.eye(self.m), self.A)
        return J
    
    def d_rotate(self, T):
        if self.rotation_type == "ortho":
            J = self._d_rotate_ortho(T)
        else:
            J = self._d_rotate_oblique(T)
        return J
    
    def f(self, T):
        L = self.rotate(T)
        return self.Q(L)
    
    def df(self, T):
        L = self.rotate(T)
        Gq = self.dQ(L)
        dR= self.d_rotate(T)
        J = invec(vec(Gq).dot(dR), *T.shape)
        return J
    
    def _rotate_ortho(self, A, vgq, T=None, alpha=1.0, tol=1e-9, n_iters=1000):
        if T is None:
            T = np.eye(A.shape[1])
        L = np.dot(A, T)
        ft, Gq = vgq(L)
        G = np.dot(A.T, Gq)
        opt_hist = []
        alpha0 = alpha
        for i in range(n_iters):
            M = np.dot(T.T, G)
            S = (M + M.T) / 2.0
            Gp = G - np.dot(T, S)
            s = np.linalg.norm(Gp)
            opt_hist.append([ft, s])
            if s<tol:
                break
            alpha = 2.0 * alpha0
            for c in range(10):
                X = T - alpha * Gp
                U, D, V = np.linalg.svd(X, full_matrices=False)
                Tt = np.dot(U, V)
                L = np.dot(A, Tt)
                ft_new, Gq = vgq(L)
                if ft_new < (ft - 0.5*s**2*alpha):
                    break
                else:
                    alpha = alpha * 0.5
            #if abs(ft - ft_new) < 1e-16:
            #    break
            ft, T =ft_new, Tt
            G = np.dot(A.T, Gq)
        return T, G, Gq, opt_hist
    
    def _rotate_obli(self,  A, vgq, T=None, alpha=1.0, tol=1e-9, n_iters=500):
        if T is None:
            T = np.eye(A.shape[1])
        Tinv = np.linalg.inv(T)
        L = np.dot(A, Tinv.T)
        ft, Gq = vgq(L)
        G = -np.linalg.multi_dot([L.T, Gq, Tinv]).T
        opt_hist = []
        for i in range(n_iters):
            TG = T*G
            Gp = G - np.dot(T, np.diag(np.sum(TG, axis=0)))
            s = np.linalg.norm(Gp)
            opt_hist.append([ft, s])
            if s<tol:
                break
            alpha = 2.0 * alpha
            for c in range(10):
                X = T - alpha * Gp
                X2 = X**2
                V = np.diag(1 / np.sqrt(np.sum(X2, axis=0)))
                Tt = np.dot(X, V)
                Tinv = np.linalg.pinv(Tt)
                L = np.dot(A, Tinv.T)
                ft_new, Gq = vgq(L)
                if ft_new < (ft - 0.5*s**2*alpha):
                    break
                else:
                    alpha = alpha * 0.5
            if abs(ft - ft_new) < 1e-16:
                break
            ft, T =ft_new, Tt
            G = -np.linalg.multi_dot([L.T, Gq, Tinv]).T
        return T, G, Gq, opt_hist
    
    def fit(self, opt_kws=None):
        opt_kws = {} if opt_kws is None else opt_kws
        if self.rotation_type == "ortho":
            T, G, Gq, opt_hist = self._rotate_ortho(self.A, self.vgQ, **opt_kws)
        else:
            T, G, Gq, opt_hist = self._rotate_obli(self.A, self.vgQ, **opt_kws)
        self.T, self.G, self.Gq, self.opt_hist = T, G, Gq, opt_hist
        
    def oblique_constraints(self, L, Phi):
        dQ = self.dQ(L)
        C = np.dot(L.T, dQ).dot(np.linalg.inv(Phi))
        C[np.diag_indices_from(C)] = 1.0
        return C
    
    def ortho_constraints(self, L, Phi):
        dQ = self.dQ(L)
        C = np.dot(L.T, dQ) - np.dot(dQ.T, L)
        return C
    
    def constraints(self, L, Phi):
        if self.rotation_type == "ortho":
            C = vecl(self.ortho_constraints(L, Phi))
        elif self.rotation_type == "oblique":
            C = vec(self.oblique_constraints(L, Phi))[self.lix]
        return C
    
    def unrotated_constraints(self, L, Phi, Psi):
        Lm = lmat(self.m)
        Nm = nmat(self.m)
        Im = np.eye(self.m)
        d_inds = vec(np.eye(self.p))==1
        Psi_inv = np.diag(1.0 / np.diag(Psi))
        LtP = L.T.dot(Psi_inv)
        J1 = Lm.dot(Nm.dot(np.kron(Im, LtP)))
        J2 =-Lm.dot(np.kron(LtP, LtP)[:, d_inds])
        J = np.concatenate([J1, J2], axis=1)
        i, j = np.tril_indices(self.p)
        J = J[i!=j]
        return J
               
    
def get_gcf_constants(method, p, m):
    if method == 'varimax':
        consts = dict(k1=0.0, k2=(p-1)/p, k3=1/p, k4=-1)
    elif method == 'quartimax':
        consts = dict(k1=0.0, k2=1, k3=1, k4=-1)
    elif method == 'equamax':
        consts = dict(k1=0.0, k2=1-m/(2.0 * p), k3=m/(2.0 * p), k4=-1)
    elif method == 'parsimax':
        consts = dict(k1=0.0, k2=1-(m-1)/(p+m-2), k3=(m-1)/(p+m-2), k4=-1)
    return consts

class GeneralizedCrawfordFerguson(RotationMethod):
    def __init__(self, A, k1, k2, k3, k4, rotation_type="ortho"):
        super().__init__(A, rotation_type)
        self.k1, self.k2, self.k3, self.k4 = k1, k2, k3, k4
        
    def Q(self, L):
        B = L**2
        f1 = self.k1 * np.sum(B)**2 
        f2 = self.k2 * np.sum(np.sum(B, axis=1)**2)
        f3 = self.k3 * np.sum(np.sum(B, axis=0)**2)
        f4 = self.k4 * np.sum(B * B)
        f = (f1 + f2 + f3 + f4) / 4
        return f
    
    def dQ(self, L):
        B = L**2
        dQ1 = self.k1 * np.sum(B) #self.k1 * Jp.dot(B).dot(Jm) 
        dQ2 = self.k2 * np.sum(B, axis=1).reshape(-1, 1) #self.k2 * (B).dot(Jm) 
        dQ3 = self.k3 * np.sum(B, axis=0).reshape(1, -1) #self.k3 * (Jp).dot(B) 
        dQ4 = self.k4 * B #self.k4 * B
        dQ = (dQ1 + dQ2 + dQ3 + dQ4) * L 
        return dQ
    
    def vgQ(self, L):
        fq = self.Q(L)
        Gq = self.dQ(L)
        return fq, Gq
    
    def dC_dL_Obl(self, L, Phi):
        B = L * L 
        dQ1 = self.k1 * np.sum(B)
        dQ2 = self.k2 * np.sum(B, axis=1).reshape(-1, 1)
        dQ3 = self.k3 * np.sum(B, axis=0).reshape(1, -1)
        dQ4 = self.k4 * B
        W = (dQ1 + dQ2 + dQ3 + dQ4) 
        Q = L * W
        V = np.linalg.inv(Phi)
        p,  m = self.p, self.m
        QV = np.dot(Q, V)
        LV = np.dot(L, V)
        LtLV = np.dot(L.T, LV)
        LtL = np.dot(L.T, L)
        J = np.zeros((m, m, p, m))
        for v, u in np.ndindex((m, m)):
            for r, i in np.ndindex((m, p)):
                if v!=u:
                    t1 = (u==r) * QV[i, v]
                    t2 = W[i, r] * L[i, u] * V[r, v]
                    t3 = 2.0 * self.k1 * L[i, r] * LtLV[u, v]
                    t4 = 2.0 * self.k2 * L[i, r] * L[i, u] * LV[i, v]
                    t5 = 2.0 * self.k3 * L[i, r] * LtL[u, r] * V[r ,v]
                    t6 = 2.0 * self.k4 * L[i, r] * L[i, r] * L[i, u] * V[r, v]
                    J[u, v, i, r] = t1 + t2 + t3 + t4 + t5 + t6
        return J
    
    def dL_dA_Ortho(self, L, Phi):
        AAt = np.dot(self.A, self.A.T)
        B = L * L
        k1, k2, k3, k4, p, m = self.k1, self.k2, self.k3, self.k4, self.p, self.m
        Jm, Jp = np.ones((m, m)), np.ones((p, p))
        Ip, Im, Kpm = np.eye(p), np.eye(m), kmat(p, m).A
        T1 = k1 * Jp.dot(B).dot(Jm) 
        T2 = k2 * (B).dot(Jm) 
        T3 = k3 * (Jp).dot(B) 
        T4 = k4 * B
        Q = (T1 + T2 + T3 + T4) * L 
        V = vdg(L)
        R1 = k1 * np.kron(Jm, Jp)
        R2 = k2 * np.kron(Jm, Ip)
        R3 = k3 * np.kron(Im, Jp)
        R4 = k4 * V
        D1 = V.dot(2.0 * (R1 + R2 + R3).dot(V) + 2.0 * R4)
        D2 = vdg(T1 + T2 + T3 + T4)
        Y = D1 + D2
        Omega = np.kron(L.T, L).dot(Kpm) - np.kron(Im, AAt)
        Xi = np.kron(L.T.dot(Q), Ip) + np.kron(Im, L.dot(Q.T))
        Gamma = np.kron(Q.T.dot(self.A), Ip) + np.kron(Q.T, self.A).dot(Kpm)
        G = np.linalg.solve(np.dot(Omega, Y) + Xi, Gamma)
        return G
    
    def dC_dL_Ortho(self, L, Phi):
        B = L * L
        k1, k2, k3, k4, p, m = self.k1, self.k2, self.k3, self.k4, self.p, self.m
        Jm, Jp = np.ones((m, m)), np.ones((p, p))
        Ip, Im, Kpm = np.eye(p), np.eye(m), kmat(p, m)
        T1 = k1 * Jp.dot(B).dot(Jm) 
        T2 = k2 * (B).dot(Jm) 
        T3 = k3 * (Jp).dot(B) 
        T4 = k4 * B
        Q = (T1 + T2 + T3 + T4) * L
        V = vdg(L)
        R1 = k1 * np.kron(Jm, Jp)
        R2 = k2 * np.kron(Jm, Ip)
        R3 = k3 * np.kron(Im, Jp)
        R4 = k4 * V
        D1 = V.dot(2.0 * (R1 + R2 + R3).dot(V) + 2.0 * R4)
        D2 = vdg(T1 + T2 + T3 + T4)
        DQ = D1 + D2
        HQ1 = np.kron(Q.T, Im).dot(Kpm.A) + np.kron(Im, L.T).dot(DQ)
        HQ2 = np.kron(L.T, Im).dot(Kpm.A).dot(DQ) + np.kron(Im, Q.T)
        HQ = HQ1 - HQ2
        return HQ
         
    def dC_dP_Obl(self, L, Phi):
        B = L * L 
        dQ1 = self.k1 * np.sum(B)
        dQ2 = self.k2 * np.sum(B, axis=1).reshape(-1, 1)
        dQ3 = self.k3 * np.sum(B, axis=0).reshape(1, -1)
        dQ4 = self.k4 * B
        W = (dQ1 + dQ2 + dQ3 + dQ4) 
        V = np.linalg.inv(Phi)
        Q = L * W
        C = np.dot(L.T, Q).dot(V)
        J = np.zeros((self.m, self.m, self.m, self.m))
        for v, u in np.ndindex((self.m, self.m)):
            for x, y in np.ndindex((self.m, self.m)):
                if (v!=u) and (x!=y):
                    t1 = (u==x) * V[y, v]
                    t2 = (u==y) * V[x, v]
                    J[u, v, x, y] = -C[u, u] * (t1 + t2)
        return J
    
    
class TargetRotation(RotationMethod):
    def __init__(self, A, H, W=None, rotation_type="ortho"):
        W = np.ones_like(A) if W is None else W
        super().__init__(A, rotation_type)
        self.W, self.H = W, H
    
    def Q(self, L):
        f = np.sum((self.W * (L - self.H))**2) / 2
        return f
    
    def dQ(self, L):
        dQ = self.W * (L - self.H)
        return dQ
    
    def vgQ(self, L):
        Gq = self.W * (L - self.H)
        fq = np.sum(Gq**2) / 2
        return fq, Gq
                   


def vgq_cf(L, gamma):
    '''
    VgQ subroutine for crawford ferguson
    
    Parameters:
        L: Loadings matrix of p features and q factors
        gamma: Coefficient that determines the type of rotation
    
    Returns:
        ft: Criteria function
        Gq: Gradient of the function at L
    '''
    p, q = L.shape
    L2 = L**2
    N = np.ones((q, q)) - np.eye(q)
    M = np.ones((p, p)) - np.eye(p)
    f1 = (1 - gamma) * np.trace(np.dot(L2.T, L2.dot(N))) / 4.0
    f2 = gamma * np.trace(np.dot(L2.T, M.dot(L2))) / 4.0
    G1 = (1.0 - gamma) * L * (L2.dot(N)) 
    G2 = gamma * L * (M.dot(L2))
    ft = f1 + f2
    Gq = G1 + G2
    return ft, Gq

    
def vgq_ob(L, gamma):
    '''
    VgQ subroutine for oblimin rotation
    
    Parameters:
        L: Loadings matrix of p features and q factors
        gamma: Coefficient that determines the type of rotation
    
    Returns:
        ft: Criteria function
        Gq: Gradient of the function at L
    '''
    p, q = L.shape
    L2 = L**2
    C = np.eye(p) -  np.ones((p, p)) * gamma / p
    N = np.ones((q, q)) - np.eye(q)
    B = C.dot(L2.dot(N))
    ft = np.trace(np.dot(L2.T, B)) / 4.0
    Gq = L * B
    return ft, Gq

def rotate_ortho(A, vgq, T=None, alpha=1.0, gamma=0, tol=1e-9, n_iters=1000):
    '''
    Orthogonal rotation
    
    Parameters:
        A: Loadings matrix
        T: Initial rotation matrix
        alpha: Coefficient that determines the step size
        gamma: Coefficient that determines the type of rotation
        tol: Tolerance that determines convergance
        n_iters: The maximum number of iterations before stopping
    
    Returns:
        T: Rotation matrix
    '''
    if T is None:
        T = np.eye(A.shape[1])
    L = np.dot(A, T)
    ft, Gq = vgq(L, gamma)
    G = np.dot(A.T, Gq)
    opt_hist = []
    alpha0 = alpha
    for i in range(n_iters):
        M = np.dot(T.T, G)
        S = (M + M.T) / 2.0
        Gp = G - np.dot(T, S)
        s = np.linalg.norm(Gp)
        opt_hist.append([ft, s])
        if s<tol:
            break
        alpha = 2.0 * alpha0
        for c in range(10):
            X = T - alpha * Gp
            U, D, V = np.linalg.svd(X, full_matrices=False)
            Tt = np.dot(U, V)
            L = np.dot(A, Tt)
            ft_new, Gq = vgq(L, gamma)
            if ft_new < (ft - 0.5*s**2*alpha):
                break
            else:
                alpha = alpha * 0.5
        ft, T =ft_new, Tt
        G = np.dot(A.T, Gq)
    return T, G, Gq, opt_hist



def rotate_obli(A, vgq, T=None, alpha=1.0, gamma=0, tol=1e-9, n_iters=500):
    '''
    Oblique rotation
    
    Parameters:
        A: Loadings matrix
        T: Initial rotation matrix
        alpha: Coefficient that determines the step size
        gamma: Coefficient that determines the type of rotation
        tol: Tolerance that determines convergance
        n_iters: The maximum number of iterations before stopping
    
    Returns:
        T: Rotation matrix
    '''
    if T is None:
        T = np.eye(A.shape[1])
    Tinv = np.linalg.inv(T)
    L = np.dot(A, Tinv.T)
    ft, Gq = vgq(L, gamma)
    G = -np.linalg.multi_dot([L.T, Gq, Tinv]).T
    opt_hist = []
    for i in range(n_iters):
        TG = T*G
        Gp = G - np.dot(T, np.diag(np.sum(TG, axis=0)))
        s = np.linalg.norm(Gp)
        opt_hist.append([ft, s])
        if s<tol:
            break
        alpha = 2.0 * alpha
        for c in range(10):
            X = T - alpha * Gp
            X2 = X**2
            V = np.diag(1 / np.sqrt(np.sum(X2, axis=0)))
            Tt = np.dot(X, V)
            Tinv = np.linalg.pinv(Tt)
            L = np.dot(A, Tinv.T)
            ft_new, Gq = vgq(L, gamma)
            if ft_new < (ft - 0.5*s**2*alpha):
                break
            else:
                alpha = alpha * 0.5
        ft, T =ft_new, Tt
        G = -np.linalg.multi_dot([L.T, Gq, Tinv]).T
    return T.T, G, Gq, opt_hist

def get_gamma_ob(criterion, p, custom_gamma):
    if (criterion == 'quartimin') or (criterion == 'quartimax'):
        gamma = 0.0
    elif (criterion == 'biquartimin') or (criterion == 'biquartimax'):
        gamma = 1.0 / 2.0
    elif (criterion == 'varimax') or (criterion == 'covarimin'):
        gamma = 1.0
    elif (criterion == 'equamax'):
        gamma = p / 2.0
    elif (criterion == 'oblique'):
        gamma = custom_gamma if custom_gamma is not None else -0.1
    else:
        gamma = 0.0
    return gamma
        
def get_gamma_cf(criterion, p, k):
    if (criterion == 'quartimax'):
        kappa = 0.0
    elif (criterion == 'varimax'):
        kappa = 1.0 / p
    elif (criterion == 'equamax'):
        kappa = k / (2.0 * p)
    elif (criterion == 'parsimax'):
        kappa = (k - 1.0) / (p + k - 2.0)
    elif (criterion == 'parsimony'):
        kappa = 1
    else:
        kappa = 0.0
    return kappa
        


#TODO remove figure out which of these are redudant/used for development
def jac_approx(f, x, eps=1e-4, tol=None, d=1e-4, args=()):
    tol = np.finfo(float).eps**(1/3) if tol is None else tol
    h = np.abs(d * x) + eps * (np.abs(x) < tol)
    n = len(x)
    m = len(f(x, *args))
    u = np.zeros_like(h)
    J = np.zeros((n, m))
    for i in range(n):
        u[i] = h[i]
        J[i] = (f(x + u, *args) - f(x - u, *args)) / (2.0 * h[i])
        u[i] = 0.0
    return J


def oblique_constraints(lvec, tvec, p, q, gamma, vgq):
    L = invec(lvec, p, q)
    T = invec(tvec, q, q)
    _, Gq = vgq(L, gamma)
    p, q  = L.shape
    I = np.eye(q)
    N = np.ones((q, q)) - I
    Phi = np.dot(T, T.T)
    J1 = L.T.dot(Gq).dot(np.linalg.inv(Phi)) * N
    #J2 = Phi * I
    J = J1 #+ J2 - I
    return vec(J)


def oblique_constraint_func(params, model):
    L, Phi, Psi = model.model_matrices_augmented(params)
    T = model.T
    _, Gq = model._vgq(L, model._gamma)
    p, q  = L.shape
    I = np.eye(q)
    N = np.ones((q, q)) - I
    Phi = np.dot(T, T.T)
    J1 = L.T.dot(Gq).dot(np.linalg.inv(Phi)) * N
    return vec(J1)


def oblique_constraint_derivs(params, model):
    """
    Derivatives of the Oblique Constraints 
    
    Parameters
    ----------
    params: ndarray
            vector containing model parameters
        
    model: FactorAnalysis object
        The factor model on which rotation is being performed
    
    Returns
    -------
    
    D: ndarray 
        Derivative of the oblique constraint matrix organized in block
    
    
    Oblique constraints can be expressed
    
        \Lambda^{T} \frac{dQ}{d\Lambda}\Phi^{-1}
    
    Where Lambda is the loadings matrix, Q is the rotation criterion,
    \frac{dQ}{d\Lambda} is the gradient of the rotation criterion wrt \Lambda,
    and \Phi^{-1} is the inverse of the implied factor covariance.
    """
    L, Phi, Psi = model.model_matrices_augmented(params)
    V = np.linalg.inv(Phi)
    gamma = model._gamma
    p, q  = L.shape
    Iq = np.eye(q)
    Kpq = kmat(p, q).A
    L2 = L**2
    A = np.eye(p) -  np.ones((p, p)) * gamma / p
    B = np.ones((q, q)) - np.eye(q)
    G = L * A.dot(L2).dot(B)
    dgL = vdg(L)
    DL1A = np.kron(V.T.dot(G.T), Iq).dot(Kpq)
    DL2A = np.kron(V.T, L.T)
    DL2B = 2.0 * dgL.dot(np.kron(B.T, A)).dot(dgL) + vdg(A.dot(L2).dot(B))
    
    DL = DL1A + DL2A.dot(DL2B)
    DPhi = -np.kron(V.T, L.T.dot(G).dot(V))
    l_ind = vecl(np.arange(q*q).reshape(q, q, order='F'))
    
    D = np.concatenate([DL, DPhi[:, l_ind], np.zeros((DL.shape[0], p))], axis=1)
    return D


def approx_oblique_constraint_derivs(params, model):
    J = jac_approx(oblique_constraint_func, params, args=(model,))
    return J

    
def rotate(A, criterion, method='oblimin', rotation_type='oblique', T=None, 
           tol=1e-8, alpha=1.0, n_iters=500, custom_gamma=None):
    '''
    Rotation of loadings matrix
    
    Parameters:
        A: Loadings Matrix
        method: Type of rotation
        T: Initial rotation matrix
        tol: Tolerance controlling convergance
        alpha: Parameter controlling step size taken in GPA algorithm
        n_iters: Maximum number of iterations before convergance
        custom_gamma: Coefficient used to customize non standard oblique rotations
    
    Returns:
        L: Rotated loadings matrix
        T: Rotation matrix
    
    Methods are:
        quartimax
        biquartimax
        varimax
        equamax
        quartimin
        biquartimin
        covarimin
        oblique
    
    '''
    p, k = A.shape
    if method == 'oblimin':
        gamma, vgq = get_gamma_ob(criterion, p, custom_gamma), vgq_ob
    elif method == 'cf':
        gamma, vgq = get_gamma_cf(criterion, p, k), vgq_cf
    
    
    if rotation_type == 'orthogonal':
        T, G, Gq, opt_hist = rotate_ortho(A, vgq, T=T, alpha=alpha, gamma=gamma, 
                                          tol=tol, n_iters=n_iters)
        L = np.dot(A, T)
  
    elif rotation_type == 'oblique':
        T, G, Gq, opt_hist = rotate_obli(A, vgq, T=T, alpha=alpha, gamma=gamma,
                                         tol=tol, n_iters=n_iters)
        L = np.dot(A, np.linalg.inv(T))

    return L, T, G, Gq, opt_hist, vgq, gamma













