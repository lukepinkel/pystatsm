#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 22:04:25 2020

@author: lukepinkel
"""

import numpy as np
import pandas as pd
from ..utilities.linalg_operations import vec, invec, vecl, vdg
from ..utilities.special_mats import kmat




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
    for i in range(n_iters):
        M = np.dot(T.T, G)
        S = (M + M.T) / 2.0
        Gp = G - np.dot(T, S)
        s = np.linalg.norm(Gp)
        opt_hist.append([ft, s])
        if s<tol:
            break
        alpha = 2.0 * alpha
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













