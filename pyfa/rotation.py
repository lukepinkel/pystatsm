#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 22:04:25 2020

@author: lukepinkel
"""

import numpy as np
import pandas as pd

def vgq_ortho(L, gamma):
    '''
    VgQ subroutine for orthogonal rotation
    
    Parameters:
        L: Loadings matrix of p features and q factors
        gamma: Coefficient that determines the type of rotation
    
    Returns:
        ft: Criteria function
        Gq: Gradient of the function at L
    '''
    p, q = L.shape
    L2 = L**2
    I, C = np.eye(p), np.ones((p, p))/p
    H = (I - gamma*C).dot(L2)
    ft = np.sum(L2*H) * 0.25
    Gq = L * H
    return -ft, -Gq

    
def vgq_obli(L, gamma):
    '''
    VgQ subroutine for oblique rotation
    
    Parameters:
        L: Loadings matrix of p features and q factors
        gamma: Coefficient that determines the type of rotation
    
    Returns:
        ft: Criteria function
        Gq: Gradient of the function at L
    '''
    p, q = L.shape
    L2 = L**2
    I, C, N = np.eye(p), np.ones((p, p))/p, np.ones((q, q)) - np.eye(q)
    H = np.linalg.multi_dot([(I - gamma*C), L2, N])
    ft = np.sum(L2*H) * 0.25
    Gq = L * H
    return ft, Gq

def rotate_ortho(A, T=None, alpha=1.0, gamma=0, tol=1e-9, n_iters=1000):
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
    ft, Gq = vgq_ortho(L, gamma)
    G = np.dot(A.T, Gq)
    for i in range(n_iters):
        M = np.dot(T.T, G)
        S = (M + M.T) / 2.0
        Gp = G - np.dot(T, S)
        s = np.linalg.norm(Gp)
        if s<tol:
            break
        alpha = 2.0 * alpha
        for c in range(10):
            X = T - alpha * Gp
            U, D, V = np.linalg.svd(X, full_matrices=False)
            Tt = np.dot(U, V)
            L = np.dot(A, Tt)
            ft_new, Gq = vgq_ortho(L, gamma)
            if ft_new < (ft - 0.5*s**2*alpha):
                break
            else:
                alpha = alpha * 0.5
        ft, T =ft_new, Tt
        G = np.dot(A.T, Gq)
    return T


def rotate_obli(A, T=None, alpha=1.0, gamma=0, tol=1e-9, n_iters=500):
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
    ft, Gq = vgq_obli(L, gamma)
    G = -np.linalg.multi_dot([L.T, Gq, Tinv]).T
    for i in range(n_iters):
        TG = T*G
        Gp = G - np.dot(T, np.diag(np.sum(TG, axis=0)))
        s = np.linalg.norm(Gp)
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
            ft_new, Gq = vgq_obli(L, gamma)
            if ft_new < (ft - 0.5*s**2*alpha):
                break
            else:
                alpha = alpha * 0.5
        ft, T =ft_new, Tt
        G = -np.linalg.multi_dot([L.T, Gq, Tinv]).T
    return T

def rotate(A, method, T=None, tol=1e-9, alpha=1.0,
           n_iters=500, custom_gamma=None, k=4):
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
    if type(A) is pd.DataFrame:
        ix, cols = A.index, A.columns
    if method == 'quartimax':
        gamma = 0
        rotation_type = 'orthogonal'
    if method == 'biquartimax':
        gamma = 0.5
        rotation_type = 'orthogonal'
    if method == 'varimax':
        gamma = 1.0
        rotation_type = 'orthogonal'
    if method == 'equamax':
        gamma = A.shape[0] / 2
        rotation_type = 'orthogonal'
    if method == 'promax':
        gamma = 1.0
        rotation_type = 'orthogonal'
    if method == 'quartimin':
        gamma = 0.0
        rotation_type = 'oblique'
    if method == 'biquartimin':
        gamma = 0.5
        rotation_type = 'oblique'
    if method == 'covarimin':
        gamma = 1.0
        rotation_type = 'oblique'
    if method == 'oblique':
        if custom_gamma is None:
            gamma = -0.1
        else:
            gamma = custom_gamma
        rotation_type = 'oblique'
        
    if rotation_type == 'orthogonal':
        gcff = (1, gamma-1, -gamma,0)
        T = rotate_ortho(A, T=T, alpha=alpha, gamma=gamma, tol=tol, n_iters=n_iters)
        L = np.dot(A, T)
        if method == 'promax':
            H = abs(L)**k/L
            V = np.linalg.multi_dot([np.linalg.pinv(np.dot(A.T, A)), A.T, H])
            D = np.diag(np.sqrt(np.diag(np.linalg.inv(np.dot(V.T, V)))))
            T = np.linalg.inv(np.dot(V, D)).T
            L = np.dot(A, T)
            
    elif rotation_type == 'oblique':
        gcff = (-gamma / A.shape[0], 1, gamma/A.shape[0], -1)
        T = rotate_obli(A, T=T, alpha=alpha, gamma=gamma, tol=tol, n_iters=n_iters)
        L = np.dot(A, np.linalg.inv(T).T)
    if type(A) is pd.DataFrame:
        L = pd.DataFrame(L, index=ix, columns=cols)
    
    return L, T, gcff