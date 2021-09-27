# -*- coding: utf-8 -*-
"""
Created on Sun Sep 26 23:45:50 2021

@author: lukepinkel
"""
import numpy as np

def soft_threshold(x, t):
    s = np.sign(x) * np.maximum(np.abs(x) - t, 0)
    return s

def lasso_obj(beta, X, y, lambda_):
    n = X.shape[0]
    r = y - X.dot(beta)
    f = np.dot(r, r) / (2.0 * n)
    h = lambda_ * np.sum(np.abs(beta))
    return f, h, f+h

def lasso_prox_grad(beta, X, y, Xty, XtX, lambda_, max_eig, n_iters=500, 
                           rtol=1e-5, atol=1e-8):
    n = y.shape[0]
    #stepsize in (0, 1/ L] ----> stepsize = midpoint (0+1/L)/2=1/2L
    alpha = 1.0 / (2 * max_eig) 
    tau = alpha * lambda_
    fhist = np.array([lasso_obj(beta, X, y, lambda_ )])
    for i in range(n_iters):
        g = (Xty - XtX.dot(beta)) / n
        beta_new = soft_threshold(beta + alpha * g, tau)
        fhist = np.append(fhist, np.array([lasso_obj(beta_new, X, y, lambda_)]), axis=0)
        if fhist[i+1, 2] > fhist[i, 2]:
            break
        dt = np.abs(beta_new - beta)
        bt = np.abs(beta) 
        if np.all(dt < bt * rtol) or np.all(dt < bt * rtol + atol):
            beta = beta_new
            break
        else:
            beta = beta_new
    return beta, fhist


def lasso_accelerated_grad(beta, X, y, Xty, XtX, lambda_, max_eig, n_iters=500, 
                           rtol=1e-5, atol=1e-8):
    n = y.shape[0]
    alpha = 1.0 / (2 * max_eig) 
    tau = alpha * lambda_
    fhist = np.array([lasso_obj(beta, X, y, lambda_ )])
    theta = beta.copy()*0.0
    for i in range(n_iters):
        g = (Xty - XtX.dot(theta)) / n
        beta_new = soft_threshold(theta + alpha * g, tau)
        theta_new = beta_new + (1+i) / (4 + i) * (beta_new - beta)
        fhist = np.append(fhist, np.array([lasso_obj(beta_new, X, y, lambda_)]), axis=0)
        if fhist[i+1, 2] > fhist[i, 2]:
            break
        dt = np.abs(beta_new - beta)
        bt = np.abs(beta) 
        if np.all(dt < bt * rtol) or np.all(dt < bt * rtol + atol):
            beta = beta_new
            break
        else:
            beta = beta_new
            theta = theta_new
    return beta, fhist  
