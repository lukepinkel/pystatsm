# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 15:19:15 2021

@author: lukepinkel
"""
import numpy as np
from pystats.pyglm.families import Gaussian
import patsy
import scipy as sp
import pandas as pd
import statsmodels.api as sm
from pystats.utilities import splines

def pirls(y, X, S, family, n_iters=200):
    mu = np.zeros_like(y)+y.mean()
    eta = family.link(mu)
    beta = np.zeros(X.shape[1])
    fit_hist = []
    for i in range(n_iters):
        gp = family.dlink(mu)
        z = gp * (y - mu) + eta
        w = 1.0 / (gp**2 * family.var_func(mu=mu))
        XWt = (X * w.reshape(-1, 1)).T
        beta_new = np.linalg.pinv(np.dot(XWt, X)+S).dot(XWt.dot(z))
        fit_hist.append(np.mean((beta-beta_new)**2))
        beta = beta_new
        eta = X.dot(beta)
        mu = family.inv_link(eta)
    return beta, fit_hist

data = pd.read_csv("C:/Users/John/Downloads/mpg.csv", index_col=0)
data.columns = [x.replace(".", "_") for x in data.columns]





formula = "city_mpg~C(fuel)+C(style)+C(drive)"



#B Splines
y, Xl = patsy.dmatrices(formula, data=data, return_type="dataframe")
y = y.values

D, Xs, S, knots = splines.get_bsplines(data["weight"].values, df=10)
sc = splines.get_penalty_scale(Xs, S)
Xs, S = splines.transform_spline_modelmat(Xs, S)
S = S / sc
X  = np.hstack((Xl, Xs))

St = sp.linalg.block_diag(np.zeros((Xl.shape[1], Xl.shape[1])), S*0.0172846 )

beta, fit_hist = pirls(y, X, St, Gaussian())
beta = pd.DataFrame(beta, index=Xl.columns.tolist()+["s%i"%i for i in range(1, Xs.shape[1]+1)])

np.sum((y - X.dot(beta))**2)

#Cubic Regression Splines

y, Xl = patsy.dmatrices(formula, data=data, return_type="dataframe")
y = y.values

D, F, Xs, S, knots = splines.get_crsplines(data["weight"].values, df=10)
sc =  splines.get_penalty_scale(Xs, S)
Xs, S = splines.transform_spline_modelmat(Xs, S)
S = S / sc
X  = np.hstack((Xl, Xs))

St = sp.linalg.block_diag(np.zeros((Xl.shape[1], Xl.shape[1])), S* 15.21952  )

beta, fit_hist = pirls(y, X, St, Gaussian())
beta = pd.DataFrame(beta, index=Xl.columns.tolist()+["s%i"%i for i in range(1, Xs.shape[1]+1)])

np.sum((y - X.dot(beta))**2)


x = data['weight'].values


#Cyclic Cubic Regression Splines


y, Xl = patsy.dmatrices(formula, data=data, return_type="dataframe")
y = y.values

Xs, BinvD, S, knots = splines.get_ccsplines(data["weight"].values, df=10)
sc =  splines.get_penalty_scale(Xs, S)
Xs, S = splines.transform_spline_modelmat(Xs, S)
S = S / sc
X  = np.hstack((Xl, Xs))

St = sp.linalg.block_diag(np.zeros((Xl.shape[1], Xl.shape[1])), S*0.7739821674)

beta, fit_hist = pirls(y, X, St, Gaussian())
beta = pd.DataFrame(beta, index=Xl.columns.tolist()+["s%i"%i for i in range(1, Xs.shape[1]+1)])

np.sum((y - X.dot(beta))**2)

def gcv(X, y, S, a):
    A = X.dot(np.linalg.inv(X.T.dot(X)+S*a)).dot(X.T)
    r = y - A.dot(y)
    n = y.shape[0]
    v = n * r.T.dot(r) / (n - np.trace(A))**2
    return v

def dgcv(X, y, S, a, gamma=1.5):
    A = X.dot(np.linalg.inv(X.T.dot(X)+S*a)).dot(X.T)
    r = y - A.dot(y)
    n = y.shape[0]
    v = n * r.T.dot(r) / (n - gamma * np.trace(A))**2
    return v

def dv_da(X, y, St, a):
    V = np.linalg.inv(X.T.dot(X)+St*a)
    A = X.dot(V).dot(X.T)
    r = y - A.dot(y)
    M = X.dot(V).dot(St).dot(V).dot(X.T)
    n = y.shape[0]
    u = n - np.trace(A)
    u2 = u**2
    u3 = u2 * u
    g = 2 * n * r.T.dot(M.dot(y) / u2 - np.trace(M) * r / u3)
    return g


def ddgcv_da(X, y, St, a, gamma=1.5):
    V = np.linalg.inv(X.T.dot(X)+St*a)
    A = X.dot(V).dot(X.T)
    r = y - A.dot(y)
    M = X.dot(V).dot(St).dot(V).dot(X.T)
    n = y.shape[0]
    u = n - gamma * np.trace(A)
    u2 = u**2
    u3 = u2 * u
    g = 2 * n * r.T.dot(M.dot(y) / u2 - gamma * np.trace(M) * r / u3)
    return g

import autograd
import autograd.numpy as anp

def gcv2(X, y, S, a):
    A = anp.dot(anp.dot(X, anp.linalg.inv(anp.dot(X.T, X)+S*a)), X.T)
    r = y - anp.dot(A, y)
    n = y.shape[0]
    v = anp.dot(r.T, r) * n / (n - anp.trace(A))**2
    return v

def dgcv2(X, y, S, a, gamma=1.5):
    A = anp.dot(anp.dot(X, anp.linalg.inv(anp.dot(X.T, X)+S*a)), X.T)
    r = y - anp.dot(A, y)
    n = y.shape[0]
    v = anp.dot(r.T, r) * n / (n - gamma * anp.trace(A))**2
    return v

Xa, ya, Sa, a = anp.asarray(X), anp.asarray(y), anp.asarray(St), 0.5

grad = autograd.jacobian(gcv2, argnum=3)
grad(Xa, ya, Sa, 3.)
dv_da(X, y, St, 3.)

grad = autograd.jacobian(dgcv2, argnum=3)
grad(Xa, ya, Sa, 3.)
ddgcv_da(X, y, St, 3.)


res = sp.optimize.minimize(lambda theta, X, y, St: dgcv(X, y, St, theta), 
                           0.0, jac=lambda theta, X, y, St: ddgcv_da(X, y, St, theta),
                           args=(X, y, St),
                           method='L-BFGS-B', options=dict(iprint=100,
                                                           gtol=1e-12,
                                                           ftol=1e-12))



res = sp.optimize.minimize(lambda theta, X, y, St: gcv(X, y, St, theta), 
                           0.0, jac=lambda theta, X, y, St: dv_da(X, y, St, theta),
                           args=(X, y, St),
                           method='L-BFGS-B', options=dict(iprint=100,
                                                           gtol=1e-12,
                                                           ftol=1e-16))








