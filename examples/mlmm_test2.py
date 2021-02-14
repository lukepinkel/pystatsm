# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 01:58:20 2021

@author: lukepinkel
"""
import scipy as sp
import scipy.stats
import numpy as np
import pandas as pd
from pystats.utilities.numerical_derivs import fo_fc_cd, so_gc_cd
from pystats.pylmm.mlmm import (MLMM, construct_model_matrices, vec,
                                      inverse_transform_theta, sparse_woodbury_inversion)
from pystats.utilities.random_corr import vine_corr, exact_rmvnorm


n_groups = 200
n_per = 10
n_obs = n_groups * n_per

S = vine_corr(4)
X = exact_rmvnorm(S, n_obs)
df = pd.DataFrame(X, columns=['x1', 'x2', 'x3', 'x4'])
df['y'] = 0
df['id1'] = np.repeat(np.arange(n_groups), n_per)

formula = "y~x1+x2+x3+x4+(1|id1)"
X, Z, y, dims, levels, _, _ = construct_model_matrices(formula, data=df)
B = np.random.normal(0, 4, size=(5, 2))

A = np.eye(n_groups)
Sigma_u = np.array([[2.0, 1.5], [1.5, 2.0]])
Sigma_e = np.array([[2.0, 0], [0, 1.0]])
U = sp.stats.matrix_normal(mean=np.zeros((A.shape[0], Sigma_u.shape[0])),
                           rowcov=A, colcov=Sigma_u).rvs(1)

E = sp.stats.multivariate_normal(mean=np.zeros(2), cov=Sigma_e).rvs(n_obs)
Y = X.dot(B) + Z.dot(U) + E
y = vec(Y.T)
df[['y1', 'y2']] = Y

formula = "(y1, y2)~x1+x2+x3+x4+(1|id1)"


model = MLMM(formula, df)


res_hess_appr = sp.optimize.minimize(model.loglike_c, model.theta, jac=model.gradient_chol, 
                                hess='3-point', bounds=model.bounds, 
                                method='trust-constr', 
                                options=dict(verbose=3))


res_hess_true = sp.optimize.minimize(model.loglike_c, model.theta, jac=model.gradient_chol, 
                                     hess=model.hessian_chol, bounds=model.bounds, 
                                     method='trust-constr', 
                                     options=dict(verbose=3, xtol=1e-21))

x = res_hess_true.x.copy()


theta_chol_hat = res_hess_true.x.copy()
theta_hat = inverse_transform_theta(theta_chol_hat.copy(), model.dims, model.indices)

theta = theta_hat.copy()
Ginv = model.update_gmat(theta, inverse=True)
Rinv = model.update_rmat(theta, inverse=True)
Vinv = sparse_woodbury_inversion(model.Zs, Cinv=Ginv, Ainv=Rinv.tocsc())
W = (Vinv.dot(model.X))
XtW = W.T.dot(model.X)
XtW_inv = np.linalg.inv(XtW)
P = Vinv - np.linalg.multi_dot([W, XtW_inv, W.T])
Py = P.dot(model.y)
H = []
PJ, yPJ = [], []
for key in (model.levels+['error']):
    J_list = model.jac_mats[key]
    for i in range(len(J_list)):
        Ji = J_list[i].T
        PJ.append((Ji.dot(P)).T)
        yPJ.append((Ji.dot(Py)).T)
t_indices = model.t_indices
for i, j in t_indices:
    PJi, PJj = PJ[i], PJ[j]
    yPJi, JjPy = yPJ[i], yPJ[j].T
    Hij = -np.einsum('ij,ji->', PJi, PJj)\
                + (2 * (yPJi.dot(P)).dot(JjPy))[0]
t_indices = model.t_indices
i, j = 0, 0
PJi, PJj = PJ[i], PJ[j]
yPJi, JjPy = yPJ[i], yPJ[j].T


def min_rnewton(func, x, grad, hess, n_iters=100, n_halves=20, ftol=1e-8):
    x_prev = x.copy()
    f_prev = func(x_prev)
    fit_hist = {}
    n_step_halves = 0
    cond = None
    for i in range(n_iters):
        H, g = hess(x_prev), grad(x_prev)
        fit_hist[i] = dict(x=x_prev, f=f_prev, grad=g, gnorm=np.linalg.norm(g))
        d = np.linalg.solve(H, g)
        x_new = x_prev - d
        f_new = func(x_new)
        n_step_halves = 0
        if f_new > f_prev:
            for j in range(n_halves):
                n_step_halves += 1
                d /= 2.0
                x_new = x_prev - d
                f_new = func(x_new)
                if f_new < f_prev:
                    break
        fit_hist[i]['n_step_halves'] = n_step_halves
        if f_prev < f_new:
            x_new = x_prev
            f_new = f_prev
            cond = "Step Halving Failed To Improve Objective Function"
            break
        x_prev = x_new
        if (f_prev - f_new) < ftol:
            cond = "Improvement Less than ftol"
            break
        f_prev = f_new
        print(fit_hist[i])
    if cond is None and i==(n_iters-1):
        cond = "Max Iterations Reached"
    res = dict(fit_hist=fit_hist, cond=cond, x=x_new, 
               f=f_new, n_iters=i)
    return res

   
res = min_rnewton(model.loglike_c,  model.theta_chol.copy(), 
                  model.gradient_chol, model.hessian_chol, 100, 50, 1e-9)  
    

class Callback:

    def __init__(self, func, grad):
        self.func = func
        self.grad = grad
        self.xvals = []
        self.fvals = []
        self.gvals = []
        self._i = 0
    
    def __call__(self, xk):
        self._i += 1
        self.fvals.append(self.func(xk))
        self.gvals.append(self.grad(xk))
        self.xvals.append(xk)            
        
        
        
cb = Callback(model.loglike_c, model.gradient_chol)
res = sp.optimize.minimize(model.loglike_c, model.theta, jac=model.gradient_chol, 
                           bounds=model.bounds, method='l-bfgs-b', callback=cb,
                           tol=1e-16, options=dict(iprint=100, disp=None, gtol=1e-12,
                                                   ftol=1e-12, maxfun=200, maxiter=500))


xvals = np.vstack([model.theta.copy()]+cb.xvals)

res2 = sp.optimize.minimize(model.loglike_c, model.theta, jac=model.gradient_chol, 
                            hess='3-point', bounds=model.bounds, 
                            method='trust-constr', options=dict(verbose=3))

inverse_transform_theta(res.x, model.dims, model.indices)

theta_chol = model.theta_chol.copy()
fo_fc_cd(model.loglike_c, theta_chol)
model.gradient_chol(theta_chol)
H = model.hessian_chol(theta_chol)
H2 = so_gc_cd(model.gradient_chol, theta_chol)

    
    