#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 16 21:52:00 2020

@author: lukepinkel
"""
import numpy as np # analysis:ignore
import scipy as sp # analysis:ignore
import scipy.sparse as sps # analysis:ignore

from ..utilities.linalg_operations import (invech, sparse_cholesky, _check_shape_nb,
                               _check_np, sparse_woodbury_inversion, _check_shape,
                               scholesky)
from .model_matrices import (make_theta, construct_model_matrices, create_gmats,
                            lsq_estimate, update_gmat,  get_jacmats2)


def lndet_gmat(theta, dims, indices):
    lnd = 0.0
    for key, value in dims.items():
        if key!='error':
            dims_i = dims[key]
            ng = dims_i['n_groups']
            Sigma_i = invech(theta[indices[key]])
            lnd += ng*np.linalg.slogdet(Sigma_i)[1]
    return lnd
    
def lndet_cmat(M):  
    L = sparse_cholesky(M)
    LA = L.A
    logdetC = np.sum(2*np.log(np.diag(LA))[:-1])
    return logdetC



class LME:
    
    def __init__(self, formula, data):
        X, Z, y, dims = construct_model_matrices(formula, data)
        dims['error'] = dict(n_groups=len(X), n_vars=1)

        theta, indices = make_theta(dims)
        XZ = sps.hstack([X, Z])
        C = XZ.T.dot(XZ)
        Xty = X.T.dot(y)
        Zty = Z.T.dot(y)
        b = np.vstack([Xty, Zty])
        Gmats, g_indices = create_gmats(theta, indices, dims)
        Gmats_inverse, _ = create_gmats(theta, indices, dims, inverse=True)
        G = sps.block_diag(list(Gmats.values())).tocsc()
        Ginv =  sps.block_diag(list(Gmats_inverse.values())).tocsc()
        Zs = sps.csc_matrix(Z)
        Ip = sps.eye(Zs.shape[0])
        self.bounds = [(None, None) if int(x)==0 else (0, None) for x in theta]
        self.G = G
        self.Ginv = Ginv
        self.g_indices = g_indices
        self.X = _check_shape_nb(_check_np(X), 2)
        self.Z = Z
        self.y = _check_shape_nb(_check_np(y), 2)
        self.XZ = XZ
        self.C = C
        self.Xty = Xty
        self.Zty = Zty
        self.b = b
        self.dims = dims
        self.indices = indices
        self.formula = formula
        self.data = data
        self.theta = lsq_estimate(dims, theta, indices, X, XZ, self.y)
        self.Zs = Zs
        self.Ip = Ip
        self.yty = y.T.dot(y)
        self.jac_mats = get_jacmats2(self.Zs, self.dims, self.indices, 
                                     self.g_indices, self.theta)
        self.t_indices = list(zip(*np.triu_indices(len(theta))))
    
    
    def _params_to_model(self, theta):
        G = update_gmat(theta, self.G.copy(), self.dims, self.indices, self.g_indices)
        Ginv = update_gmat(theta, self.G.copy(), self.dims, self.indices, self.g_indices, inverse=True)
        s = theta[-1]
        R = self.Ip * s
        Rinv = self.Ip / s 
        V = self.Zs.dot(G).dot(self.Zs.T) + R
        Vinv = sparse_woodbury_inversion(self.Zs, Cinv=Ginv, Ainv=Rinv.tocsc())
        W = Vinv.A.dot(self.X)
        return G, Ginv, R, Rinv, V, Vinv, W, s
    
    def _mme(self, theta):
        _, Ginv, _, _, _, _, _, s = self._params_to_model(theta)
        C = self.C.copy()/s
        k =  Ginv.shape[0]
        C[-k:, -k:] += Ginv
        yty = np.array(np.atleast_2d(self.yty/s))
        M = sps.bmat([[C, self.b/s],
                      [self.b.T/s, yty]])
        return M
    
    def loglike(self, theta, sparse_chol=False):
        G, Ginv, R, Rinv, V, Vinv, W, s = self._params_to_model(theta)
        C = self.C.copy()/s
        k =  Ginv.shape[0]
        C[-k:, -k:] += Ginv
        logdetR = np.log(s) * self.Z.shape[0]
        logdetG = lndet_gmat(theta, self.dims, self.indices)
        yty = np.array(np.atleast_2d(self.yty/s))
        M = sps.bmat([[C, self.b/s],
                      [self.b.T/s, yty]])
        #L = sparse_cholesky(M)
        if sparse_chol:
            L = scholesky(M.tocsc(), ordering_method='natural').A
        else:
            L = np.linalg.cholesky(M.A)
        ytPy = np.diag(L)[-1]**2
        logdetC = np.sum(2*np.log(np.diag(L))[:-1])
        ll = logdetC+logdetG+logdetR+ytPy
        return ll
    
    def gradient(self, theta):
        dims = self.dims
        G, Ginv, R, Rinv, V, Vinv, W, s = self._params_to_model(theta)
        XtW = W.T.dot(self.X)
        XtW_inv = np.linalg.inv(XtW)
        P = Vinv - np.linalg.multi_dot([W, XtW_inv, W.T])
        Py = P.dot(self.y)
        grad = []
        for key in dims.keys():
            for dVdi in self.jac_mats[key]:
                #gi = np.trace(dVdi.dot(P)) - Py.T.dot(dVdi.dot(Py))
                gi = np.einsum("ij,ji->", dVdi.A, P) - Py.T.dot(dVdi.dot(Py))
                grad.append(gi)
        grad = np.concatenate(grad)
        grad = _check_shape(np.array(grad))
        return grad
    
    def hessian(self, theta):
        dims = self.dims
        G, Ginv, R, Rinv, V, Vinv, W, s = self._params_to_model(theta)
        XtW = W.T.dot(self.X)
        XtW_inv = np.linalg.inv(XtW)
        P = Vinv - np.linalg.multi_dot([W, XtW_inv, W.T])
        Py = P.dot(self.y)
        H = []
        PJ, yPJ = [], []
        for key in dims.keys():
            J_list = self.jac_mats[key]
            for i in range(len(J_list)):
                Ji = J_list[i].T
                PJ.append((Ji.dot(P)).T)
                yPJ.append((Ji.dot(Py)).T)
        t_indices = self.t_indices
        for i, j in t_indices:
            PJi, PJj = PJ[i], PJ[j]
            yPJi, JjPy = yPJ[i], yPJ[j].T
            Hij = -(PJi.dot(PJj)).diagonal().sum()\
                        + (2 * (yPJi.dot(P)).dot(JjPy))[0]
            H.append(np.array(Hij[0]))
        H = invech(np.concatenate(H)[:, 0])
        return H
    
    def _optimize_theta(self, optimizer_kwargs={}, hess=None):
        if 'method' not in optimizer_kwargs.keys():
            optimizer_kwargs['method'] = 'trust-constr'
        if 'options' not in optimizer_kwargs.keys():
            optimizer_kwargs['options']=dict(gtol=1e-6, xtol=1e-6, verbose=3)
        optimizer = sp.optimize.minimize(self.loglike, self.theta, hess=hess,
                                         bounds=self.bounds, jac=self.gradient, 
                                         **optimizer_kwargs)
        theta = optimizer.x
        return optimizer, theta
    
    def _acov(self, theta=None):
        if theta is None:
            theta = self.theta
        H_theta = self.hessian(theta)
        Hinv_theta = np.linalg.inv(H_theta)
        SE_theta = np.sqrt(np.diag(Hinv_theta))
        return H_theta, Hinv_theta, SE_theta
    
    def _compute_effects(self, theta=None):
        G, Ginv, R, Rinv, V, Vinv, WX, s = self._params_to_model(theta)
        XtW = WX.T
        XtWX = XtW.dot(self.X)
        XtWX_inv = np.linalg.inv(XtWX)
        beta = _check_shape(XtWX_inv.dot(XtW.dot(self.y)))
        fixed_resids = _check_shape(self.y) - _check_shape(self.X.dot(beta))
        
        Zt = self.Zs.T
        u = G.dot(Zt.dot(Vinv)).dot(fixed_resids)
        
        return beta, XtWX_inv, u, G, R, Rinv, V, Vinv
    
    def _fit(self, optimizer_kwargs={}, hess=None):
        optimizer, theta = self._optimize_theta(optimizer_kwargs, hess)
        self.theta = theta
        
        H_theta, Hinv_theta, SE_theta = self._acov(theta)
        
        beta, XtWX_inv, u, G, R, Rinv, V, Vinv = self._compute_effects(theta)
        
        self._V = V
        self._Vinv = Vinv
        self._G = G
        self._R = R
        self._Rinv = Rinv
        self.beta = beta
        self.u = u
        self._SE_theta = SE_theta
        
        self.fixed_effects_acov = XtWX_inv
        self.theta_acov = Hinv_theta
        self.random_effects_acov = G
        
        self.yhat_f = self.X.dot(self.beta)
        self.yhat_r = self.Z.dot(self.u)
        self.yhat = self.yhat_f + self.yhat_r
        
        self.resid_f = _check_shape(self.y) - self.yhat_f
        self.resid_r = _check_shape(self.y) - self.yhat_r
        self.resid = _check_shape(self.y) - self.yhat
        
        self.sigma_f = np.dot(self.resid_f.T, self.resid_f) 
        self.sigma_r = np.dot(self.resid_r.T, self.resid_r) 
        self.sigma = np.dot(self.resid.T, self.resid)
        self.sigma_y = np.var(self.y) * len(self.y)
        
        self.r2_f = 1 - self.sigma_f / self.sigma_y
        self.r2_r = 1 - self.sigma_r / self.sigma_y
        self.r2 = 1 - self.sigma / self.sigma_y
        
        self.params = np.concatenate([self.beta, self.theta])
        self.se_params = np.concatenate([np.sqrt(np.diag(self.fixed_effects_acov)),
                                         SE_theta])
        self.params_ci_lower = self.params - 1.96 * self.se_params
        self.params_ci_upper = self.params + 1.96 * self.se_params
        self.params_tvalues = self.params / self.se_params
        df = self.X.shape[0]-len(self.theta)
        self.params_pvalues = sp.stats.t(df).sf(np.abs(self.params_tvalues))
        
            