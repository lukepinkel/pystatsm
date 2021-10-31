#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 22:30:15 2020

@author: lukepinkel
"""


import numpy as np
import scipy as sp
import scipy.stats
import scipy.sparse as sps
from sksparse.cholmod import cholesky# analysis:ignore
from ..utilities.data_utils import _check_np, _check_shape, dummy, _check_shape_nb
from ..utilities.linalg_operations import invech
from .model_matrices import (construct_model_matrices, 
                             make_gcov, make_theta)
from ..utilities.output import get_param_table
from ..pyglm.families import (Binomial, ExponentialFamily, Gamma, Gaussian,  # analysis:ignore
                       InverseGaussian, Poisson, NegativeBinomial)

def lndet_gmat(theta, dims, indices):
    lnd = 0.0
    for key, value in dims.items():
        if key!='error':
            dims_i = dims[key]
            ng = dims_i['n_groups']
            Sigma_i = invech(theta[indices[key]])
            lnd += ng*np.linalg.slogdet(Sigma_i)[1]
    return lnd
    
        
def gh_rules(n, wn=True):
    z, w =  sp.special.roots_hermitenorm(n)
    if wn:
        w = w / np.sum(w)
    f = sp.stats.norm(0, 1).logpdf(z)
    return z, w, f

def vech2vec(vh):
    A = invech(vh)
    v = A.reshape(-1, order='F')
    return v
    
def approx_hess(f, x, *args, eps=None):
    p = len(x)
    if eps is None:
        eps = (np.finfo(float).eps)**(1./3.)
    H = np.zeros((p, p))
    ei = np.zeros(p)
    ej = np.zeros(p)
    for i in range(p):
        for j in range(i+1):
            ei[i], ej[j] = eps, eps
            if i==j:
                dn = -f(x+2*ei)+16*f(x+ei)-30*f(x)+16*f(x-ei)-f(x-2*ei)
                nm = 12*eps**2
                H[i, j] = dn/nm  
            else:
                dn = f(x+ei+ej)-f(x+ei-ej)-f(x-ei+ej)+f(x-ei-ej)
                nm = 4*eps*eps
                
            ei[i], ej[j] = 0.0, 0.0
    return H
    
                
            
    
class GLMM_AGQ:
    def __init__(self, formula, data, family):
        if isinstance(family, ExponentialFamily)==False:
            family = family()
        self.f = family
        indices = {}
        X, Z, y, dims, levels = construct_model_matrices(formula, data)
        theta, indices["theta"] = make_theta(dims)
        G, indices["g"] = make_gcov(theta, indices, dims)
        self.X = _check_shape_nb(_check_np(X), 2)
        self.y = _check_shape_nb(_check_np(y), 1)
        self.Z = Z
        self.Zs = sps.csc_matrix(Z)
        self.Zt = self.Zs.T
        group_var, = list(dims.keys())
       
        n_vars = dims[group_var]['n_vars']
        self.J = sp.linalg.khatri_rao(dummy(data[group_var]).T, 
                                      np.ones((self.X.shape[0], n_vars)).T).T #dummy(data[group_var])
        self.n_indices = data.groupby(group_var).indices
        self.Xg, self.Zg, self.yg = {}, {}, {}
        self.u_indices, self.c_indices = {}, {}
        k = 0
        for j, (i, ix) in enumerate(self.n_indices.items()):
            self.Xg[i] = self.X[ix]
            self.Zg[i] = self.Z[ix, j][:, None]
            self.yg[i] = self.y[ix]
            self.u_indices[i] = np.arange(k, k+n_vars)
            self.c_indices[i] = (np.arange(k, k+n_vars)[:, None].T, 
                                 np.arange(k, k+n_vars)[:, None])
            k+=n_vars
        self.n_groups = len(self.Xg)
        self.n, self.p = self.X.shape
        self.q = self.Z.shape[1]
        self.nt = len(theta)
        self.params = np.zeros(self.p+self.nt)
        self.params[-self.nt:] = theta
        self.bounds = [(None, None) for i in range(self.p)]+\
                 [(None, None) if int(x)==0 else (0, None) for x in theta]
        
        self.D = np.eye(n_vars)
        self.W = sps.csc_matrix((np.ones(self.n), (np.arange(self.n), 
                                                   np.arange(self.n))))
        
        self.G = G
        self.dims = dims
        self.indices = indices
        self.levels = levels
    def update_gmat(self, theta, inverse=False):

        G = self.G
        for key in self.levels:
            ng = self.dims[key]['n_groups']
            theta_i = theta[self.indices['theta'][key]]
            if inverse:
                theta_i = np.linalg.inv(invech(theta_i)).reshape(-1, order='F')
            else:
                theta_i = invech(theta_i).reshape(-1, order='F')
            G.data[self.indices['g'][key]] = np.tile(theta_i, ng)
        return G
        
    def pirls(self, params):
        beta, theta = params[:self.p], params[self.p:]
        Psi_inv = self.update_gmat(theta, inverse=True)
        D = cholesky(Psi_inv).L()
        u = np.zeros(self.q)
        Xb = self.X.dot(beta)
        for i in range(100):
            eta = Xb + self.Z.dot(u)
            mu = self.f.inv_link(eta)
            var_mu = self.f.var_func(mu=mu)
            self.W.data = var_mu
            ZtWZ = self.Zt.dot(self.W.dot(self.Zs))
            Ztr = self.Zt.dot(_check_shape(self.y, 1) - mu)
            RtR, r = ZtWZ + Psi_inv, Ztr - Psi_inv.dot(u)
            u_new = u + sps.linalg.spsolve(RtR, r)
            diff = np.linalg.norm(u_new - u) / len(u)
            if diff<1e-6:
                break
            u = u_new
        eta = Xb + self.Zs.dot(u)
        mu = self.f.inv_link(eta)
        var_mu = self.f.var_func(mu=mu)
        self.W.data = var_mu
        ZtWZ = self.Zt.dot(self.W.dot(self.Zs))
        Ztr = self.Zt.dot(self.y - mu)
        RtR, r = ZtWZ + Psi_inv, Ztr - Psi_inv.dot(u)
        Q = cholesky(RtR.tocsc()).L()
        Qinv = sps.linalg.inv(Q)
        return dict(u=u, D=D, Xb=Xb, Qinv=Qinv, Q=Q)
    
    def _dloglike(self, db, Xb, Qinv, D, u):
        db = np.zeros(Qinv.shape[0]) + db
        u_tilde = Qinv.dot(db) + u
        eta = (self.Z.dot(u_tilde)) + Xb
        mu = self.f.inv_link(eta)
        T = self.f.canonical_parameter(mu)
        bT = self.f.cumulant(T)
        Du = D.dot(u_tilde)
        ll = (np.exp((self.y * T - bT).dot(self.J) - Du**2 / 2))
        return ll

    
    def loglike(self, params, nagq=20):
        pirls_dict = self.pirls(params)
        z, w, f = gh_rules(nagq, False)
        args = (pirls_dict['Xb'], pirls_dict['Qinv'], pirls_dict['D'], 
                pirls_dict['u'])
        sq2 = np.sqrt(2)
        ll_i = np.sum([self._dloglike(z[i], *args) * w[i] 
                     for i in range(len(w))], axis=0) * sq2
        ll_i = np.log(ll_i)
        lnd = np.linalg.slogdet(pirls_dict['D'].A)[1]\
              -np.linalg.slogdet(pirls_dict['Q'].A)[1]
        ll = -(np.sum(ll_i) + lnd)
        return ll
    
    def fit(self, nagq=20):
        self.optimizer = sp.optimize.minimize(self.loglike, self.params, 
                                   bounds=self.bounds, 
                                   options=dict(disp=1), args=(nagq,),
                                   jac='3-point')
        self.params = self.optimizer.x
        self.hess_theta = approx_hess(self.loglike, self.optimizer.x)
        self.se_params = np.sqrt(np.diag(np.linalg.inv(self.hess_theta)))
        self.res = get_param_table(self.params, self.se_params, 
                                   self.X.shape[0]-len(self.params))


    
        
       
               