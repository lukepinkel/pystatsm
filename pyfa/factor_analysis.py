#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 22:09:50 2020

@author: lukepinkel
"""
import numpy as np
import scipy as sp
import scipy.stats
import pandas as pd
from ..utilities.linalg_operations import vec, vech, invec, invech, _check_np, _check_shape
from ..utilities.data_utils import _check_type
from ..utilities.special_mats import nmat, dmat, lmat
from ..utilities.optimizer_utils import process_optimizer_kwargs
from ..utilities.output import get_param_table
from ..utilities.random_corr import multi_rand
from ..utilities.numerical_derivs import so_gc_cd, fo_fc_cd
  

def get_param_indices(params, n_vars, n_factors):
    lam_indices = np.zeros_like(params)
    phi_indices = np.zeros_like(params)
    psi_indices = np.zeros_like(params)
    
    k1 = int(n_vars * n_factors)
    k2 = int(k1 + (n_factors + 1) * n_factors / 2)
    lam_indices[:k1] = 1
    phi_indices[k1:k2] = 1
    psi_indices[k2:] = 1
    
    lam_indices = lam_indices.astype(bool)
    phi_indices = phi_indices.astype(bool)
    psi_indices = psi_indices.astype(bool) 
    param_indices = dict(Lambda=lam_indices, Phi=phi_indices, Psi=psi_indices)
    return param_indices
    
    
def srmr(Sigma, S, df):
    S = _check_np(S)
    p = S.shape[0]
    y = 0.0
    t = (p + 1.0) * p
    for i in range(p):
        for j in range(i):
            y += (Sigma[i, j]-S[i, j])**2/(S[i, i]*S[j, j])
    
    y = np.sqrt((2.0 / (t)) * y)      
    return y

def lr_test(Sigma, S, df, n):
    p = Sigma.shape[0]
    _, lndS = np.linalg.slogdet(S)
    _, lndSigma = np.linalg.slogdet(Sigma)
    Sigma_inv = np.linalg.pinv(Sigma)
    chi2 = (lndSigma + np.trace(Sigma_inv.dot(S)) - lndS - p) * n
    chi2 = np.maximum(chi2, 1e-12)
    pval = sp.stats.chi2.sf(chi2, (p + 1)*p/2)
    return chi2, pval

def gfi(Sigma, S):
    p = S.shape[0]
    tmp1 = np.linalg.pinv(Sigma).dot(S)
    tmp2 = tmp1 - np.eye(p)
    y = 1.0 - np.trace(np.dot(tmp2, tmp2)) / np.trace(np.dot(tmp1, tmp1))
    return y

def agfi(Sigma, S, df):
    p = S.shape[0]
    t = (p + 1.0) * p
    tmp1 = np.linalg.pinv(Sigma).dot(S)
    tmp2 = tmp1 - np.eye(p)
    y = 1.0 - np.trace(np.dot(tmp2, tmp2)) / np.trace(np.dot(tmp1, tmp1))
    y = 1.0 - (t / (2.0*df)) * (1.0-y)
    return y



def make_params(S, n_factors, free_loadings=None, free_lcov=None, st_lvar=True):
    n_vars = S.shape[0]
    
    if free_loadings is None:
        loadings_mask = np.ones((n_vars, n_factors)).astype(bool)
    else:
        loadings_mask = free_loadings.astype(bool)
    
    if free_lcov is None:
        lcov_mask = np.zeros((n_factors, n_factors)).astype(bool)
    else:
        lcov_mask = free_lcov.astype(bool)

    psi_mask = np.eye(n_vars).astype(bool)
    
    u, V = np.linalg.eigh(S)
    
    Lambda = V[:, :n_factors]
    Psi = np.diag((V**2)[:, :n_factors].sum(axis=1))
    Phi = np.eye(n_factors)
    
    params = np.block([vec(Lambda*loadings_mask), vech(Phi), vech(Psi)])
  
    param_indices = get_param_indices(params, n_vars, n_factors)
    free_indices = np.block([vec(loadings_mask), vech(lcov_mask), 
                             vech(psi_mask)])
    bounds = [(None, None) for i in range(n_vars * n_factors)]
    if st_lvar:
        bounds+= [(0, 1) if x==1 else (-1, 1) for x in vech(np.eye(n_factors))]
    else:     
        bounds+= [(0, None) if x==1 else (None, None) for x in vech(np.eye(n_factors))]
    bounds+= [(0, None) if x==1 else (None, None) for x in vech(np.eye(n_vars))]
    bounds =  np.asarray(bounds)[free_indices]
    bounds = [tuple(x) for x in bounds.tolist()]
    free = params[free_indices]
    
    return params, free, free_indices, param_indices, bounds

def measure_of_sample_adequacy(Sigma):
    V = np.diag(np.sqrt(1/np.diag(Sigma)))
    R = V.dot(Sigma).dot(V)
    Rinv = np.linalg.inv(R)
    D = np.diag(1.0/np.sqrt(np.diag(Rinv)))
    Q = D.dot(Rinv).dot(D)
    ix = np.tril_indices(Sigma.shape[0], -1)
    r = np.sum(R[ix]**2)
    q = np.sum(Q[ix]**2)
    msa = r / (r + q)
    return msa
    
def invech_chol(lvec):
    p = int(0.5 * ((8*len(lvec) + 1)**0.5 - 1))
    L = np.zeros((p, p))
    a, b = np.triu_indices(p)
    L[(b, a)] = lvec
    return L


class FactorAnalysis:
    
    def __init__(self, X, n_factors, free_loadings=None, free_lcov=None, 
                 st_lvar=True):
        if st_lvar and free_lcov is not None:
            free_lcov[np.diag_indices(free_lcov.shape[0])] = 0
        self.data = X
        self.X, self.xcols, self.xix, self.is_pd = _check_type(X)
        S = np.cov(self.X, rowvar=False, bias=True)
        params, free, fix, pix, bounds = make_params(S, n_factors, free_loadings,
                                                     free_lcov, st_lvar)
    
        self.n_obs = X.shape[0]
        self.S = S
        self.params = params
        self.free = free
        self.free_indices = fix
        self.param_indices = pix
        self.bounds = bounds
        self.n_factors = n_factors
        self.n_vars = X.shape[1]
        self.k1 = int(self.n_vars * n_factors)
        self.k2 = int(self.k1 + (n_factors + 1) * n_factors / 2)
        self.model_options = dict(free_loadings=free_loadings,
                                  free_lcov=free_lcov,
                                  st_lvar=st_lvar)
        
        self._Np = nmat(self.n_vars).A
        self._Ip = np.eye(self.n_vars)
        self._Dp = dmat(self.n_vars).A
        self._Lp = lmat(self.n_vars).A
        self._Dq = dmat(self.n_factors).A
        self._T = np.zeros((self.n_vars, self.n_vars))
        
    
    def params_to_mats(self, params):
        Lambda = invec(params[self.param_indices['Lambda']], self.n_vars, 
                       self.n_factors)
        PhiC = invech_chol(params[self.param_indices['Phi']])
        Psi = invech(params[self.param_indices['Psi']])
        Phi = PhiC.dot(PhiC.T)
        return Lambda, Phi, Psi, PhiC
    
    def free_to_mats(self, free):
        params = self.params.copy()
        params[self.free_indices] = free
        Lambda, Phi, Psi, _ = self.params_to_mats(params)
        return Lambda, Phi, Psi
    
    def free_to_sigma(self, free, return_mats=False):
        params = self.params.copy()
        params[self.free_indices] =  free.flatten()
        Lambda, Phi, Psi, PhiC = self.params_to_mats(params)
        Sigma = np.linalg.multi_dot([Lambda, Phi, Lambda.T]) + Psi**2
        if return_mats:
            return Sigma, Lambda, Phi, Psi, PhiC
        else:
            return Sigma
    
    def loglike(self, free):
        Sigma = self.free_to_sigma(free)
        Sigma_inv = np.linalg.pinv(Sigma)
        _, lndet = np.linalg.slogdet(Sigma)
        trS = np.trace(Sigma_inv.dot(self.S))
        ll = lndet + trS
        return ll
    
    def gradient(self, free, return_full=False):
        S = self.S
        Sigma, Lambda, Phi, Psi, PhiC = self.free_to_sigma(free, True)
        V = np.linalg.pinv(Sigma)
        R = Sigma - S
        VRV = V.dot(R).dot(V)
        VRVL = VRV.dot(Lambda)
        J1 = 2 * vec(VRVL.dot(Phi))
        J2 = 2 * vech(Lambda.T.dot(VRVL).dot(PhiC))
        J3 = 2 * vech(VRV.dot(Psi))
        g = np.block([J1, J2, J3])
        if return_full:
            return g
        else:
            return g[self.free_indices]
        
    
    def hessian(self, free):
        free = _check_shape(free, 1)
        H = so_gc_cd(self.gradient, free)
        return H
    
    def _optimize_free_params(self, optimizer_kwargs={}, hess=None):
        optimizer_kwargs = process_optimizer_kwargs(optimizer_kwargs,
                                                    'trust-constr')
        optimizer = sp.optimize.minimize(self.loglike, self.free, hess=hess,
                                         bounds=self.bounds, jac=self.gradient, 
                                         **optimizer_kwargs)
        return optimizer
        
    def fit(self, optimizer_kwargs={}, hess=True):
        if 'options' in optimizer_kwargs.keys():
            if 'verbose' not in optimizer_kwargs['options'].keys():
                optimizer_kwargs['options']['verbose'] = 0
            if 'gtol' not in optimizer_kwargs['options'].keys():
                optimizer_kwargs['options']['gtol'] = 1e-12
            if 'xtol' not in optimizer_kwargs['options'].keys():
                optimizer_kwargs['options']['xtol'] = 1e-12
            if 'maxiter' not in optimizer_kwargs['options'].keys():
                optimizer_kwargs['options']['maxiter'] = 5000
            
        else:
            optimizer_kwargs['options'] = dict(verbose=0, gtol=1e-12,
                                                xtol=1e-20,
                                                maxiter=5000)
        
        if hess:
            hessian = self.hessian
        else:
            hessian = None
        
        self.opt = self._optimize_free_params(optimizer_kwargs, hessian)
        self.free = self.opt.x
        self.opt = self._optimize_free_params(optimizer_kwargs, hessian)
        self.free = self.opt.x
        self.params[self.free_indices] = self.free
        self.Lambda, self.Phi, self.Psi, _ = self.params_to_mats(self.params)
        self.Sigma = self.Lambda.dot(self.Phi).dot(self.Lambda.T) + self.Psi**2
        SE =  np.diag(np.linalg.pinv(self.n_obs*self.hessian(self.free))).copy()
        tmp =  np.sum(self.free_indices[:self.k1]).astype(int)
        if tmp==self.k1:
            SE[:tmp] = 1.0
        self.SE = SE**0.5
        self.res = get_param_table(self.free, self.SE, self.X.shape[0]-len(self.free))
        
        if self.is_pd:
            cols = ['Factor %i'%i for i in range(1, int(self.Lambda.shape[1]+1.0))]
            self.Lambda = pd.DataFrame(self.Lambda, index=self.xcols,
                                           columns=cols)
            self.Phi = pd.DataFrame(self.Phi, columns=cols, index=cols)
            self.Psi = pd.DataFrame(self.Psi, index=self.xcols,
                                    columns=self.xcols)**2
            self.psi = pd.DataFrame(np.diag(self.Psi), index=self.xcols,
                                    columns=['psi'])**2
    
        self.sumstats = self._post_fit(self.Sigma)
        param_labels = []
        for j in range(self.Lambda.shape[1]):
            for i in range(self.Lambda.shape[0]):
                param_labels.append(f"Lambda[{i}][{j}]")
        for j, i in list(zip(*np.triu_indices(self.Phi.shape[0]))):
            param_labels.append(f"Phi[{i}][{j}]")
        for j, i in list(zip(*np.triu_indices(self.Psi.shape[0]))):
            param_labels.append(f"Psi[{i}][{j}]")
        self.param_labels = param_labels
        self.res.index = np.array(self.param_labels)[self.free_indices]
       
        
    def _post_fit(self, Sigma):
        n = self.X.shape[0]
        t = (self.n_vars + 1.0) * self.n_vars / 2.0
        degfree = t  - np.sum(self.free_indices)
        GFI = gfi(Sigma, self.S)
        AGFI = agfi(Sigma, self.S, degfree)
        chi2, chi2p = lr_test(Sigma, self.S, degfree, n-1)
        stdchi2 = (chi2 - degfree) /  np.sqrt(2 * degfree)
        RMSEA = np.sqrt(np.maximum(chi2 -degfree, 0)/(degfree*(n-1)))
        SRMR = srmr(Sigma, self.S, degfree)
        llval = -(n-1)*self.loglike(self.free)/2
        k = np.sum(self.free_indices)
        BIC = k * np.log(n) -2 * llval
        AIC = 2 * k - 2*llval
        sumstats = dict(GFI=GFI, AGFI=AGFI, chi2=chi2, chi2_pvalue=chi2p,
                         std_chi2=stdchi2, RMSEA=RMSEA, SRMR=SRMR,
                         loglikelihood=llval,
                         AIC=AIC,
                         BIC=BIC)
        return sumstats
    
        
    
        
        

        
#





'''

Lambda = np.zeros((12, 3))
Lambda[0:4, 0] = 1
Lambda[4:8, 1] = 1
Lambda[8: , 2] = 1
Phi = np.eye(3)
Psi = np.diag(np.random.uniform(low=0.2, high=1.0, size=12))

S = Lambda.dot(Phi).dot(Lambda.T) + Psi

X = multi_rand(S)

import mvpy.api as mv

model = mv.FactorAnalysis(X, 3)
model2 = FactorAnalysis(X, 3)

Sigma = model2.free_to_sigma(model2.free)
Sinv = np.linalg.inv(Sigma)
G = model2.dsigma(model2.free)


np.allclose(model2.loglike(model.free), model.loglike(model.free))
np.allclose(model2.gradient(model.free), model.gradient(model.free))
np.allclose(model2.hessian(model.free), model.hessian(model.free))
np.allclose(model2.free_indices, model.idx)

_params = model.params.copy()
_params[model.idx] = model.free

np.allclose(model2.free_to_mats(model.free)[0], model.p2m(_params)[0])
np.allclose(model2.free_to_mats(model.free)[1], model.p2m(_params)[1])
np.allclose(model2.free_to_mats(model.free)[2], model.p2m(_params)[2])


g1 = fo_fc_cd(model2.loglike, model2.free)
#g1 = nderivs.so_fc_cd(model2.loglike, model2.free)
H1 = so_gc_cd(model2.gradient, model2.free)
H2 = model2.hessian(model2.free)
ix = np.where(np.abs(H2)>1e-2)


h1, h2 = H1[ix], H2[ix]

optimizer = sp.optimize.minimize(model2.loglike, model2.free, hess=model2.hessian,
                                     bounds=model2.bounds, jac=model2.gradient, 
                                     method='trust-constr',
                                     options=dict(verbose=3, gtol=1e-9,
                                                  xtol=1e-20, maxiter=5000))
model.fit()
model2.fit()
L, _  = mv.rotate(model.Lambda, 'varimax')


'''






