#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 01:03:35 2020

@author: lukepinkel
"""
from jax.config import config
config.update("jax_enable_x64", True)
import jax # analysis:ignore
import numpy as np # analysis:ignore
import collections #analysis:ignore
import numpy as np #analysis:ignorezd
import scipy as sp #analysis:ignore
import scipy.stats #analysis:ignore
import pandas as pd # analysis:ignore
from pystats.utilities.linalg_operations import (_check_np, _check_shape, vec,  # analysis:ignore
                                                 invec,vech, invech)# analysis:ignore
from pystats.utilities.special_mats import dmat, lmat, nmat, kmat# analysis:ignore
from pystats.utilities.data_utils import _check_type# analysis:ignore
from pystats.utilities.output import get_param_table

data = pd.read_csv("/users/lukepinkel/Downloads/bollen.csv", index_col=0)
data = data[['x1', 'x2', 'x3', 'y1', 'y2', 'y3', 'y4', 'y5',
             'y6', 'y7', 'y8', ]]
L = np.array([[1, 0, 0],
              [1, 0, 0],
              [1, 0, 0],
              [0, 1, 0],
              [0, 1, 0],
              [0, 1, 0],
              [0, 1, 0],
              [0, 0, 1],
              [0, 0, 1],
              [0, 0, 1],
              [0, 0, 1]])
B = np.array([[False, False, False],
              [True,  False, False],
              [True,  True, False]])
LA = pd.DataFrame(L, index=data.columns, columns=['ind60', 'dem60', 'dem65'])
BE = pd.DataFrame(B, index=LA.columns, columns=LA.columns)
S = data.cov()
Zg = ZR = data
Lambda=LA!=0
Beta=BE!=0 
Lambda, Beta = pd.DataFrame(Lambda), pd.DataFrame(Beta)
Lambda.columns = ['ind60', 'dem60', 'dem65']
Lambda.index = Zg.columns
Beta.columns = Lambda.columns
Beta.index = Lambda.columns
Theta = pd.DataFrame(np.eye(Lambda.shape[0]),
                     index=Lambda.index, columns=Lambda.index)
Theta.loc['y1', 'y5'] = 0.05
Theta.loc['y2', 'y4'] = 0.05
Theta.loc['y2', 'y6'] = 0.05
Theta.loc['y3', 'y7'] = 0.05
Theta.loc['y4', 'y8'] = 0.05
Theta.loc['y6', 'y8'] = 0.05
Theta.loc['y5', 'y1'] = 0.05
Theta.loc['y4', 'y2'] = 0.05
Theta.loc['y6', 'y2'] = 0.05
Theta.loc['y7', 'y3'] = 0.05
Theta.loc['y8', 'y4'] = 0.05
Theta.loc['y8', 'y6'] = 0.05
data = Zg.copy()
Psi = Theta.copy()
Phi = np.eye(3)
Lambda.loc['x1', 'ind60'] = False
Lambda.loc['y1', 'dem60'] = False
Lambda.loc['y5', 'dem65'] = False
indicator_vars = Lambda.copy() * 0
indicator_vars = indicator_vars.astype(bool)
indicator_vars.loc['x1', 'ind60'] = True
indicator_vars.loc['y1', 'dem60'] = True
indicator_vars.loc['y5', 'dem65'] = True




def mat_rconj(A):
    return jax.numpy.eye(A.shape[0]) - A  


def make_indices(Lambda, Beta, Phi, Psi, chol={}):
    mat_indices, par_indices = {}, {}
    mats = dict(LA=Lambda, BE=Beta, PH=Phi, PS=Psi)
    c = int(0)
    params = []
    bounds = []
    for key, value in mats.items():
        value = np.asarray(value).astype(float)
        if key in ['PH', 'PS']:
            a, b = np.where(value)
            off_diag = a!=b
            off_diag = a[off_diag], b[off_diag]
            value[off_diag] = 0.05
            if chol[key]:
                value = jax.numpy.linalg.cholesky(value)
                dgix = np.diag_indices(value.shape[0])
                value = jax.ops.index_update(value, dgix, np.log(value[dgix])+1.0)
                bound_i = [(None, None) for x,y in list(zip(*np.where(value)))]
                
            else:
                bound_i = [(0, None) if x==y else (None, None) 
                            for x,y in list(zip(*np.where(value)))]
        else:
            bound_i = [(None, None) for x,y in list(zip(*np.where(value)))]
        m_ind = np.where(value)
        mat_indices[key] = m_ind
        bounds = bounds + bound_i
        params.append(np.asarray(value)[m_ind].astype(float))
        n_par = len(m_ind[0])
        par_indices[key] = np.arange(c, c+n_par)
        c += n_par
    params = np.concatenate(params)
    params[par_indices['LA']] = 1e-3
    return mat_indices, par_indices, params, bounds




class SEM:
    
    def __init__(self, data, Lambda, Beta, Phi, Psi, indicator_vars=None):
        X, cols, indices, is_pd = _check_type(data)
        S = np.cov(X.T, bias=True)
        
        LA = np.asarray(indicator_vars).astype(float)
        BE = np.zeros_like(Beta).astype(float)
        PH = np.zeros_like(Phi).astype(float)
        PS = np.zeros_like(Psi).astype(float)
        
        
        mat_indices, par_indices, params, bounds = make_indices(Lambda, Beta, 
                                                                Phi, Psi, 
                                                                dict(PH=True,
                                                                     PS=False))
        tmat_indices, tpar_indices, theta, _ =  make_indices(Lambda, Beta, 
                                                             Phi, Psi, 
                                                             dict(PH=False,
                                                                  PS=False))
        self.bounds = bounds
        self.params = params
        self.theta = theta
        self.X, self.cols, self.ix, self.is_pd = X, cols, indices, is_pd
        self.mat_indices, self.par_indices = mat_indices, par_indices
        self.tmat_indices, self.tpar_indices = tmat_indices, tpar_indices
        self.LA = jax.numpy.asarray(LA)
        self.BE = jax.numpy.asarray(BE)
        self.PH = jax.numpy.asarray(PH)
        self.PS = jax.numpy.asarray(PS)
        self.model_matrices = dict(LA=LA, BE=BE, PH=PH, PS=PS)
        self.S = jax.numpy.asarray(S)
        
        self.n_obs, self.p = X.shape
        tlabels = []
        if is_pd:
            if type(Lambda) in [pd.DataFrame]:
                cols = Lambda.columns
                index = Lambda.index
            else:
                cols = ['nu%i'%i for i in range(Lambda.shape[1])]
                index = ['x%i'%i for i in range(Lambda.shape[1])]
            
            for x,y in list(zip(*self.tmat_indices['LA'])):
                xn, yn = index[x], cols[y]
                tlabels.append(f"{yn} ~ {xn}")
            for x,y in list(zip(*self.tmat_indices['BE'])):
                xn, yn = cols[x], cols[y]
                tlabels.append(f"{yn} ~ {xn}")
            for x,y in list(zip(*self.tmat_indices['PH'])):
                xn, yn = cols[x], cols[y]
                tlabels.append(f"var({yn} ~ {xn})")
            for x,y in list(zip(*self.tmat_indices['PS'])):
                xn, yn = index[x], index[y]
                tlabels.append(f"vav({yn} ~ {xn})")  
        
        
        
        
        
        
        self.tlabels = tlabels
        self._grad = jax.grad(self.loglike, argnums=0)
        self._hess = jax.hessian(self.loglike, argnums=0)
        self._hess_theta = jax.hessian(self.loglike_theta, argnums=0)
    
    def params2mats(self, params):
        LA, BE, PH, PS = self.LA, self.BE, self.PH, self.PS
        LA = jax.ops.index_update(LA, self.mat_indices['LA'],
                                  params[self.par_indices['LA']])
        BE = jax.ops.index_update(BE, self.mat_indices['BE'],
                                  params[self.par_indices['BE']]) 
        
        PH = jax.ops.index_update(PH, self.mat_indices['PH'],
                                  params[self.par_indices['PH']])
        
        PH = jax.ops.index_update(PH, jax.numpy.diag_indices(PH.shape[0]),
                                  jax.numpy.exp(PH[jax.numpy.diag_indices(PH.shape[0])]))
        PH = PH.dot(PH.T)
        PS = jax.ops.index_update(PS, self.mat_indices['PS'],
                                  params[self.par_indices['PS']])     
        dg = jax.numpy.diag(jax.numpy.diag(PS))
        PS = PS + PS.T - dg
        IB = jax.numpy.linalg.inv(mat_rconj(BE))
        return LA, IB, PH, PS
    
    
    def params2theta(self, params):
        theta = self.theta
        theta = jax.ops.index_update(theta, self.tpar_indices['LA'], 
                                     params[self.par_indices['LA']])
        
        theta = jax.ops.index_update(theta, self.tpar_indices['BE'], 
                                     params[self.par_indices['BE']])
        PH = self.PH 
        PH = jax.ops.index_update(PH, self.mat_indices['PH'],
                                  params[self.par_indices['PH']])
        PH = jax.ops.index_update(PH,  jax.numpy.diag_indices(PH.shape[0]),
                                  jax.numpy.exp(PH[jax.numpy.diag_indices(PH.shape[0])]))
        PH = PH.dot(PH.T)
        theta = jax.ops.index_update(theta, self.tpar_indices['PH'],
                                     PH[self.tmat_indices['PH']])
        PS = self.PS        
        PS = jax.ops.index_update(PS, self.mat_indices['PS'],
                                  params[self.par_indices['PS']])

        PS = jax.ops.index_update(PS, self.mat_indices['PS'],
                                  params[self.par_indices['PS']])     
        dg = jax.numpy.diag(jax.numpy.diag(PS))
        PS = PS + PS.T - dg
        theta = jax.ops.index_update(theta, self.tpar_indices['PS'],
                                     PS[self.tmat_indices['PS']])
        
        return theta
    
    
    def theta2mats(self, theta):
        LA, BE, PH, PS = self.LA, self.BE, self.PH, self.PS
        LA = jax.ops.index_update(LA, self.tmat_indices['LA'],
                                  theta[self.tpar_indices['LA']])
        BE = jax.ops.index_update(BE, self.tmat_indices['BE'],
                                  theta[self.tpar_indices['BE']]) 
        
        PH = jax.ops.index_update(PH, self.tmat_indices['PH'],
                                  theta[self.tpar_indices['PH']])
        
        PS = jax.ops.index_update(PS, self.tmat_indices['PS'],
                                  theta[self.tpar_indices['PS']])        
        IB = jax.numpy.linalg.inv(mat_rconj(BE))
        return LA, IB, PH, PS
    
    
    def implied_cov(self, LA, IB, PH, PS):
        Sigma = LA.dot(IB).dot(PH).dot(IB.T).dot(LA.T) + PS
        return Sigma
    
    def loglike(self, params, full=False):
        Sigma = self.implied_cov(*self.params2mats(params))
        Sigma_inv = jax.numpy.linalg.inv(Sigma)
        _, lndet = jax.numpy.linalg.slogdet(Sigma)
        trS = jax.numpy.trace(self.S.dot(Sigma_inv))
        ll = lndet + trS
        if full:
            ll = self.n_obs / 2 * ll
        return ll
    
    
    def loglike_theta(self, theta, full=False):
        Sigma = self.implied_cov(*self.theta2mats(theta))
        Sigma_inv = jax.numpy.linalg.inv(Sigma)
        _, lndet = jax.numpy.linalg.slogdet(Sigma)
        trS = jax.numpy.trace(self.S.dot(Sigma_inv))
        ll = lndet + trS
        if full:
            ll = self.n_obs / 2 * ll
        return ll
    
    def gradient(self, params, full=False):
        return self._grad(params, full)
    
    def hessian(self, params, full=False):
        return self._hess(params, full)
    
    def hessian_theta(self, theta, full=False):
        return self._hess_theta(theta, full)
    
    
    def _fit(self):
        self.optimizer = sp.optimize.minimize(self.loglike, self.params, 
                                              jac=self.gradient, 
                                              hess=self.hessian,
                                              method='trust-constr',
                                              options=dict(verbose=4,
                                                           gtol=1e-14,
                                                           xtol=1e-100),
                                              bounds=self.bounds)
        self.degfree = self.n_obs - len(self.theta)
        self.params = self.optimizer.x
        self.theta = self.params2theta(self.params)
        self.H = self.hessian_theta(self.theta, True)
        self.Hinv = np.linalg.inv(self.H)
        self.se_theta = np.sqrt(np.diag(self.Hinv))
        self.res = get_param_table(np.asarray(self.theta),
                                   np.asarray(self.se_theta), self.degfree)
        self.res.index = self.tlabels


for i in range(11):
    Psi.iloc[i, i] *=50


model = SEM(data, Lambda, Beta, Phi, Psi, indicator_vars)
model._fit()
model.res

