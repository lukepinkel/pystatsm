#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 17:53:00 2021

@author: lukepinkel
"""
import tqdm
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
from ..utilities.linalg_operations import vech, invech
from ..utilities.numerical_derivs import so_gc_cd

class VarCorrReparam:
    
    def __init__(self, dims, indices):
        gix, tix, dix, start = {}, {}, {}, 0
        
        for key, value in dims.items():
            n_vars = value['n_vars']
            n_params = int(n_vars * (n_vars+1) //2)
            i, j = np.triu_indices(n_vars)
            ix = np.arange(start, start+n_params)
            start += n_params
            gix[key] = {"v":np.diag_indices(n_vars), "r":np.tril_indices(n_vars, k=-1)}
            tix[key] = {"v":ix[i==j], "r":ix[i!=j]}
            i, j = np.tril_indices(n_vars)
            dix[key] = i[i!=j], j[i!=j]
        self.dims = dims
        self.ix = indices
        self.gix, self.tix, self.dix = gix, tix, dix
        self.n_pars = start+1
    
    def transform(self, theta):
        tau = theta.copy()
        for key in self.dims.keys():
            G = invech(theta[self.ix['theta'][key]])
            V = np.diag(np.sqrt(1.0/np.diag(G)))
            R = V.dot(G).dot(V)
            gixr, tixr = self.gix[key]['r'], self.tix[key]['r']
            gixv, tixv = self.gix[key]['v'], self.tix[key]['v']
            
            tau[tixr] = np.arctanh(R[gixr])
            tau[tixv] = G[gixv]
        tau[self.ix['theta']['resid']] = theta[self.ix['theta']['resid']]
        return tau
    
    def inverse_transform(self, tau):
        theta = tau.copy()
        for key in self.dims.keys():
            G = invech(tau[self.ix['theta'][key]])
            V = np.diag(np.sqrt(np.diag(G)))
            G[self.gix[key]['v']] = 1.0
            G[self.gix[key]['r']] = np.tanh(G[self.gix[key]['r']])
            G = V.dot(G).dot(V)
            theta[self.ix['theta'][key]] = vech(G)
        theta[self.ix['theta']['resid']] = tau[self.ix['theta']['resid']]
        return theta
    
    def jacobian(self, theta):
        tau = self.transform(theta)
        J = np.zeros((self.n_pars, self.n_pars))
        for key in self.dims.keys():
            G = invech(theta[self.ix['theta'][key]])
            tixr = self.tix[key]['r']
            gixv, tixv = self.gix[key]['v'], self.tix[key]['v']
            v = G[gixv]
            i, j = self.dix[key]
            si, sj = np.sqrt(v[i]), np.sqrt(v[j])
            J[tixv, tixv] = 1.0
            u = np.tanh(tau[tixr])
            J[tixr, tixr] = si * sj * (1-u**2)
            J[tixv[j], tixr] = u * si / (sj  * 2)
            J[tixv[i], tixr] = u * sj / (si * 2)
        J[self.ix['theta']['resid'], self.ix['theta']['resid']] = 1
        return J
    
    
def vcrepara_grad(tau, model, reparam):
    theta = reparam.inverse_transform(tau)
    J = reparam.jacobian(theta)
    g = model.gradient(theta)
    return J.dot(g)

class RestrictedModel:

    def __init__(self, model, reparam):
        self.model = model
        self.reparam = reparam
        self.tau = reparam.transform(model.theta.copy())
    
    def get_bounds(self, free_ix):
        bounds = np.asarray(self.model.bounds)[free_ix].tolist()
        return bounds
    
    def llgrad(self, tau_f, free_ix, t):
        tau = self.tau.copy()
        tau[free_ix] = tau_f
        tau[~free_ix] = t
        theta = self.reparam.inverse_transform(tau)
        ll = self.model.loglike(theta)
        g = self.model.gradient(theta)
        J = self.reparam.jacobian(theta)
        return ll, J.dot(g)[free_ix]
    
    
def profile(n_points, model, tb=3):
    theta = model.theta.copy()
    free_ix = np.ones_like(theta).astype(bool)
    reparam = VarCorrReparam(model.dims, model.indices) 
    rmodel = RestrictedModel(model, reparam)
    tau = reparam.transform(theta)

    n_theta = len(theta)
    llmax = model.loglike(model.theta.copy())
    
    H = so_gc_cd(vcrepara_grad, tau, args=(model, reparam,))
    se = np.diag(np.linalg.inv(H/2.0))**0.5
    thetas, zetas = np.zeros((n_theta*n_points, n_theta)), np.zeros(n_theta*n_points)
    k = 0
    pbar = tqdm.tqdm(total=n_theta*n_points, smoothing=0.001)
    for i in range(n_theta):
        free_ix[i] = False
        t_mle = tau[i]
        tau_r = tau.copy()
        if model.bounds[i][0]==0:
            lb = np.maximum(0.01, t_mle-tb*se[i])
        else:
            lb = t_mle - tb * se[i]
        ub = t_mle + tb * se[i]
        tspace = np.linspace(lb, ub, n_points)
        for t0 in tspace:
            x = tau[free_ix]
            func = lambda x: rmodel.llgrad(x, free_ix, t0)
            bounds = rmodel.get_bounds(free_ix)
            opt = sp.optimize.minimize(func, x, jac=True, bounds=bounds,
                                       method='trust-constr',
                                       options=dict(initial_tr_radius=0.5))
    
            tau_r[free_ix] = opt.x
            tau_r[~free_ix] = t0
            LR = (opt.fun - llmax)
            zeta = np.sqrt(LR) * np.sign(t0 - tau[~free_ix])
            zetas[k] = zeta
            thetas[k] = reparam.inverse_transform(tau_r)
            k+=1
            pbar.update(1)
        free_ix[i] = True
    pbar.close()
    ix = np.repeat(np.arange(n_theta), n_points)
    return thetas, zetas, ix 
    
    
def plot_profile(model, thetas, zetas, ix, quantiles=None, figsize=(16, 8)):
    if quantiles is None:
        quantiles = np.array([60, 70, 80, 90, 95, 99])
        quantiles = np.concatenate([(100-quantiles[::-1])/2, 100-(100-quantiles)/2])
    theta = model.theta.copy()
    se_theta = model.se_theta.copy()
    n_thetas = thetas.shape[1]
    q = sp.stats.norm(0, 1).ppf(np.array(quantiles)/100)
    fig, axes = plt.subplots(figsize=(14, 4), ncols=n_thetas, sharey=True)
    plt.subplots_adjust(wspace=0.05, left=0.05, right=0.95)
    for i in range(n_thetas):
        ax = axes[i]
        x = thetas[ix==i, i]
        y = zetas[ix==i]
        trunc = (y>-5)&(y<5)
        x, y = x[trunc], y[trunc]
        f_interp = sp.interpolate.interp1d(y, x, fill_value="extrapolate")
        xq = f_interp(q)
        ax.plot(x,y)
        ax.set_xlim(x.min(), x.max())
        ax.axhline(0, color='k')
        sgs = np.zeros((len(q), 2, 2))
        sgs[:, 0, 0] = sgs[:, 1, 0] = xq
        sgs[:, 1, 1] = q
        xqt = theta[i] + q * se_theta[i]
        ax.axvline(theta[i], color='k')
        norm = mpl.colors.TwoSlopeNorm(vcenter=0, vmin=q.min(), vmax=q.max())
        lc = mpl.collections.LineCollection(sgs, cmap=plt.cm.bwr, norm=norm)
        lc.set_array(q)
        lc.set_linewidth(2)
        ax.add_collection(lc)
        ax.scatter(xqt, np.zeros_like(xqt), c=q, cmap=plt.cm.bwr, norm=norm,
                   s=20)
    ax.set_ylim(-5, 5)
    return fig, axes
    
        
    







        
        
    
    
    
 