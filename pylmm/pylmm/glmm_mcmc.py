#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 20:00:38 2020

@author: lukepinkel
"""
import tqdm
import arviz as az
import numpy as np
import scipy as sp
import scipy.stats
import pandas as pd # analysis:ignore
from .lmm_chol import LMEC, make_theta
from ..utilities.linalg_operations import vech, _check_shape
from sksparse.cholmod import cholesky
from ..utilities.trnorm import trnorm
from ..utilities.wishart import r_invwishart, r_invgamma
from ..utilities.poisson import poisson_logp

    
def log1p(x):
    return np.log(1+x)


def rtnorm(mu, sd, lower, upper):
    a = (lower - mu) / sd
    b = (upper - mu) / sd
    return sp.stats.truncnorm(a=a, b=b, loc=mu, scale=sd).rvs()
    

def get_u_indices(dims): 
    u_indices = {}
    start=0
    for key, val in dims.items():
        q = val['n_groups']*val['n_vars']
        u_indices[key] = np.arange(start, start+q)
        start+=q
    return u_indices

def wishart_info(dims):
    ws = {}
    for key in dims.keys():
        ws[key] = {}
        q = dims[key]['n_groups']
        k = dims[key]['n_vars']
        nu = q-(k+1)
        ws[key]['q'] = q
        ws[key]['k'] = k
        ws[key]['nu'] = nu
    return ws

def sample_gcov(theta, u, wsinfo, indices, key, priors):
    u_i = u[indices['u'][key]]
    U_i = u_i.reshape(-1, wsinfo[key]['k'], order='C')
    Sg_i =  U_i.T.dot(U_i)
    Gs = r_invwishart(wsinfo[key]['nu']+priors[key]['n'], 
                             Sg_i+priors[key]['V'])
    theta[indices['theta'][key]] = vech(Gs)
    return theta

def sample_rcov(theta, y, yhat, wsinfo, priors):
    resid = y - yhat
    sse = resid.T.dot(resid)
    nu = wsinfo['r']['nu'] 
    ss =r_invgamma((nu+priors['R']['n']), 
                            scale=(sse+priors['R']['V']))
    theta[-1] = ss 
    return theta





class MixedMCMC(LMEC):
    
    def __init__(self, formula, data, priors=None):
        super().__init__(formula, data) 
        self.t_init, _ = make_theta(self.dims)
        self.W = sp.sparse.csc_matrix(self.XZ)
        self.wsinfo = wishart_info(self.dims)
        self.y = _check_shape(self.y, 1)
        
        self.indices['u'] = get_u_indices(self.dims)
        self.n_re = self.G.shape[0]
        self.n_fe = self.X.shape[1]
        self.n_lc = self.W.shape[1]
        self.n_ob = self.W.shape[0]
        self.re_mu = np.zeros(self.n_re)
        self.n_params = len(self.t_init)+self.n_fe
        if priors is None:
            priors = dict(R=dict(V=0.500*self.n_ob, n=self.n_ob))
            for level in self.levels:
                Vi = np.eye(self.dims[level]['n_vars'])*0.001
                priors[level] = dict(V=Vi, n=4)
        self.priors = priors
        self.wsinfo['r'] = dict(nu=self.n_ob-2)
        self.offset = np.zeros(self.n_lc)
        self.location = np.zeros(self.n_re+self.n_fe)
        self.zero_mat = sp.sparse.eye(self.n_fe)*0.0
        self.ix0, self.ix1 = self.y==0, self.y==1
        self.jv0, self.jv1 = np.ones(len(self.ix0)), np.ones(len(self.ix1))
    
    def sample_location(self, theta, x1, x2, y):
        """

        Parameters
        ----------
        theta: array_like
            Vector of covariance parameters
        
        x1: array_like
            Sample from a standard normal distribution 
        
        x2: array_like
            Sample from a standard normal distribution 
        
        y: array_like
            Dependent variable
        
        Returns
        -------
        location: array_like
            Sample from P(beta, u|y, G, R)
        
        Notes
        -----
        
        
        
        """
        s, s2 =  np.sqrt(theta[-1]), theta[-1]
        WtR = self.W.copy().T / s2
        M = WtR.dot(self.W)
        Ginv = self.update_gmat(theta, inverse=True).copy()
        Omega = sp.sparse.block_diag([self.zero_mat, Ginv])
        M+=Omega
        chol_fac = cholesky(Ginv, ordering_method='natural')
        a_star = chol_fac.solve_Lt(x1, use_LDLt_decomposition=False)
        y_z = y - (self.Z.dot(a_star) + x2 * s)
        ofs = self.offset.copy()
        ofs[-self.n_re:] = a_star
        location = ofs + sp.sparse.linalg.spsolve(M, WtR.dot(y_z))
        return location
    
    def mh_lvar_binomial(self, pred, s, z, x_step, u_accept, propC):
        z_prop = x_step * propC + z
        
        mndenom1, mndenom2 = np.exp(z)+1, np.exp(z_prop)+1
        densityl1 = (self.y*z) - np.log(mndenom1) 
        densityl2 = (self.y*z_prop) - np.log(mndenom2)
        densityl1 += sp.stats.norm(pred, s).logpdf(z)
        densityl2 += sp.stats.norm(pred, s).logpdf(z_prop)
        density_diff = densityl2 - densityl1
        accept = (density_diff>u_accept)
        z[accept] = z_prop[accept]
        return z, accept
    
    def mh_lvar_poisson(self, pred, s, z, x_step, u_accept, propC):
        z_prop = x_step * propC + z
        
        densityl1 = poisson_logp(self.y, np.exp(z))
        densityl2 = poisson_logp(self.y, np.exp(z_prop))
        densityl1 += sp.stats.norm(pred, s).logpdf(z)
        densityl2 += sp.stats.norm(pred, s).logpdf(z_prop)
        density_diff = densityl2 - densityl1
        accept = (density_diff>u_accept)
        z[accept] = z_prop[accept]
        return z, accept
    
    def sample_theta(self, theta, u, z, pred, freeR=True):
        for key in self.levels:
            theta = sample_gcov(theta.copy(), u, self.wsinfo, self.indices,
                                key, self.priors)
        if freeR:
            theta = sample_rcov(theta, z, pred, self.wsinfo, self.priors)
        return theta
    
    def slice_sample_lvar(self, rexpon, v, z, theta, pred):
        v[self.ix1] = z[self.ix1] - log1p(np.exp(z[self.ix1]))
        v[self.ix1]-= rexpon[self.ix1]
        v[self.ix1] = v[self.ix1] - log1p(-np.exp(v[self.ix1]))
        
        v[self.ix0] = -log1p(np.exp(z[self.ix0]))
        v[self.ix0]-= rexpon[self.ix0]
        v[self.ix0] = log1p(-np.exp(v[self.ix0])) - v[self.ix0]
        s = np.sqrt(theta[-1])
        z[self.ix1] = trnorm(mu=pred[self.ix1], sd=s*self.jv1, 
                             lb=v[self.ix1], ub=200*self.jv1)
        z[self.ix0] = trnorm(mu=pred[self.ix0], sd=s*self.jv0, 
                             lb=-200*self.jv0, ub=v[self.ix0])
        return z
   
    def sample_mh_gibbs(self, n_samples, propC=1.0, chain=0, store_z=False, freeR=True):
        param_samples = np.zeros((n_samples, self.n_params))
        acceptances = np.zeros((n_samples, self.n_ob))
        if store_z:
            z_samples = np.zeros((n_samples, self.n_ob))
        
        x_step = sp.stats.norm(0.0, 1.0).rvs((n_samples, self.n_ob))
        x_astr = sp.stats.norm(0.0, 1.0).rvs((n_samples, self.n_ob))
        x_ranf = sp.stats.norm(0.0, 1.0).rvs((n_samples, self.n_re))
        u_accp = np.log(sp.stats.uniform(0, 1).rvs(n_samples, self.n_ob))
        
        location = self.location.copy()
        pred =  self.W.dot(location)
        z = sp.stats.norm(self.y, self.y.var()).rvs()
        theta = self.t_init.copy()
       
        progress_bar = tqdm.tqdm(range(n_samples))
        counter = 1
        for i in progress_bar:
            s2 = theta[-1]
            s = np.sqrt(s2)
            z, accept = self.mh_lvar_binomial(pred, s, z, x_step[i], u_accp[i], propC)
            location = self.sample_location(theta, x_ranf[i], x_astr[i], z)
            pred = self.W.dot(location)
            u = location[-self.n_re:]
            theta  = self.sample_theta(theta, u, z, pred, freeR)
            
            param_samples[i, self.n_fe:] = theta.copy()
            param_samples[i, :self.n_fe] = location[:self.n_fe]
            acceptances[i] = accept
            if store_z:
                z_samples[i] = z
            counter+=1
            if counter==1000:
                acceptance = np.sum(acceptances)/(float((i+1)*self.n_ob))
                progress_bar.set_description(f"Chain {chain} Acceptance Prob: {acceptance:.4f}")
                counter = 1
        progress_bar.close()
        
        if store_z:
            return param_samples, acceptances, z_samples
        else:
            return param_samples, acceptances
        
    def sample_slice_gibbs(self, n_samples, chain=0, save_pred=False, save_u=False, save_lvar=False,
                           freeR=False):
        normdist = sp.stats.norm(0.0, 1.0).rvs

        n_pr, n_ob, n_re = self.n_params, self.n_ob, self.n_re
        n_smp = n_samples
        samples = np.zeros((n_smp, n_pr))
        
        x_astr, x_ranf = normdist((n_smp, n_ob)), normdist((n_smp, n_re))
        rexpon = sp.stats.expon(scale=1).rvs((n_smp, n_ob))
        
        location, pred = self.location.copy(), self.W.dot(self.location)
        theta, z = self.t_init.copy(), sp.stats.norm(0, 1).rvs(n_ob)
        secondary_samples = {}
        if save_pred:
            secondary_samples['pred'] = np.zeros((n_smp, n_ob))
        if save_u:
            secondary_samples['u'] = np.zeros((n_smp, n_re))
        if save_lvar:
            secondary_samples['lvar'] = np.zeros((n_smp, n_ob))
        v = np.zeros_like(z).astype(float)
        progress_bar = tqdm.tqdm(range(n_smp))
        progress_bar.set_description(f"Chain {chain}")
        for i in progress_bar:
            #P(z|location, theta)
            z = self.slice_sample_lvar(rexpon[i], v, z, theta, pred)
            #P(location|z, theta)
            location = self.sample_location(theta, x_ranf[i], x_astr[i], z)
            pred, u = self.W.dot(location), location[-self.n_re:]
            #P(theta|z, location)
            theta  = self.sample_theta(theta, u, z, pred, freeR)
            samples[i, self.n_fe:] = theta.copy()
            samples[i, :self.n_fe] = location[:self.n_fe]
            if save_pred:
                secondary_samples['pred'][i] = pred
            if save_u:
                secondary_samples['u'][i] = u
            if save_lvar:
                secondary_samples['lvar'][i] = z
        progress_bar.close()
        return samples, secondary_samples
    
    def gibbs_normal(self, n_samples, chain=0, save_pred=False, save_u=False, save_lvar=False,
                     freeR=True):
        normdist = sp.stats.norm(0.0, 1.0).rvs

        n_pr, n_ob, n_re = self.n_params, self.n_ob, self.n_re
        n_smp = n_samples
        samples = np.zeros((n_smp, n_pr))
        
        x_astr, x_ranf = normdist((n_smp, n_ob)), normdist((n_smp, n_re))
        y = self.y
        location, pred = self.location.copy(), self.W.dot(self.location)
        theta = self.t_init.copy()
        secondary_samples = {}
        if save_pred:
            secondary_samples['pred'] = np.zeros((n_smp, n_ob))
        if save_u:
            secondary_samples['u'] = np.zeros((n_smp, n_re))
        progress_bar = tqdm.tqdm(range(n_smp))
        progress_bar.set_description(f"Chain {chain}")
        for i in progress_bar:
            #P(location|y, theta)
            location = self.sample_location(theta, x_ranf[i], x_astr[i], y)
            pred, u = self.W.dot(location), location[-self.n_re:]
            #P(theta|z, location)
            theta  = self.sample_theta(theta, u, y, pred, freeR)
            samples[i, self.n_fe:] = theta.copy()
            samples[i, :self.n_fe] = location[:self.n_fe]
            if save_pred:
                secondary_samples['pred'][i] = pred
            if save_u:
                secondary_samples['u'][i] = u
        progress_bar.close()
        return samples, secondary_samples
    
    def fit(self, n_samples=5000, n_chains=8, burnin=1000, vnames=None, sample_kws={},
            method='MH-Gibbs'):
        samples = np.zeros((n_chains, n_samples, self.n_params))        
        if vnames is None:
            vnames =  {"$\\beta$":np.arange(self.n_fe), 
                       "$\\theta$":np.arange(self.n_fe, self.n_params)}
    
        if method=='MH-Gibbs':
            func = self.sample_mh_gibbs
        elif method=='Normal-Gibbs':
            func = self.gibbs_normal
        elif method=='Slice-Gibbs':
            func = self.sample_slice_gibbs
        for i in range(n_chains):
            samples[i], _ = func(n_samples, chain=i, **sample_kws)

        az_dict = to_arviz_dict(samples,  vnames, burnin=burnin)
        az_data = az.from_dict(az_dict)
        summary = az.summary(az_data, hdi_prob=0.95)
        return samples, az_data, summary
           
            
        
                    
                        


def to_arviz_dict(samples, var_dict, burnin=0):
    az_dict = {}
    for key, val in var_dict.items():
        az_dict[key] = samples[:, burnin:, val]
    return az_dict    

            
                
        
        
        
        
                
                
                
                
                        
                        
                        
                        
                
                


    




    
    
    
    
    
    


