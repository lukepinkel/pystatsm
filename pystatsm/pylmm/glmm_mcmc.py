#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 20:58:05 2022

@author: lukepinkel
"""

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
import matplotlib.pyplot as plt
from .lmm import LMM
from ..utilities.data_utils import _check_shape, sign_change
from ..utilities.linalg_operations import vech
from sksparse.cholmod import cholesky
from ..utilities.random import trnorm, r_invwishart, r_invgamma
from ..utilities.func_utils import poisson_logp, log1p, norm_cdf
from .ranef_terms import RandomEffects, RandomEffectTerm
from ..utilities.linalg_operations import invech_chol, invech, vech
from ..utilities.numerical_derivs import fo_fc_cd, so_fc_cd, so_gc_cd
from ..utilities.special_mats import lmat
from ..utilities.formula import parse_random_effects





def to_arviz_dict(samples, var_dict, burnin=0):
    az_dict = {}
    for key, val in var_dict.items():
        az_dict[key] = samples[:, burnin:, val]
    return az_dict    


def _plot_trace(idata, plot_kws=None):
    default_plot_kws = dict(combined=False, compact=False, figsize=(12, 8))
    plot_kws = {} if plot_kws is None else plot_kws
    plot_kws = {**default_plot_kws, **plot_kws} 
    axes = az.plot_trace(idata, **plot_kws)
    n = axes.shape[0]
    for i in range(n):
        label = axes[i, 0].get_title().replace("\n", "[")+"]"
        axes[i, 0].set_title("")
        axes[i, 1].set_title("")
        axes[i, 0].set_ylabel(label, rotation=0, labelpad=15)
    plt.subplots_adjust(left=0.04, right=0.97, top=0.97, bottom=0.04, wspace=0.1)
    return axes
        
    
    
    
def _plot_ess(idata, plot_kws=None):
    default_plot_kws = dict(figsize=(16, 8))
    plot_kws = {} if plot_kws is None else plot_kws
    plot_kws = {**default_plot_kws, **plot_kws} 
    
    axes = az.plot_ess(idata, **plot_kws)
    if axes.ndim==1:
        axes = axes.reshape(1, -1)
    n, m = axes.shape
    for i in range(n):
        for j in range(m):
            label = axes[i, j].get_title().replace("\n", "[")+"]"
            axes[i, j].set_title(label)
            axes[i, j].xaxis.set_tick_params(labelsize=8)
            axes[i, j].yaxis.set_tick_params(labelsize=8)
            if j>0:
                axes[i, j].set_ylabel("")
            else:
                label = axes[i, j].get_ylabel()
                axes[i, j].set_ylabel(label, size=12)
                
            if (i+1)<n:
                axes[i, j].set_xlabel("")
            else:
                label = axes[i, j].get_xlabel()
                axes[i, j].set_xlabel(label, size=12)
        
    plt.subplots_adjust(left=0.08, right=0.97, top=0.97, bottom=0.1, wspace=0.15, 
                        hspace=0.15)
    return axes
 
    
def _plot_rank(idata, plot_kws=None):
    default_plot_kws = dict(figsize=(12, 10))
    plot_kws = {} if plot_kws is None else plot_kws
    plot_kws = {**default_plot_kws, **plot_kws} 
    
    axes = az.plot_rank(idata, **plot_kws)
    if axes.ndim==1:
        axes = axes.reshape(1, -1)
    n, m = axes.shape
    for i in range(n):
        for j in range(m):
            label = axes[i, j].get_title().replace("\n", "[")+"]"
            axes[i, j].set_title(label)
            axes[i, j].xaxis.set_tick_params(labelsize=8)
            axes[i, j].yaxis.set_tick_params(labelsize=8)
            if j>0:
                axes[i, j].set_ylabel("")
            else:
                label = axes[i, j].get_ylabel()
                axes[i, j].set_ylabel(label, size=12)
                
            if (i+1)<n:
                axes[i, j].set_xlabel("")
            else:
                label = axes[i, j].get_xlabel()
                axes[i, j].set_xlabel(label, size=12)
        
    plt.subplots_adjust(left=0.08, right=0.97, top=0.97, bottom=0.08, wspace=0.15, 
                        hspace=0.15)
    return axes
        
         

class MixedMCMC(LMM):
    
    def __init__(self, formula, data, response_dist, priors=None, weights=None, 
                 rng=None, vnames=None, freeR=None):
        """
        

        Parameters
        ----------
        formula : string
            lme4 style formula specifying random effects with parentheses and 
            a vertical bar.
        data : dataframe
            Dataframe containing data.  Missing values should be dropped 
            manually before passing the dataframe.
        response_dist : string
            String specifying the type of model. One of `ordinal_probit`,
            `bernoulli`, `binomial`, or `normal`.
        priors : dict, optional
            A dictionary with keys consisting of random effect factors or 
            `R` for the residual covariance. The corresponding value should be
            a dictionary with keys `V` and `n`, specifying the prior covariance
            matrix and number of observations. The default is None.
        weights : ndarray, optional
            Weight matrix. The default is None.
        rng : numpy.random._generator.Generator, optional
            Numpy random generator. The default is None.
        vnames : list, optional
            List of variable names. The default is None.
        freeR : bool, optional
            If true, allows the residual covariance to vary in `binomial` and
            `bernoulli` models. The default is None.

        Returns
        -------
        None.

        """
        super().__init__(formula, data) 
        
        #Initialize parameters, get misc information for sampling
        self.t_init = self.theta.copy()
        self.t_init[-1] = 1.0
        self.W = sp.sparse.hstack([self.X, self.Z]).tocsc()
        self.WtW = self.W.T.dot(self.W)
        self.wsinfo = self.random_effects.wishart_info()
        self.y = _check_shape(self.y, 1)
        self.u_indices = self.random_effects.get_u_indices()
        
        if weights is None:
            self.weights = np.ones_like(self.y)
        else:
            self.weights=weights
            
        #Get various model constants 
        self.n_re = self.G.shape[0]
        self.n_fe = self.X.shape[1]
        self.n_lc = self.W.shape[1]
        self.n_ob = self.W.shape[0]
        self.n_params = len(self.t_init)+self.n_fe
        
        #Handle priors
        if priors is None:
            if response_dist == 'normal':
                priors = dict(R=dict(V=0, n=-2))
            else:
                priors = dict(R=dict(V=0.500*self.n_ob, n=self.n_ob))
            for i in range(self.levels):
                Vi = np.eye(self.random_effects.n_rvars[i])*0.001
                priors[i] = dict(V=Vi, n=4)
        self.priors = priors
        self.wsinfo['r'] = dict(nu=self.n_ob-2)
        
        #Initialize containers requiring constants above
        self.offset = np.zeros(self.n_lc)
        self.location = np.zeros(self.n_re+self.n_fe)
        self.zero_mat = sp.sparse.eye(self.n_fe)*0.0
        self.re_mu = np.zeros(self.n_re)
        
        #Get model specific indices and constants
        if response_dist == 'bernoulli':
            self.ix0, self.ix1 = self.y==0, self.y==1
            self.jv0, self.jv1 = np.ones(len(self.ix0)), np.ones(len(self.ix1))
            freeR = False if freeR is None else freeR
        elif response_dist == 'ordinal_probit':
            self.jv = np.ones(self.n_ob)
            self.n_cats = np.unique(self.y).shape[0]
            self.n_thresh = self.n_cats - 1
            self.n_tau = self.n_cats + 1
            self.y_cat = np.zeros((self.n_ob, self.n_cats))
            self.y_ix = {}
            for i in range(self.n_cats):
                self.y_ix[i] = self.y==i
                self.y_cat[self.y==i, i] = 1
            freeR = False if freeR is None else freeR
        elif response_dist == 'binomial':
            freeR = False if freeR is None else freeR
        elif response_dist == 'normal':
            freeR = True if freeR is None else freeR
            
        self.response_dist = response_dist
        self.freeR = freeR
        self.rng = np.random.default_rng() if rng is None else rng
        
        
        #Handle variable names for arviz and variable names for results
        if vnames is None:
            vnames = {"$\\beta$":np.arange(self.n_fe), 
                       "$\\theta$":np.arange(self.n_fe, self.n_params)}
            if response_dist == 'ordinal_probit':
                    vnames["$\\tau$"]=np.arange(self.n_params+1, self.n_params+self.n_thresh)
            if not self.freeR:
                vnames["$\\theta$"] = np.arange(self.n_fe, self.n_params-1)
        self.vnames = vnames

        param_names = list(self.fe_vars)
        for i in range(self.levels):
            group = self.random_effects.terms[i].gr_form
            for i, j in list(zip(*np.triu_indices(self.random_effects.n_rvars[i]))):
                param_names.append(f"{group}:G[{i}][{j}]")
        
        if self.freeR:
            param_names.append("resid_cov")
        if response_dist=='ordinal_probit':
            param_names = param_names + [f"t{i}" for i in range(1, self.n_thresh)]
        self.param_names = param_names
        

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
        
        """
        s, s2 =  np.sqrt(theta[-1]), theta[-1]
        M = self.WtW.copy() / s2
        Ginv = self.update_gmat(theta, inverse=True).copy()
        Omega = sp.sparse.block_diag([self.zero_mat, Ginv])
        M+=Omega
        chol_fac = cholesky(Ginv, ordering_method='natural')
        a_star = chol_fac.solve_Lt(x1, use_LDLt_decomposition=False)
        y_z = y - (self.Z.dot(a_star) + x2 * s)
        ofs = self.offset.copy()
        ofs[-self.n_re:] = a_star
        u = sp.sparse.csc_matrix.dot(y_z, self.W) / s2
        if M.nnz / np.prod(M.shape) > 0.05:
            m_chol = sp.linalg.cholesky(M.toarray(), lower=True)
            location = ofs + sp.linalg.cho_solve((m_chol, True), u)
        else:
            m_chol = cholesky(M.tocsc())
            location = ofs + m_chol.solve_A(u) 
        return location
    
    def mh_lvar_binomial(self, pred, s, z, x_step, u_accept, propC):
        z_prop = x_step * propC + z
        
        mndenom1, mndenom2 = np.exp(z)+1, np.exp(z_prop)+1
        densityl1 = ((self.y*z) - np.log(mndenom1))*self.weights
        densityl2 = ((self.y*z_prop) - np.log(mndenom2))*self.weights
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
    
    def mh_lvar_ordinal_probit(self, theta, t, pred, v):
        tau = np.pad(t, ((1, 1)), mode='constant', constant_values=[-1e17, 1e17])
        mu, sd = np.zeros_like(pred), np.ones_like(pred)
        s = np.sqrt(theta[-1])
        for i in range(1, self.y_cat.shape[1]+1):
            ix = self.y_ix[i-1]
            j = self.jv[ix]
            mu = pred[ix]
            sd = j*s
            lb = j*tau[i-1]
            ub = j*tau[i]
            v[ix] = trnorm(mu, sd, lb, ub)
        return v
        
        
    def sample_gcov(self, theta, u, i):
        u_i = u[self.u_indices[i]]
        U_i = u_i.reshape(-1, self.wsinfo[i]['k'], order='C')
        Sg_i =  U_i.T.dot(U_i)
        Gs = r_invwishart(self.wsinfo[i]['nu']+self.priors[i]['n'], Sg_i+self.priors[i]['V'])
        theta[self.t_inds[i]] = vech(Gs)
        return theta
    
    def sample_rcov(self, theta, y, yhat):
        resid = y - yhat
        sse = resid.T.dot(resid)
        nu = self.wsinfo['r']['nu'] 
        ss = r_invgamma((nu + self.priors['R']['n'])/2, scale=(sse + self.priors['R']['V'])/2)
        theta[-1] = ss 
        return theta

    def sample_theta(self, theta, u, z, pred, freeR=True):
        for i in range(self.levels):
            theta = self.sample_gcov(theta.copy(), u, i)
        if freeR:
            theta = self.sample_rcov(theta, z, pred)
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
    
    
    def sample_tau(self, theta, t, pred, v, propC):
        alpha_prev = np.pad(np.log(np.diff(t)), (1, 0), mode='constant', constant_values=[np.log(t[0])])
        alpha_prop = self.rng.normal(alpha_prev, propC)
        ll = 0.0
        t_prop = np.cumsum(np.exp(alpha_prop))
        s = np.sqrt(theta[-1])
        for i in range(1, self.n_thresh):
            m = pred[self.y_ix[i]]
            ll += (np.log(norm_cdf((t_prop[i]-m) / s)-norm_cdf((t_prop[i-1]-m) / s))).sum()
            ll -= (np.log(norm_cdf((t[i]-m)/s)-norm_cdf((t[i-1]-m)/s))).sum()
        m = pred[self.y_ix[i+1]]
        ll += np.sum(np.log(1.0 - norm_cdf((t_prop[i]-m)/s)))
        ll -= np.sum(np.log(1.0 - norm_cdf((t[i]-m)/s)))
        if ll>np.log(np.random.uniform(0, 1)):
            t_accept = True
            t = t_prop
        else:
            t_accept = False
            t = t
        return t, t_accept 
    
    def sample_ordinal_probit(self, n_samples, chain=0, save_pred=False, 
                              save_u=False, save_lvar=False, propC=0.04, damping=0.99,
                              adaption_rate=1.02,  target_accept=0.44, n_adapt=None):

        if n_adapt is None:
            n_adapt = np.minimum(int(n_samples/2), 1000)
        
        freeR = self.freeR
        param_samples = np.zeros((n_samples, self.n_params+self.n_thresh))
        t_acceptances = np.zeros((n_samples))
        rng = self.rng
        location = self.location.copy()
        pred =  self.W.dot(location)
        theta = self.t_init.copy()
        t = sp.stats.norm(0, 1).ppf((self.y_cat.sum(axis=0).cumsum()/np.sum(self.y_cat))[:-1])
        t = t - t[0]
        t = t+1.0
        z = np.zeros_like(self.y).astype(float)
        wtrace, waccept = 1.0, 1.0
        pbar = tqdm.tqdm(range(n_samples), smoothing=0.01)
        for i in range(n_samples):
            t, t_accept = self.sample_tau(theta, t, pred, z, propC)
            wtrace = wtrace * damping + 1.0
            waccept *= damping
            if t_accept:
                waccept +=1
            if i<n_adapt:
                propC = propC * np.sqrt(adaption_rate**((waccept/wtrace)-target_accept))
            if t_accept:
                z = self.mh_lvar_ordinal_probit(theta, t, pred, z)
            location = self.sample_location(theta, rng.normal(0, 1, size=self.n_re), 
                                            rng.normal(0, 1, size=self.n_ob), z)
            pred = self.W.dot(location)
            u = location[-self.n_re:]
            theta  = self.sample_theta(theta, u, z, pred, freeR)
            param_samples[i, self.n_fe:-self.n_thresh] = theta.copy()
            param_samples[i, :self.n_fe] = location[:self.n_fe]
            param_samples[i, -self.n_thresh:] = t
            t_acceptances[i] = t_accept
            self._secondary_samples(chain, i, pred, u, z)
            if i>1:
                pbar.set_description(f"Chain {chain+1} Tau Acceptance Prob: {t_acceptances[:i].mean():.3f} C: {propC:.5f}")
            pbar.update(1)
        pbar.close() 
        return param_samples
    
    def sample_binomial(self, n_samples, chain=0, save_pred=False, save_u=False,
                        save_lvar=False, propC=1.0, damping=0.99, 
                        adaption_rate=1.01,  target_accept=0.44,
                        n_adapt=None):
        freeR = self.freeR
        if n_adapt is None:
            n_adapt = np.minimum(int(n_samples/2), 1000)
            
        param_samples = np.zeros((n_samples, self.n_params))
        rng = self.rng
        acceptances = np.zeros((n_samples))
        location = self.location.copy()
        pred =  self.W.dot(location)
        z = sp.stats.norm(self.y, self.y.var()).rvs()
        theta = self.t_init.copy()
        progress_bar = tqdm.tqdm(range(n_samples), smoothing=0.01)
        wtrace, waccept = 1.0, 1.0

        for i in progress_bar:
            s2 = theta[-1]
            s = np.sqrt(s2)
            z, accept = self.mh_lvar_binomial(pred, s, z, rng.normal(0, 1, size=self.n_ob),
                                              np.log(rng.uniform(0, 1, size=self.n_ob)), propC)
            
            mean_accept = accept.mean()
            wtrace = wtrace * damping + 1.0
            waccept = waccept * damping + mean_accept

            if i<n_adapt:
                propC = propC * np.sqrt(adaption_rate**((waccept/wtrace)-target_accept))
                
            location = self.sample_location(theta, rng.normal(0, 1, size=self.n_re), 
                                            rng.normal(0, 1, size=self.n_ob), z)
            pred = self.W.dot(location)
            u = location[-self.n_re:]
            theta = self.sample_theta(theta, u, z, pred, freeR)
            acceptances[i] = mean_accept
            param_samples[i, self.n_fe:] = theta.copy()
            param_samples[i, :self.n_fe] = location[:self.n_fe]
            self._secondary_samples(chain, i, pred, u, z)
            if i>1:
                progress_bar.set_description(f"Chain {chain+1} Acceptance Prob: {acceptances[:i].mean():.4f} C: {propC:.5f}")
        progress_bar.close()
        return param_samples
    
    def sample_bernoulli(self, n_samples, chain=0, save_pred=False, save_u=False, 
                         save_lvar=False):
        freeR = self.freeR
        rng = self.rng
        n_pr, n_ob = self.n_params, self.n_ob
        n_smp = n_samples
        samples = np.zeros((n_smp, n_pr))
        location, pred = self.location.copy(), self.W.dot(self.location)
        theta, z = self.t_init.copy(), sp.stats.norm(0, 1).rvs(n_ob)
        v = np.zeros_like(z).astype(float)
        progress_bar = tqdm.tqdm(range(n_smp), smoothing=0.01)
        progress_bar.set_description(f"Chain {chain+1}")
        for i in progress_bar:
            #P(z|location, theta)
            z = self.slice_sample_lvar(rng.exponential(scale=1.0, size=self.n_ob),
                                       v, z, theta, pred)
            #P(location|z, theta)
            location = self.sample_location(theta, rng.normal(0, 1, size=self.n_re), 
                                             rng.normal(0, 1, size=self.n_ob), z)
            pred, u = self.W.dot(location), location[-self.n_re:]
            #P(theta|z, location)
            theta  = self.sample_theta(theta, u, z, pred, freeR)
            samples[i, self.n_fe:] = theta.copy()
            samples[i, :self.n_fe] = location[:self.n_fe]
            self._secondary_samples(chain, i, pred, u, z)
        progress_bar.close()
        return samples
    
    def sample_normal(self, n_samples, chain=0, save_pred=False, save_u=False, 
                      save_lvar=False):
        freeR = self.freeR
        rng = self.rng
        n_pr= self.n_params
        n_smp = n_samples
        samples = np.zeros((n_smp, n_pr))
        
        y = self.y
        location, pred = self.location.copy(), self.W.dot(self.location)
        theta = self.t_init.copy()
        progress_bar = tqdm.tqdm(range(n_smp), smoothing=0.01)
        progress_bar.set_description(f"Chain {chain+1}")
        for i in progress_bar:
            #P(location|y, theta)
            location = self.sample_location(theta, rng.normal(0, 1, size=self.n_re), 
                                             rng.normal(0, 1, size=self.n_ob), y)
            pred, u = self.W.dot(location), location[-self.n_re:]
            #P(theta|z, location)
            theta  = self.sample_theta(theta, u, y, pred, freeR)
            samples[i, self.n_fe:] = theta.copy()
            samples[i, :self.n_fe] = location[:self.n_fe]
            self._secondary_samples(chain, i, pred, u, None)
        progress_bar.close()
        return samples
           
    
    def _secondary_samples(self, chain, i, pred=None, u=None, z=None):
        if self.save_pred:
            self.secondary_samples["pred"][chain, i] = pred
        if self.save_u:
            self.secondary_samples["u"][chain, i] = u        
        if self.save_lvar:
            self.secondary_samples["lvar"][chain, i] = z
            
    def sample(self, n_samples=5000, n_chains=8, burnin=1000, save_pred=False, 
               save_u=False, save_lvar=False, sampling_kws={},
               summary_kws={}):
        n_params = self.n_params
        if self.response_dist=="ordinal_probit":
            n_params = n_params+np.unique(self.y).shape[0]-1
        samples = np.zeros((n_chains, n_samples, n_params))

        self.secondary_samples = {}
        if save_u:
            self.secondary_samples['u'] = np.zeros((n_chains, n_samples, self.n_re))
        if save_pred:
            self.secondary_samples['pred'] = np.zeros((n_chains, n_samples, self.n_ob))        
        if save_lvar:
            self.secondary_samples['lvar'] = np.zeros((n_chains, n_samples, self.n_ob))
        self.save_pred, self.save_u, self.save_lvar = save_pred, save_u, save_lvar

        if self.response_dist=='binomial':
            func = self.sample_binomial
        elif self.response_dist=='bernoulli':
            func = self.sample_bernoulli
        elif self.response_dist=='ordinal_probit':
            func = self.sample_ordinal_probit
        elif self.response_dist=='normal':
            func = self.sample_normal
            
        for i in range(n_chains):
            samples[i] = func(n_samples, chain=i, **sampling_kws)
        
        #Heuristic for a heuristic...
        if burnin=="heuristic":
            tmp = []
            for i in np.concatenate(list(self.vnames.values())):
                xv = samples[:, :, i].T
                xm = np.mean(xv, axis=0)
                tmp.append(sign_change(xv, nth_change=50, offset=xm))
            burnin = np.max(tmp)
        
        if self.response_dist == 'bernoulli':
            c = np.sqrt(1.0+(16*np.sqrt(3))/(15*np.pi))
            if save_u:
                self.secondary_samples['u'] /= c
            if save_pred:
                self.secondary_samples['pred'] /=c
            samples[:, :, self.vnames["$\\beta$"]] /= c
            samples[:, :, self.vnames["$\\theta$"]] /= c
        
        self.burnin = burnin
        self.samples = samples
        self.az_dict = to_arviz_dict(samples, self.vnames, burnin=burnin)
        self.az_data = az.from_dict(self.az_dict)
        self.summary = az.summary(self.az_data, round_to=6, **summary_kws)
        self.res = self.summary.copy()
        self.res.index = self.param_names
        self.res.insert(2, "z-value", self.res["mean"]/self.res["sd"])
        self.res.insert(3, "p-value", sp.stats.norm(0, 1).sf(np.abs(self.res["z-value"])))
        self.beta = np.mean(samples[:, burnin:, self.vnames["$\\beta$"]], axis=(0, 1))
        self.theta = np.mean(samples[:, burnin:, self.vnames["$\\theta$"]], axis=(0, 1))
        if self.response_dist == 'ordinal_probit':
            self.tau = np.mean(samples[:, burnin:, self.vnames["$\\tau$"]], axis=(0, 1))
            
    
    def aggregate_samples(self, burnin=None):
        burnin = self.burnin if burnin is None else burnin
        agg_samples = {}
        for name, ix in self.vnames.items():
            x = self.samples[:, burnin, ix]
            x = x.reshape(np.product(x.shape[:-1]), x.shape[-1])
            agg_samples[name] = x
        
        if 'u' in self.secondary_samples.keys():
            u = self.secondary_samples["u"][:, burnin:]
            u = u.reshape(np.product(u.shape[:-1]), u.shape[-1])
        agg_samples['u'] = u
        return agg_samples

        
    def plot_trace(self, plot_kws=None):
        axes = _plot_trace(self.az_data, plot_kws)
        return axes
    
    def plot_ess(self, plot_kws=None):
        axes = _plot_ess(self.az_data, plot_kws)
        return axes
    
    def plot_rank(self, plot_kws=None):
        axes = _plot_rank(self.az_data, plot_kws)
        return axes
    

    


        
        
                
                
                
                
                        
                        
                        
                        
                
                


    




    
    
    
    
    
    


