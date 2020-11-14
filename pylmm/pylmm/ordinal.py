# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 03:21:18 2020

@author: lukepinkel
"""

import tqdm
import arviz as az
import numpy as np
import scipy as sp
import scipy.stats
#import pandas as pd # analysis:ignore
from .lmm_chol import LMEC, make_theta
from ..utilities.linalg_operations import vech, _check_shape#, invech
from sksparse.cholmod import cholesky
from ..utilities.trnorm import trnorm, scalar_truncnorm
from ..utilities.wishart import r_invwishart, r_invgamma
#from pystats.pylmm.tests.test_data2 import generate_data
#from pystats.utilities.random_corr import vine_corr

SQRT2 = np.sqrt(2)
def csd(x):
    x = (x - x.mean(axis=0)) / x.std(axis=0)
    return x

def to_arviz_dict(samples, var_dict, burnin=0):
    az_dict = {}
    for key, val in var_dict.items():
        az_dict[key] = samples[:, burnin:, val]
    return az_dict    
   
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

def norm_cdf(x, mean=0.0, sd=1.0):
    z = (x - mean) / sd
    p = (sp.special.erf(z/SQRT2) + 1.0) / 2.0
    return p

class OrdinalMCMC(LMEC):
    
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
        self.jv = np.ones(self.n_ob)
        self.n_cats = np.unique(self.y).shape[0]
        self.n_thresh = self.n_cats - 1
        self.n_tau = self.n_cats + 1
        self.y_cat = np.zeros((self.n_ob, self.n_cats))
        self.y_ix = {}
        for i in range(self.n_cats):
            self.y_ix[i] = self.y==i
            self.y_cat[self.y==i, i] = 1
    
    def sample_location(self, theta, x1, x2, y):
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
    
    def sample_tau(self, t, pred, v, propC):
        t_prop = t.copy()
        ll = 0.0
        tau = np.pad(t, ((1, 1)), mode='constant', constant_values=[-1e17, 1e17])
        for i in range(1, self.n_thresh):
            t_prop[i] = scalar_truncnorm(t[i], propC, t_prop[i-1], tau[i+2])
        for i in range(1, self.n_thresh-1):
            a = (t[i+1]-t[i])/propC
            b = (t_prop[i-1]-t[i])/propC
            c = (t_prop[i+1]-t_prop[i])/propC
            d = (t[i-1]-t_prop[i])/propC
            ll += np.sum(np.log(norm_cdf(a)-norm_cdf(b)))
            ll -= np.sum(np.log(norm_cdf(c)-norm_cdf(d)))
        
        for i in range(1, self.n_thresh):
            m = v[self.y_ix[i]]
            ll += (np.log(norm_cdf(t_prop[i]-m)-norm_cdf(t_prop[i-1]-m))).sum()
            ll -= (np.log(norm_cdf(t[i]-m)-norm_cdf(t[i-1]-m))).sum()
        m = v[self.y_ix[i+1]]
        ll += np.sum(np.log(1.0 - norm_cdf(t_prop[i]-m)))
        ll -= np.sum(np.log(1.0 - norm_cdf(t[i]-m)))
        if ll>np.log(np.random.uniform(0, 1)):
            t_accept = True
            t = t_prop
        else:
            t_accept = False
            t = t
        return t, t_accept
    
    def sample_lvar(self, theta, t, pred, v):
        tau = np.pad(t, ((1, 1)), mode='constant', constant_values=[-1e17, 1e17])
        mu, sd = np.zeros_like(pred), np.ones_like(pred)
        for i in range(1, self.y_cat.shape[1]+1):
            ix = self.y_ix[i-1]
            j = self.jv[ix]
            mu = pred[ix] - v[ix]
            sd = j
            lb = j*tau[i-1]
            ub = j*tau[i]
            v[ix] = trnorm(mu, sd, lb, ub)
        return v
    
    
    def sample_theta(self, theta, u, z, pred, freeR=True):
        for key in self.levels:
            theta = sample_gcov(theta.copy(), u, self.wsinfo, self.indices,
                                key, self.priors)
        if freeR:
            theta = sample_rcov(theta, z, pred, self.wsinfo, self.priors)
        return theta
    
    
    def sample_gibbs_clm(self, n_samples, chain=0, store_z=False,  freeR=False,
                         propC=0.1, max_iters=None):
        if max_iters is None:
            max_iters = n_samples*100
        param_samples = np.zeros((n_samples, self.n_params+self.n_thresh))
        t_acceptances = np.zeros((n_samples))
        t_accept_prob = np.zeros((n_samples))
        if store_z:
            z_samples = np.zeros((n_samples, self.n_ob))
        else:
            z_samples = None
        x_astr = sp.stats.norm(0.0, 1.0).rvs((n_samples, self.n_ob))
        x_ranf = sp.stats.norm(0.0, 1.0).rvs((n_samples, self.n_re))
        location = self.location.copy()
        pred =  self.W.dot(location)
        z = sp.stats.norm(self.y, self.y.var()).rvs()
        theta = self.t_init.copy()
        t = sp.stats.norm(0, 1).ppf((self.y_cat.sum(axis=0).cumsum()/np.sum(self.y_cat))[:-1])
        t = t - t[0]
        z = np.zeros_like(z).astype(float)
        pbar = tqdm.tqdm(range(n_samples))
        j = 0
        for i in range(n_samples):
            t, t_accept = self.sample_tau(t, pred, z, propC)
            if t_accept:
                j+=1
            z = self.sample_lvar(theta, t, pred, z)
            location = self.sample_location(theta, x_ranf[i], x_astr[i], z)
            pred = self.W.dot(location)
            u = location[-self.n_re:]
            theta  = self.sample_theta(theta, u, z, pred, freeR)
            
            param_samples[i, self.n_fe:-self.n_thresh] = theta.copy()
            param_samples[i, :self.n_fe] = location[:self.n_fe]
            param_samples[i, -self.n_thresh:] = t
            t_acceptances[i] = t_accept
            t_accept_prob[i] = (j+1) /(i+1)
            if store_z:
                z_samples[j] = z.copy()
            if i>1:
                pbar.set_description(f"Chain {chain} Tau Acceptance Prob: {t_accept_prob[i]:.4f}")
            pbar.update(1)
        pbar.close() 
        return param_samples, t_acceptances, z_samples
        
    def fit(self, n_samples=5000, n_chains=8, burnin=1000, vnames=None, sample_kws={},
            method='MH-Gibbs'):
        samples = np.zeros((n_chains, n_samples, self.n_params+np.unique(self.y).shape[0]-1))        
        acceptances = np.zeros((n_chains, n_samples))
        vnames =  {"$\\beta$":np.arange(self.n_fe), 
                   "$\\theta$":np.arange(self.n_fe, self.n_params)}
        vnames['$\\tau$'] = np.arange(self.n_params, 
                                      self.n_params+np.unique(self.y).shape[0]-1)
    
        func = self.sample_gibbs_clm
        for i in range(n_chains):
            samples[i], acceptances[i], z_samples = func(n_samples, chain=i, **sample_kws)
        az_dict = to_arviz_dict(samples,  vnames, burnin=burnin)
        az_data = az.from_dict(az_dict)
        summary = az.summary(az_data, hdi_prob=0.95)
        return samples, az_data, summary, acceptances, z_samples
                      

# formula = "y~x1+x2+x3+x4+(1+x5|id1)"
# model_dict = {}
# #model_dict['gcov'] = {'id1':invech(np.array([1.0]))}
# model_dict['gcov'] = {'id1':invech(np.array([1.0, -0.3, 1.0]))}
# model_dict['ginfo'] = {'id1':dict(n_grp=300, n_per=10)} 
# model_dict['mu'] = np.zeros(5)
# model_dict['vcov'] = vine_corr(5, 100)/10
# model_dict['beta'] = np.array([0.0, 0.2, -0.2, 0.4, -0.4])
# model_dict['n_obs'] = 3000

# df, formula, u, linpred = generate_data(formula, model_dict, r=0.6**0.5)
# #model = LMEC(formula, df)
# #model.fit()
# eta = linpred + csd(np.random.normal(0, np.sqrt(linpred.var()), size=model_dict['n_obs']))

# tau = sp.stats.scoreatpercentile(eta, [10, 20, 50, 75])
# df['y'] = pd.cut(eta, np.pad(tau, ((1, 1)), mode='constant', constant_values=[-np.inf, np.inf])).cat.codes.astype(float)

# model = OrdinalMCMC(formula, df)
# samples, az_data, summary, accept, z_samples = model.fit(20_000, 4, 5000, sample_kws=dict(propC=0.04, store_z=True, freeR=False))
# az.plot_trace(az_data, var_names=["$\\tau$"], coords={'$\\tau$_dim_0':[1, 2, 3]})
# az.plot_trace(az_data, var_names=["$\\tau$"], coords={'$\\tau$_dim_0':[1, 2, 3]})
# az.plot_trace(az_data, var_names=['$\\theta$'], coords={'$\\theta$_dim_0':[0, 1, 2]})
# az.plot_trace(az_data, var_names=['$\\beta$'])

# t_accepted = [[samples[i, accept[i]==1, j][200:] for i in range(4)] for j in range(samples.shape[2]-3, samples.shape[2])]
# min_samples = np.min([[len(x) for x in y] for y in t_accepted])
# t_accepted = [np.vstack([y[:min_samples] for y in x]).T[np.newaxis] for x in t_accepted]
# t_accepted = np.concatenate(t_accepted).T
# az.plot_trace(az.convert_to_dataset(t_accepted))

# z_samples = np.concatenate([x[np.newaxis] for x in z_samples], axis=0)

# az_z_samples = az.convert_to_dataset(z_samples)
# z_summary = az.summary(az_z_samples)

# import matplotlib.pyplot as plt
# import seaborn as sns
# prediction_df = pd.DataFrame(np.vstack((z_summary['mean'].values, eta, linpred.values, df['y'].values)).T,
#                               columns=['z', 'eta', 'mu', 'yobs'])
# sns.regplot(y=z_summary['mean'].values, x=linpred.values)
# sns.regplot(y=z_summary['mean'].values[df['y']==4], x=linpred.values[df['y']==4])

# sns.lmplot(x='z', y='eta', data=prediction_df)
# sns.lmplot(x='mu', y='z', data=prediction_df)
# sns.lmplot(x='mu', y='z', hue='yobs', data=prediction_df)

# fig, ax = plt.subplots(ncols=5)
# for i in range(5):
#     sns.regplot(x='mu', y='z', data=prediction_df[prediction_df['yobs']==i], ax=ax[i])


# fig, ax = plt.subplots(ncols=5)
# for i in range(5):
#     sns.regplot(x='eta', y='z', data=prediction_df[prediction_df['yobs']==i], ax=ax[i])



