#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 21:34:06 2020

@author: lukepinkel
"""
import re
import numpy as np
import scipy as sp
import scipy.stats
from ..pylmm.lmm_chol import LMEC
from ..utilities.linalg_operations import invech, vech
from ..utilities.random_corr import vine_corr
from sksparse.cholmod import cholesky
import pandas as pd # analysis:ignore
from ..pylmm.model_matrices import construct_model_matrices


def replace_duplicate_operators(match):
    return match.group()[-1:]


def parse_vars(formula, model_dict):
    n_obs = model_dict['n_obs']
    matches = re.findall("\([^)]+[|][^)]+\)", formula)
    groups = [re.search("\(([^)]+)\|([^)]+)\)", x).groups() for x in matches]
    frm = formula
    for x in matches:
        frm = frm.replace(x, "")
    fe_form = re.sub("(\+|\-)(\+|\-)+", replace_duplicate_operators, frm)
    re_form, re_groupings = list(zip(*groups))
    re_form, re_groupings = set(re_form), set(re_groupings)

    yvars, fe_form = re.split("[~]", fe_form)
    fe_form = re.sub("\+$", "", fe_form)
    
    fixed_vars = re.split("[(\+|\-)]", fe_form)
    randm_vars = [re.split("[(\+|\-)]", x) for x in re_form]
    randm_vars = set([x for y in randm_vars for x in y])
    vars_ = sorted(randm_vars.union(fixed_vars).union(re_groupings))
    cont_vars = sorted(randm_vars.union(fixed_vars))
    if str(1) in cont_vars:
        cont_vars.remove(str(1))
        
    vars_ = sorted(set(cont_vars).union(re_groupings))
    if str(1) in vars_:
        vars_.remove(str(1))
        
    df = pd.DataFrame(np.zeros((n_obs, len(vars_))), columns=vars_)
    df[yvars] = 0
    return df, re_groupings, cont_vars

def _generate_model(df, formula, re_groupings, cont_vars, model_dict, r=0.5):
    beta =  model_dict['beta']
    gcov = model_dict['gcov']
    ginfo = model_dict['ginfo']
    mu = model_dict['mu']
    n_obs = model_dict['n_obs']
    vcov = model_dict['vcov']

    for x in re_groupings:
        n_grp, n_per = ginfo[x]['n_grp'], ginfo[x]['n_per']
        df[x] = np.kron(np.arange(n_grp), np.ones(n_per))


    df[list(cont_vars)] = sp.stats.multivariate_normal(mu, vcov).rvs(n_obs)
    X, Z, y, dims = construct_model_matrices(formula, data=df)
    U = []
    for x in re_groupings:
        n_grp = ginfo[x]['n_grp'],
        Gi = gcov[x]
        Ui = sp.stats.multivariate_normal(np.zeros(Gi.shape[1]), 
                                         Gi).rvs(n_grp).flatten()
        U.append(Ui)
    u = np.concatenate(U)
    eta = X.dot(beta)+Z.dot(u)
    eta_var = eta.var()
    rsq = r**2
    df['y'] = sp.stats.norm(eta, np.sqrt((1-rsq)/rsq*eta_var)).rvs()
    return df, u, eta
       
def generate_data(formula, model_dict, r=0.5):
    df, re_groupings, cont_vars = parse_vars(formula, model_dict)
    df, u, eta = _generate_model(df, formula, re_groupings, cont_vars,  model_dict, r)
    return df, formula, u, eta
    
      

formula = "y~1+x1+x2+x3+(1+x4|id1)"
model_dict = {}
model_dict['gcov'] = {'id1':invech(np.array([4., -1.0, 2.]))}
model_dict['ginfo'] = {'id1':dict(n_grp=250, n_per=20)} 
model_dict['mu'] = np.zeros(4)
model_dict['vcov'] = vine_corr(4, 20)
model_dict['beta'] = np.array([1, -1, 2, -2])
model_dict['n_obs'] = 5000
data, formula, u, eta = generate_data(formula, model_dict, r=0.9**0.5)

u_true = u.copy()

model = LMEC(formula, data)

model._fit(opt_kws=dict(verbose=3), use_hess=True)
model._post_fit()

theta = LMEC(formula, data).theta.copy()


XZ = model.XZ
XZtXZ = XZ.T.dot(XZ)
u_indices = {}
start=0
for key, val in model.dims.items():
    q = val['n_groups']*val['n_vars']
    u_indices[key] = np.arange(start, start+q)
    start+=q

t_init = LMEC(formula, data).theta.copy()
V_priors = dict(id1=np.eye(2))
beta_draws, theta_draws, ll_evals = [], [], []
start_multiplier = [0.1, 0.25, 0.5, 1.0, 1.5, 2.0 ]
for chain, start_multiplier in enumerate(start_multiplier):
    theta = start_multiplier*t_init
    beta_samples, theta_samples, ll_eval = [], [], []
    for i in range(1500):
        G = model.update_gmat(theta).copy()
        Ginv = model.update_gmat(theta, inverse=True).copy()

        chol_fac = cholesky(G, ordering_method='natural').L()
        a_star = chol_fac.dot(sp.stats.norm(np.zeros(G.shape[0])).rvs())
    
        Za = model.Z.dot(a_star)
        z_star = sp.stats.norm(Za, np.sqrt(theta[-1])).rvs()
        r = model.y[:, 0] - z_star
        
        C = sp.sparse.csc_matrix(XZtXZ.copy())
        C[-Ginv.shape[0]:, -Ginv.shape[0]:] += Ginv
        Ftz = model.XZ.T.dot(r)
        offset = np.zeros(model.Z.shape[1]+model.X.shape[1])
        offset[-model.Z.shape[1]:] = a_star
        loc_sample = offset+sp.sparse.linalg.spsolve(C, Ftz)
        u = loc_sample[-model.Z.shape[1]:]
        
        
        resid = model.y[:, 0] - XZ.dot(loc_sample)
        sse = resid.T.dot(resid)
        
        nu = model.X.shape[0] - 2
        sc = sse/nu
        
        ss = sp.stats.invgamma(nu/2, scale=(nu*sc/2)).rvs()
        
        beta_samples.append(loc_sample[:model.X.shape[1]])
        for key in model.levels:
            q = model.dims[key]['n_vars']*model.dims[key]['n_groups']
            u_i = u[u_indices[key]]
    
            U_i = u_i.reshape(-1, model.dims[key]['n_vars'], order='C')
            Sg_i =  U_i.T.dot(U_i)
            k = model.dims[key]['n_vars']
            nu = q-(k+1)
            Gs = sp.stats.invwishart(nu, Sg_i).rvs()
            theta[model.indices['theta'][key]] = vech(Gs)
            
        theta[-1] = ss 
        theta_scaled = theta.copy()
        theta_scaled[:-1] = theta_scaled[:-1]*theta_scaled[-1]
        theta_samples.append(theta_scaled.copy())
        ll_eval.append(model.loglike(theta_scaled))
        print(chain, i, theta, ll_eval[i])
            
    beta_samples = np.vstack(beta_samples)
    theta_samples = np.vstack(theta_samples)   

    beta_draws.append(beta_samples[np.newaxis])
    theta_draws.append(theta_samples[np.newaxis])
    ll_evals.append(np.array(ll_eval)[np.newaxis])
    
    
import arviz as az
import seaborn as sns
import matplotlib.pyplot as plt
betas = np.concatenate(beta_draws, axis=0)[:, 200:]
thetas = np.concatenate(theta_draws, axis=0)[:, 200:]
param_samples = np.concatenate((betas, thetas), axis=2)
summary = az.summary(param_samples)

az.plot_trace(thetas)


fig, ax = plt.subplots(ncols=4)
for i in range(4):
    sns.distplot(betas.reshape(-1, 4)[:, i], ax=ax[i])
mng = plt.get_current_fig_manager()
mng.window.showMaximized()
plt.subplots_adjust(left=0.01, right=0.99, top=0.7, bottom=0.3)


fig, ax = plt.subplots(ncols=4)
for i in range(4):
    sns.distplot(thetas.reshape(-1, 4)[:, i], ax=ax[i])
mng = plt.get_current_fig_manager()
mng.window.showMaximized()
plt.subplots_adjust(left=0.01, right=0.99, top=0.7, bottom=0.3)


