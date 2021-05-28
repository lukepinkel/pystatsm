# -*- coding: utf-8 -*-
"""
Created on Wed May 26 06:27:13 2021

@author: lukepinkel
"""
import re
import numpy as np
import scipy as sp
import pandas as pd
from ..pylmm.model_matrices import construct_model_matrices
from ..utilities.random_corr import exact_rmvnorm
from ..utilities.linalg_operations import invech, vech
from ..utilities.numerical_derivs import so_gc_cd

def invech_chol(lvec):
    p = int(0.5 * ((8*len(lvec) + 1)**0.5 - 1))
    L = np.zeros((p, p))
    a, b = np.triu_indices(p)
    L[(b, a)] = lvec
    return L

def transform_theta(theta, dims, indices):
    for key in dims.keys():
        G = invech(theta[indices['theta'][key]])
        L = np.linalg.cholesky(G)
        theta[indices['theta'][key]] = vech(L)
    return theta
        
    
def inverse_transform_theta(theta, dims, indices):
    for key in dims.keys():
        L = invech_chol(theta[indices['theta'][key]])
        G = L.dot(L.T)
        theta[indices['theta'][key]] = vech(G)
    return theta
        
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

class MixedModelSim:
    
    def __init__(self, formula, model_dict, rng=None, group_dict={}):
        df, re_groupings, cont_vars = parse_vars(formula, model_dict)
        rng = np.random.default_rng() if rng is None else rng
        ginfo = model_dict['ginfo']
        for x in re_groupings:
            if x not in group_dict.keys():
                n_grp, n_per =  ginfo[x]['n_grp'],  ginfo[x]['n_per']
                df[x] = np.kron(np.arange(n_grp), np.ones(n_per))
            else:
                df[x] = group_dict[x]
        n_obs = model_dict['n_obs']
        x_mean, x_cov = model_dict['mu'], np.atleast_2d(model_dict['vcov'])
        xvals = exact_rmvnorm(x_cov, n_obs, mu=x_mean)
        df[list(cont_vars)] = xvals
        X, Z, y, dims = construct_model_matrices(formula, data=df)
        
        self.rng = rng
        self.formula, self.model_dict, self.ginfo = formula, model_dict, ginfo
        self.df, self.re_groupings, self.cont_vars = df, re_groupings, cont_vars
        self.X, self.Z, self.dims = X, Z, dims
        self.eta_fe, self.n_obs = X.dot(model_dict['beta']),  n_obs
    
    def simulate_ranefs(self, exact_ranefs=True):
        U = []
        for x in self.re_groupings:
            Gi, n_grp = self.model_dict['gcov'][x], self.model_dict['ginfo'][x]['n_grp']
            u_mean = np.zeros(len(Gi))
            if exact_ranefs:
                Ui = exact_rmvnorm(np.atleast_2d(Gi), n_grp, mu=u_mean).flatten()
            else:
                Ui = self.rng.multivariate_normal(mean=u_mean, cov=Gi, size=n_grp).flatten()
            U.append(Ui)
        u = np.concatenate(U)
        return u
    
    def simulate_linpred(self, exact_ranefs=True):
        u = self.simulate_ranefs(exact_ranefs=exact_ranefs)
        eta = self.eta_fe + self.Z.dot(u)
        return eta
    
    def simulate_response(self, rsq=None, resid_scale=None, exact_ranefs=True, 
                          exact_resids=True):
        eta = self.simulate_linpred(exact_ranefs=exact_ranefs)
        
        if resid_scale is None:
            rsq = 0.64 if rsq is None else rsq
            s = np.sqrt((1-rsq)/rsq*eta.var())
        else:
            s = resid_scale
        if exact_resids:
            resids = self.rng.normal(0, 1, size=self.n_obs)
            resids = (resids - resids.mean()) / resids.std() * s
            y = eta + resids
        else:
            y = self.rng.normal(eta, scale=s)
        return y
    
    def update_model(self, model, rsq=None, resid_scale=None, exact_ranefs=True,
                     exact_resids=True):
        y = self.simulate_response(rsq, resid_scale, exact_ranefs, exact_resids).reshape(-1, 1)
        model.y = y
        model.Xty, model.Zty, model.yty = model.X.T.dot(y), model.Z.T.dot(y), y.T.dot(y)
        model.m = sp.sparse.csc_matrix(np.vstack([model.Xty, model.Zty]))
        model.M = sp.sparse.bmat([[model.C, model.m], [model.m.T, model.yty]])
        return model
    
    def sim_fit(self, model, theta_init, opt_kws={}):
        opt = sp.optimize.minimize(model.loglike_c, theta_init.copy(), 
                                   jac=model.gradient_chol, bounds=model.bounds, 
                                   method='l-bfgs-b', **opt_kws)
        theta_chol = opt.x
        theta = inverse_transform_theta(theta_chol.copy(), model.dims, model.indices)
        beta, XtWX_inv, _, _, _, _ = model._compute_effects(theta)
        se_beta = np.sqrt(np.diag(XtWX_inv))
        Hinv_theta = np.linalg.pinv(so_gc_cd(model.gradient, theta)/2.0)
        se_theta = np.sqrt(np.diag(Hinv_theta))
        params = np.concatenate((beta, theta))
        se = np.concatenate((se_beta, se_theta))
        return params, se, opt
        
