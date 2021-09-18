# -*- coding: utf-8 -*-
"""
Created on Wed May 26 06:27:13 2021

@author: lukepinkel
"""
import re
import numpy as np
import scipy as sp
import pandas as pd
import scipy.sparse as sps
from ..pylmm.model_matrices import construct_model_matrices
from ..utilities.random_corr import exact_rmvnorm, multivariate_t, _exact_cov
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
    theta[-1] = np.log(theta[-1])
    return theta
        
    
def inverse_transform_theta(theta, dims, indices):
    for key in dims.keys():
        L = invech_chol(theta[indices['theta'][key]])
        G = L.dot(L.T)
        theta[indices['theta'][key]] = vech(G)
    theta[-1] = np.exp(theta[-1])
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


def _make_sim_theta(model_dict):
    theta_init, theta_true, indices, index_start =[],  [], {}, 0
    gcov = model_dict["gcov"].copy()
    gcov['resid'] = np.eye(1)
    for key, Gi in gcov.items():
        n_vars = Gi.shape[0]
        n_params = int(n_vars * (n_vars+1) //2)
        indices[key] = np.arange(index_start, index_start+n_params)
        theta_true.append(vech(Gi))
        theta_init.append(vech(np.eye(n_vars)))
        index_start += n_params
    theta_true = np.concatenate(theta_true)
    theta_init = np.concatenate(theta_init)
    return theta_true, theta_init, indices

def make_gcov(theta, indices, dims, inverse=False):
    Gmats, g_indices, start = {}, {}, 0
    for key, value in dims.items():
        dims_i = dims[key]
        ng, nv = dims_i['n_groups'],  dims_i['n_vars']
        nv2, nvng = nv*nv, nv*ng
        theta_i = theta[indices['theta'][key]]
        if inverse:
            theta_i = np.linalg.inv(invech(theta_i)).reshape(-1, order='F')
        else:
            theta_i = invech(theta_i).reshape(-1, order='F')
        row = np.repeat(np.arange(nvng), nv)
        col = np.repeat(np.arange(ng)*nv, nv2)
        col = col + np.tile(np.arange(nv), nvng)
        data = np.tile(theta_i, ng)
        Gmats[key] = sps.csc_matrix((data, (row, col)))
        g_indices[key] = np.arange(start, start+len(data))
        start += len(data)
    G = sps.block_diag(list(Gmats.values())).tocsc()
    return G, g_indices


def get_var_comps(Xb, Z, G):
    re_var = np.mean(np.einsum("ij,jj,ij->i", Z, G.A, Z))
    fe_var = np.dot(Xb.T, Xb) / Xb.shape[0]
    return fe_var, re_var


class MixedModelSim:
    
    def __init__(self, formula, model_dict, rng=None, group_dict={}, var_ratios=None,
                 ranef_dist=None, resid_dist=None):
        rng = np.random.default_rng() if rng is None else rng
        ranef_dist =  rng.multivariate_normal if ranef_dist is None else ranef_dist
        resid_dist =  rng.normal if resid_dist is None else resid_dist
        
        df, re_groupings, cont_vars = parse_vars(formula, model_dict)
        
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
        self.rng, self.ranef_dist, self.resid_dist = rng, ranef_dist, resid_dist
        self.formula, self.model_dict, self.ginfo = formula, model_dict, ginfo
        self.df, self.re_groupings, self.cont_vars = df, re_groupings, cont_vars
        self.X, self.Z, self.dims = X, Z, dims
        self.eta_fe, self.n_obs = X.dot(model_dict['beta']),  n_obs
        self.indices = {}
        self.theta_true, self.theta_init, self.indices["theta"] = _make_sim_theta(model_dict)
        self.G, self.indices["g"] = make_gcov(self.theta_true, self.indices, self.dims)
        self.beta = model_dict['beta']
        if var_ratios is not None:
            r_fe, r_re = var_ratios[0], var_ratios[1]
            v_fe, v_re = get_var_comps(self.eta_fe, self.Z, self.G)
            c = (v_fe / v_re) * (r_re / r_fe)
            for key in self.dims.keys():
                ng = self.dims[key]['n_groups']
                t = self.theta_true[self.indices['theta'][key]] * c
                Gi = invech(t)
                self.model_dict["gcov"][key] = Gi
                theta_i = Gi.reshape(-1, order='F')
                self.G.data[self.indices['g'][key]] = np.tile(theta_i, ng)
                self.theta_true[self.indices['theta'][key]] = t
            self.v_fe, self.v_re = get_var_comps(self.eta_fe, self.Z, self.G)
            self.c = c
            rt = r_re + r_fe
            self.v_rs = (1.0 - rt) / rt * (self.v_re+self.v_fe)
            self.theta_true[-1] = self.v_rs
        else:
            self.v_re, self.v_fe, self.v_rs, self.c = None, None, None, None
        self.var_ratios = var_ratios
    
    def simulate_ranefs(self, exact_ranefs=True, ranef_dist=None, ranef_kws={}):
        U = []
        dist = self.ranef_dist if ranef_dist is None else ranef_dist
        for x in self.re_groupings:
            Gi, n_grp = self.model_dict['gcov'][x], self.model_dict['ginfo'][x]['n_grp']
            u_mean = np.zeros(len(Gi))
            Ui = dist(mean=u_mean, cov=Gi, size=n_grp, **ranef_kws)
            if exact_ranefs:
                Ui = _exact_cov(Ui, mean=u_mean, cov=Gi)
            u_i = Ui.flatten()
            U.append(u_i)
        u = np.concatenate(U)
        return u
    
    def simulate_linpred(self, exact_ranefs=True, ranef_dist=None, ranef_kws={}):
        u = self.simulate_ranefs(exact_ranefs=exact_ranefs, ranef_dist=ranef_dist,
                                 ranef_kws=ranef_kws)
        eta = self.eta_fe + self.Z.dot(u)
        return eta
    
    def simulate_response(self, rsq=None, resid_scale=None, exact_ranefs=True, 
                          exact_resids=True, ranef_dist=None, resid_dist=None, 
                          ranef_kws={}, resid_kws={}):
        eta = self.simulate_linpred(exact_ranefs=exact_ranefs, ranef_dist=ranef_dist,
                                    ranef_kws=ranef_kws)
        dist = self.resid_dist if resid_dist is None else resid_dist
        if resid_scale is None and self.v_rs is None:
            rsq = 0.64 if rsq is None else rsq
            s = np.sqrt((1-rsq)/rsq*eta.var())
        elif resid_scale is None and self.v_rs is not None:
            s = np.sqrt(self.v_rs)
        else:
            s = resid_scale
            
        if exact_resids:
            resids = dist(loc=0, scale=1, size=self.n_obs, **resid_kws)
            resids = (resids - resids.mean()) / resids.std() * s
            y = eta + resids
        else:
            y = dist(loc=eta, scale=s, size=self.n_obs, **resid_kws)
        return y
    
    def update_model(self, model, y):
        model.y = y
        model.Xty, model.Zty, model.yty = model.X.T.dot(y), model.Z.T.dot(y), y.T.dot(y)
        model.m = sp.sparse.csc_matrix(np.vstack([model.Xty, model.Zty]))
        model.M = sp.sparse.bmat([[model.C, model.m], [model.m.T, model.yty]])
        return model
    
    def sim_fit(self, model, theta_init, method='l-bfgs-b', bounds=None,
                opt_kws={}):
        if bounds is None:
            bounds = model.bounds_2
        opt = sp.optimize.minimize(model.loglike_c, theta_init.copy(),
                                   jac=model.gradient_chol, bounds=bounds,
                                   method=method, **opt_kws)
        theta_chol = opt.x
        theta = inverse_transform_theta(theta_chol.copy(), model.dims, model.indices)
        beta, XtWX_inv, u = model._compute_effects(theta)
        se_beta = np.sqrt(np.diag(XtWX_inv))
        Hinv_theta = np.linalg.pinv(so_gc_cd(model.gradient, theta)/2.0)
        se_theta = np.sqrt(np.diag(Hinv_theta))
        params = np.concatenate((beta, theta))
        se = np.concatenate((se_beta, se_theta))
        return params, se, opt
        
