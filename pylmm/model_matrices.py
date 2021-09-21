#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 16 21:36:15 2020

@author: lukepinkel
"""
import re
import patsy
import numpy as np
import scipy as sp # analysis:ignore
import scipy.stats # analysis:ignore
import scipy.sparse as sps
from ..utilities.linalg_operations import (_check_np, invech, vech, dummy,
                                           invech_chol)
from ..utilities.special_mats import lmat


def replace_duplicate_operators(match):
    return match.group()[-1:]

def parse_random_effects(formula):
    """
    Parameters
    ----------
    formula : string
        Model formula.
    Returns
    -------
    fe_form : string
        Fixed effects formula.
    groups : list
        List of tuples of strings.  Each tuple contains the formula for the
        random effect term as the first element and the group/level as the 
        second element
    """
    matches = re.findall("\([^)]+[|][^)]+\)", formula)
    groups = [re.search("\(([^)]+)\|([^)]+)\)", x).groups() for x in matches]
    frm = formula
    for x in matches:
        frm = frm.replace(x, "")
    fe_form = re.sub("(\+|\-)(\+|\-)+", replace_duplicate_operators, frm)
    return fe_form, groups

def construct_random_effects(groups, data, n_vars):
    re_vars, re_groupings = list(zip(*groups))
    re_vars, re_groupings = set(re_vars), set(re_groupings)
    Zdict = dict(zip(re_vars, [_check_np(patsy.dmatrix(x, data=data, return_type='dataframe')) for x in re_vars]))
    Jdict = dict(zip(re_groupings, [dummy(data[x]) for x in re_groupings]))
    dim_dict = {}
    Z = []
    for x, y in groups:
        Ji, Xi = Jdict[y], Zdict[x]
        dim_dict[y] = {'n_groups':Ji.shape[1], 'n_vars':Xi.shape[1]}
        Zi = sp.linalg.khatri_rao(Ji.T, Xi.T).T
        Z.append(Zi)
    Z = np.concatenate(Z, axis=1)
    return Z, dim_dict

def construct_model_matrices(formula, data, return_fe=False):
    fe_form, groups = parse_random_effects(formula)
    yvars, fe_form = re.split("[~]", fe_form)
    fe_form = re.sub("\+$", "", fe_form)
    yvars = re.split(",", re.sub("\(|\)", "", yvars))
    yvars = [x.strip() for x in yvars]
    n_vars = len(yvars)
    Z, dim_dict = construct_random_effects(groups, data, n_vars)
    X = patsy.dmatrix(fe_form, data=data, return_type='dataframe')
    fe_vars = X.columns
    y = data[yvars]
    X, y = _check_np(X), _check_np(y)
    if return_fe:
        return X, Z, y, dim_dict, list(dim_dict.keys()), fe_vars
    else:
        return X, Z, y, dim_dict, list(dim_dict.keys())

def handle_missing(formula, data):
    fe_form, groups = parse_random_effects(formula)
    yvars, fe_form = re.split("[~]", fe_form)
    fe_form = re.sub("\+$", "", fe_form)
    g_vars = [x for y in groups for x in y]
    g_vars = [re.split("[+\-\*:]", x) for x in g_vars]
    re_vars = [x for y in g_vars for x in y]
    fe_vars = re.split("[+\-\*:]", fe_form)
    if type(yvars) is str:
        yvars = [yvars]
    vars_ = set(re_vars + fe_vars+yvars)
    cols = set(data.columns)
    var_subset = vars_.intersection(cols)
    valid_ind = ~data[var_subset].isnull().any(axis=1)
    return valid_ind


def make_theta(dims):
    theta, indices, index_start = [], {}, 0
    dims = dims.copy()
    dims['resid'] = dict(n_groups=0, n_vars=1)
    for key, value in dims.items():
        n_vars = value['n_vars']
        n_params = int(n_vars * (n_vars+1) //2)
        indices[key] = np.arange(index_start, index_start+n_params)
        theta.append(vech(np.eye(n_vars)))
        index_start += n_params
    theta = np.concatenate(theta)
    return theta, indices

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

def lndet_gmat(theta, dims, indices):
    lnd = 0.0
    for key, value in dims.items():
        dims_i = dims[key]
        ng = dims_i['n_groups']
        Sigma_i = invech(theta[indices['theta'][key]])
        lnd += ng*np.linalg.slogdet(Sigma_i)[1]
    return lnd

def lndet_gmat_chol(theta, dims, indices):
    lnd = 0.0
    for key, value in dims.items():
        dims_i = dims[key]
        ng = dims_i['n_groups']
        theta_i = theta[indices['theta'][key]]
        L_i = invech_chol(theta_i)
        Sigma_i = L_i.dot(L_i.T)            
        lnd += ng*np.linalg.slogdet(Sigma_i)[1]
    return lnd

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
        
def get_d2_chol(dim_i):
    p = dim_i['n_vars']
    Lp = lmat(p).A
    T = np.zeros((p, p))
    H = []
    Ip = np.eye(p)
    for j, i in list(zip(*np.triu_indices(p))):
        T[i, j] = 1
        Hij = (Lp.dot(np.kron(Ip, T+T.T)).dot(Lp.T))[np.newaxis]
        H.append(Hij)
        T[i, j] = 0
    H = np.concatenate(H, axis=0)
    return H
        
         
def get_jacmats(Zs, dims, indices, g_indices, theta):
    start = 0
    jac_mats = {}
    jac_inds = {}
    for key, value in dims.items():
        nv, ng =  value['n_vars'], value['n_groups']
        jac_mats[key] = []
        theta_i = theta[indices[key]]
        jac_inds[key] = np.arange(start, start+ng*nv)
        nv2, nvng = nv*nv, nv*ng
        row = np.repeat(np.arange(nvng), nv)
        col = np.repeat(np.arange(ng)*nv, nv2)
        col = col + np.tile(np.arange(nv), nvng)
        for i in range(len(theta_i)):
            dtheta_i = np.zeros_like(theta_i)
            dtheta_i[i] = 1.0
            dtheta_i = invech(dtheta_i).reshape(-1, order='F')
            data = np.tile(dtheta_i, ng)
            dGi = sps.csc_matrix((data, (row, col)))
            jac_mats[key].append(dGi)
        start+=ng*nv
    jac_mats['resid'] = [sps.eye(Zs.shape[0])]
    return jac_mats, jac_inds

    
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
    
    
def vcrepara_grad(tau, gradient, reparam):
    theta = reparam.inverse_transform(tau)
    J = reparam.jacobian(theta)
    g = gradient(theta)
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
    

        