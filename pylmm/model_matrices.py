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

from ..utilities.linalg_operations import (_check_np, khatri_rao, invech, vech, dummy,
                                           _check_shape)
from ..utilities.special_mats import (kronvec_mat, dmat)



def replace_duplicate_operators(match):
    return match.group()[-1:]

def parse_random_effects(formula):
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
        Zi = khatri_rao(Ji.T, Xi.T).T
        Z.append(Zi)
    Z = np.concatenate(Z, axis=1)
    return Z, dim_dict

def construct_model_matrices(formula, data):
    fe_form, groups = parse_random_effects(formula)
    yvars, fe_form = re.split("[~]", fe_form)
    fe_form = re.sub("\+$", "", fe_form)
    yvars = re.split(",", re.sub("\(|\)", "", yvars))
    yvars = [x.strip() for x in yvars]
    n_vars = len(yvars)
    Z, dim_dict = construct_random_effects(groups, data, n_vars)
    X = patsy.dmatrix(fe_form, data=data, return_type='dataframe')
    y = data[yvars]
    return X, Z, y, dim_dict

def vech2vec(vh):
    A = invech(vh)
    v = A.reshape(-1, order='F')
    return v
    

def make_theta(dims):
    theta, indices, index_start = [], {}, 0
    for key, value in dims.items():
        n_vars = value['n_vars']
        n_params = int(n_vars * (n_vars+1) //2)
        indices[key] = np.arange(index_start, index_start+n_params)
        theta.append(vech(np.eye(n_vars)))
        index_start += n_params
    theta = np.concatenate(theta)
    return theta, indices

def create_gmats(theta, indices, dims, inverse=False):
    Gmats, g_indices, start = {}, {}, 0
    for key, value in dims.items():
        if key!='error':
            dims_i = dims[key]
            ng, nv = dims_i['n_groups'],  dims_i['n_vars']
            nv2, nvng = nv*nv, nv*ng
            theta_i = theta[indices[key]]
            if inverse:
                theta_i = np.linalg.inv(invech(theta_i)).reshape(-1, order='F')
            else:
                theta_i = vech2vec(theta_i)
            row = np.repeat(np.arange(nvng), nv)
            col = np.repeat(np.arange(ng)*nv, nv2)
            col = col + np.tile(np.arange(nv), nvng)
            data = np.tile(theta_i, ng)
            Gmats[key] = sps.csc_matrix((data, (row, col)))
            g_indices[key] = np.arange(start, start+len(data))
            start += len(data)
    return Gmats, g_indices
                
def update_gmat(theta, G, dims, indices, g_indices, inverse=False):
    for key in g_indices.keys():
        ng = dims[key]['n_groups']
        theta_i = theta[indices[key]]
        if inverse:
            theta_i = np.linalg.inv(invech(theta_i)).reshape(-1, order='F')
        else:
            theta_i = vech2vec(theta_i)
        G.data[g_indices[key]] = np.tile(theta_i, ng)
    return G
        
  
        
def sparse_woodbury_inversion(Umat, Vmat=None, C=None, Cinv=None, A=None, Ainv=None):
    if Ainv is None:
        Ainv = sps.linalg.inv(A)
    if Cinv is None:
        Cinv = sps.linalg.inv(C)
    if Vmat is None:
        Vmat = Umat.T
    T = Ainv.dot(Umat)
    H = sps.linalg.inv(Cinv + Vmat.dot(T))
    W = Ainv - T.dot(H).dot(Vmat).dot(Ainv)
    return W

def lndet_gmat(theta, dims, indices):
    lnd = 0.0
    for key, value in dims.items():
        if key!='error':
            dims_i = dims[key]
            ng = dims_i['n_groups']
            Sigma_i = invech(theta[indices[key]])
            lnd += ng*np.linalg.slogdet(Sigma_i)[1]
    return lnd
    
        
def get_derivmats(Zs, dims):
    start = 0
    deriv_mats = {}
    for key, value in dims.items():
        nv, ng =  value['n_vars'], value['n_groups']
        Sv_shape = nv, nv
        Av_shape = ng, ng
        Kv = kronvec_mat(Av_shape, Sv_shape)
        Ip = sps.csc_matrix(sps.eye(np.product(Sv_shape)))
        vecAv = sps.csc_matrix(sps.eye(ng)).reshape((-1, 1), order='F')
        D = sps.csc_matrix(Kv.dot(sps.kron(vecAv, Ip)))
        if key != 'error':
            Zi = Zs[:, start:start+ng*nv]
            ZoZ = sps.kron(Zi, Zi)
            D = sps.csc_matrix(ZoZ.dot(D))
            start+=ng*nv
        sqrd = int(np.sqrt(D.shape[1]))
        tmp = sps.csc_matrix(dmat(sqrd))
        deriv_mats[key] = D.dot(tmp)
    return deriv_mats
  
def get_jacmats2(Zs, dims, indices, g_indices, theta):
    start = 0
    jac_mats = {}
    for key, value in dims.items():
        nv, ng =  value['n_vars'], value['n_groups']
        jac_mats[key] = []
        if key != 'error':
            Zi = Zs[:, start:start+ng*nv]
            theta_i = theta[indices[key]]
            nv2, nvng = nv*nv, nv*ng
            row = np.repeat(np.arange(nvng), nv)
            col = np.repeat(np.arange(ng)*nv, nv2)
            col = col + np.tile(np.arange(nv), nvng)
            for i in range(len(theta_i)):
                dtheta_i = np.zeros_like(theta_i)
                dtheta_i[i] = 1.0
                dtheta_i = vech2vec(dtheta_i)
                data = np.tile(dtheta_i, ng)
                dGi = sps.csc_matrix((data, (row, col)))
                dVi = Zi.dot(dGi).dot(Zi.T)
                jac_mats[key].append(dVi)
            start+=ng*nv
        else:
            jac_mats[key].append(sps.eye(Zs.shape[0]))
    return jac_mats

def jac2deriv(jac_mats):
    deriv_mats = {}
    for key, value in jac_mats.items():
        deriv_mats[key] = []
        for dVi in value:
            deriv_mats[key].append(dVi.reshape((-1, 1), order='F'))
        deriv_mats[key] = sps.hstack(deriv_mats[key])
    return deriv_mats
        
    

def lsq_estimate(dims, theta, indices, X, XZ, y):
    b, _, _, _, _, _, _, _, _, _ = (sps.linalg.lsqr(XZ, y))
    u_hat = b[X.shape[1]:]
    i = 0
    for key, value in dims.items():
        if key!='error':
            ng, nv = value['n_groups'], value['n_vars']
            n_u = ng * nv
            ui = u_hat[i:i+n_u].reshape(ng, nv)
            theta[indices[key]] = vech(np.atleast_2d(np.cov(ui, rowvar=False)))
            i+=n_u
    r =  _check_shape(y) - _check_shape(XZ.dot(b))
    theta[-1] = np.dot(r.T, r) / len(r)
    return theta
        
def get_jacmats(deriv_mats):
    jac_mats = {}
    for key, value in deriv_mats.items():
        jmi = []
        for i in range(value.shape[1]):
            n2 = value.shape[0]
            n = int(np.sqrt(n2))
            jmi.append(value[:, i].reshape(n, n))
        jac_mats[key] = jmi
    return jac_mats
            
            
        