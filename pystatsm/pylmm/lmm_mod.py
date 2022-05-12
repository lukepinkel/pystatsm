#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 11 17:59:10 2022

@author: lukepinkel
"""
import re
import patsy
import numpy as np
import scipy as sp
import scipy.sparse as sps
from ..utilities.linalg_operations import invech_chol, invech, vech
from sksparse.cholmod import cholesky


def _dummy_encode(x, categories=None):
    categories = np.unique(x) if categories is None else categories
    n_cols = len(categories)
    rows, cols = [], []
    for i, c in enumerate(categories):
        rows_c, = np.where(x==c)
        cols_c = np.repeat(i, len(rows_c))
        rows.append(rows_c)
        cols.append(cols_c)
    row_inds = np.concatenate(rows)
    col_inds = np.concatenate(cols)
    return row_inds, col_inds, n_cols

def dummy_encode(x, categories=None):
    categories = np.unique(x) if categories is None else categories
    n_rows = x.shape[0]
    row_inds, col_inds, n_cols = _dummy_encode(x, categories)
    data = np.ones(n_rows)
    X = sps.csc_matrix((data, (row_inds, col_inds)), shape=(n_rows, n_cols))
    return X
    

class RandomEffectTerm(object):
    
    def __init__(self, re_form, gr_form, data):
        Xi = patsy.dmatrix(re_form, data=data, return_type='dataframe').values
        j_rows, j_cols, q =  _dummy_encode(data[gr_form])
        j_sort = np.argsort(j_rows)
        n, p = Xi.shape
        z_rows = np.repeat(np.arange(n), p)
        z_cols = np.repeat(j_cols[j_sort] * p, p) + np.tile(np.arange(p), n)
        
        g_cols = np.repeat(np.arange(p * q), p)
        g_rows = np.repeat(np.arange(q)*p, p * p) + np.tile(np.arange(p), p * q)
        ij = g_rows>= g_cols
        l_rows, l_cols = g_rows[ij], g_cols[ij]
        m = int(p * (p + 1) //2)
        d_theta = np.zeros(m)
        G_deriv = []
        L_deriv = []
        for i in range(m):
            d_theta[i] = 1
            dG_theta = invech(d_theta).reshape(-1, order='F')
            dGi = sps.csc_matrix((np.tile(dG_theta, q), (g_rows, g_cols)))
            dLi = sps.csc_matrix((np.tile(d_theta, q), (l_rows, l_cols)))
            G_deriv.append(dGi)
            L_deriv.append(dLi)
            d_theta[i] = 0
            
            
        self.G_deriv = G_deriv
        self.L_deriv = L_deriv
        self.re_form = re_form
        self.gr_form = gr_form
        self.Xi = Xi
        self.j_rows, self.j_cols = j_rows, j_cols
        self.z_rows, self.z_cols = z_rows, z_cols
        self.g_rows, self.g_cols = g_rows, g_cols
        self.l_rows, self.l_cols = l_rows, l_cols
        self.n_group = self.q = q
        self.n_rvars = self.p = p
        self.n_param = m
        self.g_size = p * q

        
   

class RandomEffects(object):
    
    def __init__(self, terms):
        z_offset, g_offset, l_offset, t_offset, cov_offset = 0, 0, 0, 0, 0
        z_data, z_rows, z_cols = [], [], []
        g_data, g_cols, g_rows = [], [], []
        l_data, l_cols, l_rows = [], [], []
        g_inds, l_inds, t_inds = [], [], []
        theta = []
        jac_inds = []


        
        
        for ranef in terms: 
            zi_rows, zi_cols = ranef.z_rows, ranef.z_cols + z_offset
            z_rows.append(zi_rows)
            z_cols.append(zi_cols)
            z_data.append(ranef.Xi.flatten())
            jac_inds.append(np.arange(z_offset, z_offset + ranef.p * ranef.q))
            Gi = np.eye(ranef.n_rvars)
            g_vech = vech(Gi)
            g_vec = Gi.reshape(-1, order='F')
            gi_rows, gi_cols = ranef.g_rows + cov_offset, ranef.g_cols + cov_offset
            g_inds.append(np.arange(g_offset, g_offset + ranef.p * ranef.p * ranef.q))

            g_rows.append(gi_rows)
            g_cols.append(gi_cols)
            g_data.append(np.tile(g_vec, ranef.n_group))
            
            li_rows, li_cols = ranef.l_rows + cov_offset, ranef.l_cols + cov_offset
            l_inds.append(np.arange(l_offset, l_offset + ranef.n_param * ranef.q))
            l_rows.append(li_rows)
            l_cols.append(li_cols)
            l_data.append(np.tile(g_vech, ranef.n_group))
            
            theta.append(g_vech)
            t_inds.append(np.arange(t_offset, t_offset+ranef.n_param))
            z_offset = z_offset + ranef.p * ranef.q
            t_offset = t_offset + ranef.n_param
            g_offset = g_offset + ranef.p * ranef.p * ranef.q
            l_offset = l_offset + ranef.n_param * ranef.q
            cov_offset = cov_offset + ranef.g_size
            


    
        theta = np.concatenate(theta)  
        n_rows = terms[0].Xi.shape[0]
        n_cols = z_offset
        z_rows, z_cols = np.concatenate(z_rows),  np.concatenate(z_cols)
        g_rows, g_cols = np.concatenate(g_rows),  np.concatenate(g_cols)
        l_rows, l_cols = np.concatenate(l_rows),  np.concatenate(l_cols)

        z_data = np.concatenate(z_data)
        g_data = np.concatenate(g_data)
        l_data = np.concatenate(l_data)
        self.z_rows, self.z_cols, self.z_data = z_rows, z_cols, z_data
        self.g_rows, self.g_cols = g_rows, g_cols
        self.l_rows, self.l_cols = l_rows, l_cols
        self.Z = sps.csc_matrix((z_data, (z_rows, z_cols)), shape=(n_rows, n_cols))
        self.G = sps.csc_matrix((g_data, (g_rows, g_cols)), shape=(n_cols, n_cols))
        self.L = sps.csc_matrix((l_data, (l_rows, l_cols)), shape=(n_cols, n_cols))
        self.terms = terms
        self.theta = theta
        self.t_inds = t_inds
        self.l_inds = l_inds
        self.g_inds = g_inds
        self.jac_inds = jac_inds
        self.g_data = g_data
        self.l_data = l_data
        self.group_sizes = [term.n_group for term in terms]

        
        
        

def replace_duplicate_operators(match):
    return match.group()[-1:]

def parse_random_effects(formula):
    matches = re.findall("\([^)]+[|][^)]+\)", formula)
    groups = [re.search("\(([^)]+)\|([^)]+)\)", x).groups() for x in matches]
    frm = formula
    for x in matches:
        frm = frm.replace(x, "")
    fe_form = re.sub("(\+|\-)(\+|\-)+", replace_duplicate_operators, frm)
    yvars, fe_form = re.split("[~]", fe_form)
    fe_form = re.sub("\+$", "", fe_form)
    y_vars = re.split(",", re.sub("\(|\)", "", yvars))
    y_vars = [x.strip() for x in y_vars]
    return y_vars, fe_form, groups


class LMM(object):
    
    def __init__(self, formula, data):
        y_vars, fe_form, groups = parse_random_effects(formula)
        terms = [RandomEffectTerm(re_form, gr_form, data) 
                       for re_form, gr_form in groups]
        random_effects = RandomEffects(terms)
        X = patsy.dmatrix(fe_form, data=data, return_type='dataframe')
        fe_vars = X.columns
        y = data[y_vars]
        X, Z, y = X.values, random_effects.Z, y.values
        L = random_effects.L
        XtX = sp.sparse.csc_matrix(X.T.dot(X))   # XtX  XtZ  Xty
        ZtX = sp.sparse.csc_matrix(Z.T.dot(X))   # ZtX  ZtZ  Zty
        ytX = sp.sparse.csc_matrix(y.T.dot(X))   # ytX  ytZ  yty
        ZtZ = Z.T.dot(Z)
        ytZ = sp.sparse.csc_matrix(Z.T.dot(y)).T
        yty = sp.sparse.csc_matrix(y.T.dot(y))
        
        M = sp.sparse.bmat([[XtX, ZtX.T, ytX.T],
                            [ZtX, ZtZ,   ytZ.T],
                            [ytX, ytZ,   yty]])
        
        zero_mat1 = sp.sparse.eye(X.shape[1])*0.0
        zero_mat2 = sp.sparse.eye(1)*0.0
        C = sp.sparse.block_diag([zero_mat1, L, zero_mat2])
        
        self.X, self.Z, self.y = X, Z, y
        self.random_effects = random_effects
        self.fe_vars, self.y_vars = fe_vars, y_vars
        self.fe_form, self.groups = fe_form, groups
        self.fe_vars = fe_vars
        self.M = M
        self.C = C
        self.G = random_effects.G
        self.L = random_effects.L
        self.levels = len(random_effects.terms)
        self.g_inds = random_effects.g_inds
        self.l_inds = random_effects.l_inds
        self.t_inds = random_effects.t_inds
        self.ng = random_effects.group_sizes
        self.zero_mat1, self.zero_mat2 = zero_mat1, zero_mat2
        self.ZtZ = ZtZ
        self.Zty = ytZ.T
        self.ZtX = ZtX
        self.Xty = ytX.T
        
    def update_gmat(self, theta, inverse=False):
        G = self.G
        for i in range(self.levels):
            theta_i = theta[self.t_inds[i]]
            if inverse:
                theta_i = np.linalg.inv(invech(theta_i)).reshape(-1, order='F')
            else:
                theta_i = invech(theta_i).reshape(-1, order='F')
            G.data[self.g_inds[i]] = np.tile(theta_i, self.ng[i])
        return G
    
    def update_lmat(self, theta, inverse=False):
        L = self.L
        for i in range(self.levels):
            theta_i = theta[self.t_inds[i]].copy()
            theta_i = vech(np.linalg.cholesky(invech(theta_i)))
            if inverse:
                theta_i = vech(np.linalg.inv(invech_chol(theta_i)))
            L.data[self.l_inds[i]] = np.tile(theta_i, self.ng[i])
        return L
    
    
    def update_cmat(self, theta):
        G = self.update_gmat(theta, inverse=True)
        C = sp.sparse.block_diag([self.zero_mat1, G, self.zero_mat2])
        return C
    
    def update_mme(self, Ginv, theta):
        M =  self.M.copy()/theta[-1]
        C = sp.sparse.block_diag([self.zero_mat1, Ginv, self.zero_mat2])
        M+=C
        return M
    
    def lndet_gmat(self, theta):
        lnd = 0.0
        for i in range(self.levels):
            Sigma_i = invech(theta[self.t_inds[i]])
            lnd += self.ng[i]*np.linalg.slogdet(Sigma_i)[1]
        return lnd
        
    
    def loglike(self, theta, reml=True, use_sw=False, use_sparse=True):
        Ginv = self.update_gmat(theta, inverse=True)
        M = self.update_mme(Ginv, theta)
        if (M.nnz / np.product(M.shape) < 0.05) and use_sparse:
            L = cholesky(M.tocsc()).L().A
        else:
            L = np.linalg.cholesky(M.A)
        ytPy = np.diag(L)[-1]**2
        logdetG = self.lndet_gmat(theta)
        logdetR = np.log(theta[-1]) * self.Z.shape[0]
        if reml:
            logdetC = np.sum(2*np.log(np.diag(L))[:-1])
            ll = logdetR + logdetC + logdetG + ytPy
        else:
            Q = Ginv + self.ZtZ / theta[-1]
            _, logdetV = cholesky(Q).slogdet()
            ll = logdetR + logdetV + logdetG + ytPy
        return ll
    
    


        
        