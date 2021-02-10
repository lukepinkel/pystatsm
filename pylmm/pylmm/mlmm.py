# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 18:01:18 2021

@author: lukepinkel
"""

import re
import patsy
import numpy as np # analysis:ignore
import scipy as sp # analysis:ignore
import scipy.sparse as sps # analysis:ignore
from ..utilities.linalg_operations import (dummy, vech, invech, _check_np, 
                                           khatri_rao, sparse_woodbury_inversion,
                                           _check_shape, vec)
from ..utilities.special_mats import lmat, nmat
from sksparse.cholmod import cholesky



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

def construct_random_effects(groups, data, n_yvars):
    re_vars, re_groupings = list(zip(*groups))
    re_vars, re_groupings = set(re_vars), set(re_groupings)
    Zdict = dict(zip(re_vars, [_check_np(patsy.dmatrix(x, data=data, return_type='dataframe')) for x in re_vars]))
    Jdict = dict(zip(re_groupings, [dummy(data[x]) for x in re_groupings]))
    dim_dict = {}
    Z = []
    for x, y in groups:
        Ji, Xi = Jdict[y], Zdict[x]
        dim_dict[y] = {'n_groups':Ji.shape[1], 'n_vars':Xi.shape[1]*n_yvars}
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
    n_yvars = len(yvars)
    Z, dim_dict = construct_random_effects(groups, data, n_yvars)
    X = patsy.dmatrix(fe_form, data=data, return_type='dataframe')
    fe_vars = X.columns
    y = data[yvars]
    X, y = _check_np(X), _check_np(y)
    return X, Z, y, dim_dict, list(dim_dict.keys()), fe_vars, yvars

def make_theta(dims, n_yvars):
    theta, indices, index_start = [], {}, 0
    dims = dims.copy()
    for key, value in dims.items():
        n_vars = value['n_vars']
        n_params = int(n_vars * (n_vars+1) //2)
        indices[key] = np.arange(index_start, index_start+n_params)
        theta.append(vech(np.eye(n_vars)))
        index_start += n_params
    for i in range(n_yvars):
        indices[f"y{i+1}_error"] = np.arange(index_start, index_start+1)
        theta.append(np.ones(1))
        index_start+=1
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
        
      
def get_jacmats2(Zs, dims, indices, g_indices, theta):
    start = 0
    jac_mats = {}
    for key, value in dims.items():
        nv, ng =  value['n_vars'], value['n_groups']
        jac_mats[key] = []
        Zi = Zs[:, start:start+ng*nv]
        theta_i = theta[indices[key]]
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
            dVi = Zi.dot(dGi).dot(Zi.T)
            jac_mats[key].append(dVi)
        start+=ng*nv
    jac_mats['error'] = [sps.eye(Zs.shape[0])]
    return jac_mats
      
    

class MLMM:
    
    def __init__(self, formula, data):
        X, Z, y, dims, levels, fe_vars, yvars = construct_model_matrices(formula, data=data)
        n_yvars = len(yvars)
        if n_yvars>1:
            X = np.kron(X, np.eye(n_yvars))
            Z = np.kron(Z, np.eye(n_yvars))
            y = vec(y.T)
        
        indices = {}
        theta, theta_indices = make_theta(dims, n_yvars)
        indices['theta'] = theta_indices

        G, g_indices = make_gcov(theta, indices, dims)
        indices['g'] = g_indices
        n = y.shape[0]
        R = sps.eye(n).tocsc()
        r_indices = {}
        self.ylevels = []
        for i in range(n_yvars):
            r_indices[f"y{i+1}_error"] = np.arange(i, n, n_yvars)
            self.ylevels.append(F"y{i+1}_error")
        self.n_yvars = n_yvars
        indices['error'] = r_indices
        XZ = np.hstack([X, Z])
        XZy = np.hstack([XZ, y.reshape(-1, 1)])
        self.X, self.Z, self.y, self.dims, self.levels = X, Z, y.reshape(-1, 1), dims, levels
        self.XZ, self.XZy = XZ, XZy
        self.theta, self.theta_chol = theta, transform_theta(theta, dims, indices)
        self.G = G
        self.indices = indices
        self.R = R.todia()
        self.Zs = sps.csc_matrix(Z)
        self.jac_mats = get_jacmats2(self.Zs, self.dims, self.indices['theta'], 
                                     self.indices['g'], self.theta)
        
        self.jac_mats['error'] = []
        for key, ix in r_indices.items():
            Rtemp = R*0
            Rtemp[ix, ix] = 1
            self.jac_mats['error'].append(Rtemp)
        self.t_indices = list(zip(*np.triu_indices(len(theta))))
        self.elim_mats, self.symm_mats, self.iden_mats = {}, {}, {}
        self.d2g_dchol = {}
        for key in self.levels:
            p = self.dims[key]['n_vars']
            self.elim_mats[key] = lmat(p).A
            self.symm_mats[key] = nmat(p).A
            self.iden_mats[key] = np.eye(p)
            self.d2g_dchol[key] = get_d2_chol(self.dims[key])
        self.bounds = [(0, None) if x==1 else (None, None) for x in self.theta]
        self.bounds_2 = [(1e-6, None) if x==1 else (None, None) for x in self.theta]
        self.zero_mat = sp.sparse.eye(self.X.shape[1])*0.0
        self.zero_mat2 = sp.sparse.eye(1)*0.0
        
    def update_mme(self, Ginv, Rinv):
        RZXy = Rinv.dot(self.XZy)
        M = self.XZy.T.dot(RZXy)
        Omega = sp.sparse.block_diag([self.zero_mat, Ginv, self.zero_mat2])
        M+=Omega
        return M
    
    def update_gmat(self, theta, inverse=False):
        G = self.G
        for key in self.levels:
            ng = self.dims[key]['n_groups']
            theta_i = theta[self.indices['theta'][key]]
            if inverse:
                theta_i = np.linalg.inv(invech(theta_i)).reshape(-1, order='F')
            else:
                theta_i = invech(theta_i).reshape(-1, order='F')
            G.data[self.indices['g'][key]] = np.tile(theta_i, ng)
        return G
    
    def update_rmat(self, theta, inverse=False):
        R = self.R.copy()
        for key in self.ylevels:
            theta_i = theta[self.indices['theta'][key]]
            if inverse:
                theta_i = 1.0 / theta_i
            else:
                theta_i = theta_i
            R.data[0, self.indices['error'][key]] = theta_i
        return R
        
    def loglike(self, theta, use_sw=False):
        Ginv = self.update_gmat(theta, inverse=True)
        Rinv = self.update_rmat(theta, inverse=True)
        R = self.update_rmat(theta)
        M = self.update_mme(Ginv, Rinv)
        logdetG = lndet_gmat(theta, self.dims, self.indices)
        L = np.linalg.cholesky(M.A)
        ytPy = np.diag(L)[-1]**2
        logdetC = np.sum(2*np.log(np.diag(L))[:-1])
        logdetR = np.log(R.data).sum()
        ll = logdetR + logdetC + logdetG + ytPy
        return ll
    
    def gradient(self, theta, use_sw=False):
        G = self.update_gmat(theta, inverse=False)
        R = self.update_rmat(theta, inverse=False)
        V = self.Zs.dot(G).dot(self.Zs.T) + R
        chol_fac = cholesky(V)
        if use_sw:
            Rinv = self.update_rmat(theta, inverse=True)
            Ginv = self.update_gmat(theta, inverse=True)
            RZ = Rinv.dot(self.Zs)
            Q = Ginv + self.Zs.T.dot(RZ)
            Vinv = Rinv - RZ.dot(cholesky(Q).inv()).dot(RZ.T)
        else:
            Vinv = chol_fac.solve_A(sp.sparse.eye(V.shape[0], format='csc'))
        
        if Vinv.nnz / np.product(Vinv.shape) > 0.1:
            Vinv = Vinv.A
        W = chol_fac.solve_A(self.X)
        XtW = W.T.dot(self.X)
        U = np.linalg.solve(XtW, W.T)
        Py = chol_fac.solve_A(self.y) - W.dot(U.dot(self.y))
        
        grad = []
        for key in (self.levels+['error']):
            for dVdi in self.jac_mats[key]:
                VdVdi = dVdi.dot(Vinv).diagonal().sum()
                trPdV = VdVdi - np.einsum('ij,ji->', W,
                                          sp.sparse.csc_matrix.dot(U, dVdi))
                gi = trPdV - Py.T.dot(dVdi.dot(Py))
                grad.append(gi)
        grad = np.concatenate(grad)
        grad = _check_shape(np.array(grad))
        return grad
    
    def hessian(self, theta):
        Ginv = self.update_gmat(theta, inverse=True)
        Rinv = self.update_rmat(theta, inverse=True)
        Vinv = sparse_woodbury_inversion(self.Zs, Cinv=Ginv, Ainv=Rinv.tocsc())
        W = (Vinv.dot(self.X))
        XtW = W.T.dot(self.X)
        XtW_inv = np.linalg.inv(XtW)
        P = Vinv - np.linalg.multi_dot([W, XtW_inv, W.T])
        Py = P.dot(self.y)
        H = []
        PJ, yPJ = [], []
        for key in (self.levels+['error']):
            J_list = self.jac_mats[key]
            for i in range(len(J_list)):
                Ji = J_list[i].T
                PJ.append((Ji.dot(P)).T)
                yPJ.append((Ji.dot(Py)).T)
        t_indices = self.t_indices
        for i, j in t_indices:
            PJi, PJj = PJ[i], PJ[j]
            yPJi, JjPy = yPJ[i], yPJ[j].T
            Hij = -np.einsum('ij,ji->', PJi, PJj)\
                        + (2 * (yPJi.dot(P)).dot(JjPy))[0]
            H.append(np.array(Hij[0]))
        H = invech(np.concatenate(H)[:, 0])
        return H
    
    def update_chol(self, theta, inverse=False):
        L_dict = {}
        for key in self.levels:
            theta_i = theta[self.indices['theta'][key]]
            L_i = invech_chol(theta_i)
            L_dict[key] = L_i
        return L_dict
    
    def dg_dchol(self, L_dict):
        Jf = {}
        for key in self.levels:
            L = L_dict[key]
            E = self.elim_mats[key]
            N = self.symm_mats[key]
            I = self.iden_mats[key]
            Jf[key] = E.dot(N.dot(np.kron(L, I))).dot(E.T)
        return Jf
    
    def loglike_c(self, theta_chol, use_sw=False):
        theta = inverse_transform_theta(theta_chol.copy(), self.dims, self.indices)
        theta[-self.n_yvars:] = theta_chol[-self.n_yvars:]
        return self.loglike(theta, use_sw)
    
    def gradient_c(self, theta_chol, use_sw=False):
        theta = inverse_transform_theta(theta_chol.copy(), self.dims, self.indices)
        theta[-self.n_yvars:] = theta_chol[-self.n_yvars:]
        return self.gradient(theta, use_sw)
    
    def hessian_c(self, theta_chol):
        theta = inverse_transform_theta(theta_chol.copy(), self.dims, self.indices)
        theta[-self.n_yvars:] = theta_chol[-self.n_yvars:]
        return self.hessian(theta)
    
    def gradient_chol(self, theta_chol, use_sw=False):
        L_dict = self.update_chol(theta_chol)
        Jf_dict = self.dg_dchol(L_dict)
        Jg = self.gradient_c(theta_chol, use_sw)
        Jf = sp.linalg.block_diag(*Jf_dict.values()) 
        Jf = np.pad(Jf, [[0, self.n_yvars]])
        for i in range(1, 1+self.n_yvars):
            Jf[-i, -i] = 1
        return Jg.dot(Jf)
    
    def hessian_chol(self, theta_chol):
        L_dict = self.update_chol(theta_chol)
        Jf_dict = self.dg_dchol(L_dict)
        Hq = self.hessian_c(theta_chol)
        Jg = self.gradient_c(theta_chol)
        Hf = self.d2g_dchol
        Jf = sp.linalg.block_diag(*Jf_dict.values()) 
        Jf = np.pad(Jf, [[0, self.n_yvars]])
        for i in range(1, 1+self.n_yvars):
            Jf[-i, -i] = 1
        A = Jf.T.dot(Hq).dot(Jf)  
        B = np.zeros_like(Hq)
        
        for key in self.levels:
            ix = self.indices['theta'][key]
            Jg_i = Jg[ix]
            Hf_i = Hf[key]
            C = np.einsum('i,ijk->jk', Jg_i, Hf_i)  
            B[ix, ix[:, None]] += C
        H = A + B
        return H
    
    
    
    
    
    
    