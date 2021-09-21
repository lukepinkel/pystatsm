# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 05:55:20 2021

@author: lukepinkel
"""

import re
import patsy
import numpy as np
import scipy as sp 
import scipy.sparse as sps
import pandas as pd
from ..utilities.data_utils import dummy, _check_np, _check_shape
from ..utilities.linalg_operations import vech, invech, vec
from ..utilities.special_mats import lmat, nmat
from ..utilities.numerical_derivs import so_gc_cd, so_fc_cd, fo_fc_cd
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
        Zi = sp.linalg.khatri_rao(Ji.T, Xi.T).T
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
        indices[f"y{i+1}_resid"] = np.arange(index_start, index_start+1)
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
    theta[indices['rt']] = np.log(theta[indices['rt']])
    return theta
        
    
def inverse_transform_theta(theta, dims, indices):
    for key in dims.keys():
        L = invech_chol(theta[indices['theta'][key]])
        G = L.dot(L.T)
        theta[indices['theta'][key]] = vech(G)
    theta[indices['rt']] = np.exp(theta[indices['rt']])
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
        
      
def get_jacmats(Zs, dims, indices, theta):
    start = 0
    jac_mats = {}
    jac_inds = {}
    for key, value in dims.items():
        nv, ng =  value['n_vars'], value['n_groups']
        jac_mats[key] = []
        jac_inds[key] = np.arange(start, start+ng*nv)
        theta_i = theta[indices['theta'][key]]
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
    jac_mats['resid'] = []
    for key, ix in indices['resid'].items():
        x = np.zeros(Zs.shape[0])
        x[ix] = 1.0
        dRi = sps.diags(x)
        jac_mats['resid'].append(dRi)
    return jac_mats, jac_inds
      
def sptrace(A):
    tr = A.diagonal().sum()
    return tr

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
            r_indices[f"y{i+1}_resid"] = np.arange(i, n, n_yvars)
            self.ylevels.append(F"y{i+1}_resid")
        self.n_yvars = n_yvars
        indices['resid'] = r_indices
        indices['rt'] = np.arange(len(theta)-n_yvars, len(theta))
        XZ = np.hstack([X, Z])
        XZy = sps.csc_matrix(np.hstack([XZ, y.reshape(-1, 1)]))
        self.X, self.Z, self.y, self.dims, self.levels = X, Z, y.reshape(-1, 1), dims, levels
        self.XZ, self.XZy = XZ, XZy
        self.theta, self.theta_chol = theta, transform_theta(theta, dims, indices)
        self.G = G
        self.indices = indices
        self.R = R.todia()
        self.Zs = sps.csc_matrix(Z)
        self.g_derivs, self.jac_inds = get_jacmats(self.Zs, self.dims, self.indices, self.theta)
        self.n_theta = len(self.theta)
        self.n_gpars = self.n_theta - self.n_yvars
        self.n_rpars = self.n_yvars
        self.fe_vars = fe_vars
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
            R.data[0, self.indices['resid'][key]] = theta_i
        return R
        
    def loglike(self, theta, use_sparse=True, reml=True):
        Ginv = self.update_gmat(theta, inverse=True)
        Rinv = self.update_rmat(theta, inverse=True).copy()
        R = self.update_rmat(theta, inverse=False).copy()
        M = self.update_mme(Ginv, Rinv)
        if (M.nnz / np.product(M.shape) < 0.05) and use_sparse:
            L = cholesky(M.tocsc()).L().A
        else:
            L = np.linalg.cholesky(M.A)
        ytPy = np.diag(L)[-1]**2
        logdetG = lndet_gmat(theta, self.dims, self.indices)
        logdetR = np.log(R.data).sum()
        if reml:
            logdetC = np.sum(2*np.log(np.diag(L))[:-1])
            ll = logdetR + logdetC + logdetG + ytPy
        else:
            Rinv = self.R / theta[-1]
            RZ = Rinv.dot(self.Zs)
            Q = Ginv + self.Zs.T.dot(RZ)
            _, logdetV = cholesky(Q).slogdet()
            ll = logdetR + logdetV + logdetG + ytPy
        return ll
    
    def gradient(self, theta, reml=True):
        Ginv = self.update_gmat(theta, inverse=True).copy()
        Rinv = self.update_rmat(theta, inverse=True).copy()
        RZ = Rinv.dot(self.Zs)
        RX = Rinv.dot(self.X)
        Ry = Rinv.dot(self.y)
        ZtRZ = RZ.T.dot(self.Zs)
        XtRX = self.X.T.dot(RX) 
        ZtRX = RZ.T.dot(self.X)
        ZtRy = RZ.T.dot(self.y)
            
        Q = Ginv + ZtRZ
        M = cholesky(Q).inv()
        
        ZtWZ = ZtRZ - ZtRZ.dot(M).dot(ZtRZ)
        
        MZtRX = M.dot(ZtRX)
        
        XtWX = XtRX - ZtRX.T.dot(MZtRX)
        XtWX_inv = np.linalg.inv(XtWX)
        ZtWX = ZtRX - ZtRZ.dot(MZtRX)
        WX = RX - RZ.dot(MZtRX)
        U = XtWX_inv.dot(WX.T)
        Vy = Ry - RZ.dot(M.dot(ZtRy))
        Py = Vy - WX.dot(U.dot(self.y))
        ZtPy = self.Zs.T.dot(Py)
        grad = []
        for key in self.levels:
            ind = self.jac_inds[key]
            ZtWZi = ZtWZ[ind][:, ind]
            ZtWXi = ZtWX[ind]
            ZtPyi = ZtPy[ind]
            for dGdi in self.g_derivs[key]:
                g1 = dGdi.dot(ZtWZi).diagonal().sum() 
                g2 = ZtPyi.T.dot(dGdi.dot(ZtPyi))
                if reml:
                    g3 = np.trace(XtWX_inv.dot(ZtWXi.T.dot(dGdi.dot(ZtWXi))))
                else:
                    g3 = 0
                gi = g1 - g2 - g3
                grad.append(gi)

        for dR in self.g_derivs['resid']:
            g1 = sptrace(Rinv.dot(dR)) - sptrace(M.dot(RZ.T.dot(dR).dot(RZ)))
            g2 = (dR.dot(Py)).T.dot(Py)
            if reml:
                g3 = np.trace(XtWX_inv.dot(WX.T.dot(dR.dot(WX))))
            else:
                g3 = 0
            gi = g1 - g2 - g3
            grad.append(gi)
        grad = np.concatenate(grad)
        grad = _check_shape(np.array(grad))
        return grad
    
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
    
    def loglike_c(self, theta_chol, reml=True):
        theta = inverse_transform_theta(theta_chol.copy(), self.dims, self.indices)
        return self.loglike(theta, reml)
    
    def gradient_c(self, theta_chol, reml=True):
        theta = inverse_transform_theta(theta_chol.copy(), self.dims, self.indices)
        return self.gradient(theta, reml)
    
    
    def gradient_chol(self, theta_chol, reml=True):
        L_dict = self.update_chol(theta_chol)
        Jf_dict = self.dg_dchol(L_dict)
        Jg = self.gradient_c(theta_chol, reml)
        Jf = sp.linalg.block_diag(*Jf_dict.values()) 
        Jf = np.pad(Jf, [[0, self.n_yvars]])
        ix = self.indices['rt']
        Jf[ix, ix] = np.exp(theta_chol[ix])
        return Jg.dot(Jf)
    
    def hessian_chol(self, theta_chol, reml=True):
        H = so_gc_cd(self.gradient_chol, theta_chol, args=(reml,))
        return H
    
    def hessian(self, theta, reml=True):
        H = so_gc_cd(self.gradient, theta, reml=True)
        return H
    
    def _compute_effects(self, theta=None):
        theta = self.theta if theta is None else theta
        Ginv = self.update_gmat(theta, inverse=True)
        Rinv = self.update_rmat(theta, inverse=True).copy()
        M = self.update_mme(Ginv, Rinv)
        RXZy = self.XZ.T.dot(Rinv.dot(self.y))
        chol_fac = cholesky(M[:-1, :-1].tocsc())
        betau = chol_fac.solve_A(RXZy)
        u = betau[self.X.shape[1]:].reshape(-1)
        beta = betau[:self.X.shape[1]].reshape(-1)
        
        RZ = Rinv.dot(self.Zs)
        Q = Ginv + self.Zs.T.dot(RZ)
        M = cholesky(Q).inv()
        XtRinvX = self.X.T.dot(Rinv.dot(self.X)) 
        XtRinvZ = self.X.T.dot(Rinv.dot(self.Z)) 
        XtVinvX = XtRinvX - XtRinvZ.dot(M.dot(XtRinvZ.T))
        XtVinvX_inv = np.linalg.inv(XtVinvX)
        return beta, XtVinvX_inv, u
    
    def _optimize(self, reml=True, use_grad=True, use_hess=False, approx_hess=False,
                  opt_kws={}):

        default_opt_kws = dict(verbose=0, gtol=1e-6, xtol=1e-6)
        for key, value in default_opt_kws.items():
                if key not in opt_kws.keys():
                    opt_kws[key] = value
        if use_grad:

            if use_hess:
               hess = self.hessian_chol
            elif approx_hess:
                hess = lambda x, reml: so_gc_cd(self.gradient_chol, x, args=(reml,))
            else:
                hess = "3-point"
            optimizer = sp.optimize.minimize(self.loglike_c, self.theta, args=(reml,),
                                             jac=self.gradient_chol, hess=hess, 
                                             options=opt_kws, bounds=self.bounds,
                                             method='trust-constr')
        else:
            jac = lambda x, reml: fo_fc_cd(self.loglike_c, x, args=(reml,))
            hess = lambda x, reml: so_fc_cd(self.loglike_c, x, args=(reml,))
            optimizer = sp.optimize.minimize(self.loglike_c, self.theta, args=(reml,),
                                             jac=jac, hess=hess, bounds=self.bounds,
                                             method='trust-constr', options=opt_kws)
        theta_chol = optimizer.x
        theta = inverse_transform_theta(theta_chol.copy(), self.dims, self.indices)
        return theta, theta_chol, optimizer
    
    def _post_fit(self, theta, theta_chol, optimizer, reml=True,
                  use_grad=True, analytic_se=False):
        
        beta, XtWX_inv, u = self._compute_effects(theta)
        params = np.concatenate([beta, theta])
        re_covs, re_corrs = {}, {}
        for key, value in self.dims.items():
            re_covs[key] = invech(theta[self.indices['theta'][key]].copy())
            C = re_covs[key]
            v = np.diag(np.sqrt(1/np.diag(C)))
            re_corrs[key] = v.dot(C).dot(v)
        
        if analytic_se:
            Htheta = self.hessian(theta)
        elif use_grad:
            Htheta = so_gc_cd(self.gradient, theta)
        else:
            Htheta = so_fc_cd(self.loglike, theta)
        
        self.theta, self.beta, self.u, self.params = theta, beta, u, params
        self.Hinv_beta = XtWX_inv
        self.Hinv_theta = np.linalg.pinv(Htheta/2.0)
        self.se_beta = np.sqrt(np.diag(XtWX_inv))
        self.se_theta = np.sqrt(np.diag(self.Hinv_theta))
        self.se_params = np.concatenate([self.se_beta, self.se_theta])  
        self.optimizer = optimizer
        self.theta_chol = theta_chol
        if reml:
            self.llconst = (self.X.shape[0] - self.X.shape[1])*np.log(2*np.pi)
        else:
            self.llconst = self.X.shape[0] * np.log(2*np.pi)
        self.lltheta = self.optimizer.fun
        self.ll = (self.llconst + self.lltheta)
        self.llf = self.ll / -2.0
        self.re_covs = re_covs
        self.re_corrs = re_corrs
        if reml:
            n = self.X.shape[0] - self.X.shape[1]
            d = len(self.theta)
        else:
            n = self.X.shape[0]
            d = self.X.shape[1] + len(self.theta)
        self.AIC = self.ll + 2.0 * d
        self.AICC = self.ll + 2 * d * n / (n-d-1)
        self.BIC = self.ll + d * np.log(n)
        self.CAIC = self.ll + d * (np.log(n) + 1)
        sumstats = np.array([self.ll, self.llf, self.AIC, self.AICC,
                             self.BIC, self.CAIC])
        self.sumstats = pd.DataFrame(sumstats, index=['ll', 'llf', 'AIC', 'AICC',
                                                      'BIC', 'CAIC'], columns=['value'])
    
    def predict(self, X=None, Z=None):

        if X is None:
            X = self.X
        if Z is None:
            Z = self.Z
        yhat = X.dot(self.beta)+Z.dot(self.u)
        return yhat
    
    def fit(self, reml=True, use_grad=True, use_hess=False, approx_hess=False,
            analytic_se=False, opt_kws={}):

        theta, theta_chol, optimizer = self._optimize(reml, use_grad, use_hess, 
                                                      approx_hess, opt_kws)
        self._post_fit(theta, theta_chol, optimizer, reml, use_grad, 
                       analytic_se)
        param_names = []
        for i in range(self.n_yvars):
            for s in list(self.fe_vars):
                param_names.append(f"y{i+1}~{s}")
        for level in self.levels:
            n_vars = self.dims[level]['n_vars']
            nv = n_vars // self.n_yvars
            a = np.repeat([f"y{i+1}" for i in range(self.n_yvars)], nv)
            b = np.tile(np.arange(nv), self.n_yvars)
            for i, j in list(zip(*np.triu_indices(n_vars))):
                param_names.append(f"{level}:({a[j]}, {a[i]}):G[{b[j]+1}][{b[i]+1}]")
        for i in range(self.n_yvars):
            param_names.append(f"y{i+1} resid_cov")
        self.param_names = param_names
        res = np.vstack((self.params, self.se_params)).T
        res = pd.DataFrame(res, index=param_names, columns=['estimate', 'SE'])
        res['t'] = res['estimate'] / res['SE']
        res['p'] = sp.stats.t(self.X.shape[0]-self.X.shape[1]).sf(np.abs(res['t']))
        self.res = res
    
    
    
    
    
    