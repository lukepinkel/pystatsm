# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 00:01:29 2021

@author: lukepinkel
"""

import re
import tqdm
import patsy
import pandas as pd
import numpy as np # analysis:ignore
import scipy as sp # analysis:ignore
import matplotlib.pyplot as plt
import scipy.sparse as sps # analysis:ignore
from ..utilities.linalg_operations import (dummy, vech, invech, _check_np, 
                                           sparse_woodbury_inversion,
                                           _check_shape)
from ..utilities.special_mats import lmat, nmat
from ..utilities.numerical_derivs import so_gc_cd, so_fc_cd, fo_fc_cd
from ..pyglm.families import (Binomial, ExponentialFamily, Poisson, NegativeBinomial, Gaussian, InverseGaussian)
from ..utilities.output import get_param_table
from sksparse.cholmod import cholesky

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
    jac_mats['resid'] = [sps.eye(Zs.shape[0])]
    return jac_mats
         
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
    

class LMM:
    
    def __init__(self, formula, data, weights=None, rcov=None):
        """
        Parameters
        ----------
        formula : string
            lme4 style formula with random effects specified by terms in 
            parentheses with a bar
            
        data : dataframe
            Dataframe containing data.  Missing values should be dropped 
            manually before passing the dataframe.
            
        weights : ndarray, optional
            Array of model weights. The default is None, which sets the
            weights to one internally.

        Returns
        -------
        None.

        """
        indices = {}
        X, Z, y, dims, levels, fe_vars = construct_model_matrices(formula, data, return_fe=True)
        theta, theta_indices = make_theta(dims)
        indices['theta'] = theta_indices
    
        G, g_indices = make_gcov(theta, indices, dims)
    
        indices['g'] = g_indices
    
    
        XZ, Xty, Zty, yty = np.hstack([X, Z]), X.T.dot(y), Z.T.dot(y), y.T.dot(y)
        XZ = sp.sparse.csc_matrix(XZ)
        C, m = sps.csc_matrix(XZ.T.dot(XZ)), sps.csc_matrix(np.vstack([Xty, Zty]))
        M = sps.bmat([[C, m], [m.T, yty]])
        M = M.tocsc()
        self.fe_vars = fe_vars
        self.X, self.Z, self.y, self.dims, self.levels = X, Z, y, dims, levels
        self.XZ, self.Xty, self.Zty, self.yty = XZ, Xty, Zty, yty
        self.C, self.m, self.M = C, m, M
        self.theta, self.theta_chol = theta, transform_theta(theta, dims, indices)
        self.G = G
        self.indices = indices
        self.R = sps.eye(Z.shape[0])
        self.Zs = sps.csc_matrix(Z)
        self.g_derivs, self.jac_inds = get_jacmats(self.Zs, self.dims, 
                                                   self.indices['theta'],
                                                   self.indices['g'], self.theta)
        self.t_indices = list(zip(*np.triu_indices(len(theta))))
        self.elim_mats, self.symm_mats, self.iden_mats = {}, {}, {}
        self.d2g_dchol = {}
        for key in self.levels:
            p = self.dims[key]['n_vars']
            self.elim_mats[key] = lmat(p).A
            self.symm_mats[key] = nmat(p).A
            self.iden_mats[key] = np.eye(p)
            self.d2g_dchol[key] = get_d2_chol(self.dims[key])
        self.bounds = [(0, None) if x==1 else (None, None) for x in self.theta[:-1]]+[(None, None)]
        self.bounds_2 = [(1e-6, None) if x==1 else (None, None) for x in self.theta[:-1]]+[(None, None)]
        self.zero_mat = sp.sparse.eye(self.X.shape[1])*0.0
        self.zero_mat2 = sp.sparse.eye(1)*0.0
        self.rcov = rcov
        
        if rcov is None:
            self.XtX = self.X.T.dot(self.X)
            self.ZtZ = self.Zs.T.dot(self.Zs)
            self.ZtX = self.Zs.T.dot(self.X)
 

       
    def update_mme(self, Ginv, Rinv):
        """
        Parameters
        ----------
        Ginv: sparse matrix
             scipy sparse matrix with inverse covariance block diagonal
            
        s: float
            resid covariance
        
        Returns
        -------
        M: sparse matrix
            updated mixed model matrix
            
        """
        if type(Rinv) in [float, int, np.float64, np.float32, np.float16,
                          np.int, np.int16, np.int32, np.int64]:
            M = self.M.copy()/Rinv
        else:
            RZX = Rinv.dot(self.XZ)
            C = sps.csc_matrix(RZX.T.dot(self.XZ))
            Ry = Rinv.dot(self.y)
            m = sps.csc_matrix(np.vstack([self.X.T.dot(Ry), self.Zs.T.dot(Ry)]))
            M = sps.bmat([[C, m], [m.T, self.y.T.dot(Ry)]]).tocsc()
        Omega = sp.sparse.block_diag([self.zero_mat, Ginv, self.zero_mat2])
        M+=Omega
        return M
    
    def update_gmat(self, theta, inverse=False):
        """
        Parameters
        ----------
        theta: ndarray
             covariance parameters on the original scale
            
        inverse: bool
            whether or not to inverse G
        
        Returns
        -------
        G: sparse matrix
            updated random effects covariance
            
        """
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
        
    def loglike(self, theta, reml=True, use_sw=False, use_sparse=True):
        """
        Parameters
        ----------
        
        theta: array_like
            The original parameterization of the model parameters
        
        Returns
        -------
        loglike: scalar
            Log likelihood of the model
        """
        Ginv = self.update_gmat(theta, inverse=True)
        M = self.update_mme(Ginv, theta[-1])
        if (M.nnz / np.product(M.shape) < 0.05) and use_sparse:
            L = cholesky(M.tocsc()).L().A
        else:
            L = np.linalg.cholesky(M.A)
        ytPy = np.diag(L)[-1]**2
        logdetG = lndet_gmat(theta, self.dims, self.indices)
        logdetR = np.log(theta[-1]) * self.Z.shape[0]
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

    def vinvcrossprod(self, X, theta):
        """

        Parameters
        ----------
        X : ndarray
            Array with first dimension equal to number of observations.
        theta : ndarray
            covariance parameters.

        Returns
        -------
        XtVX : ndarray
            X' V^{-1} X.

        """
        Rinv = self.R / theta[-1]
        Ginv = self.update_gmat(theta, inverse=True)
        RZ = Rinv.dot(self.Zs)
        Q = Ginv + self.Zs.T.dot(RZ)
        M = cholesky(Q).inv()
        XtRX = X.T.dot(Rinv.dot(X)) 
        XtRZ = X.T.dot(Rinv.dot(self.Z)) 
        XtVX = XtRX - XtRZ.dot(M.dot(XtRZ.T))
        return XtVX

    
    def gradient(self, theta, reml=True, use_sw=False):
        """
        Parameters
        ----------
        theta: array_like
            The original parameterization of the components
        
        Returns
        -------
        gradient: array_like
            The gradient of the log likelihood with respect to the covariance
            parameterization
        
        Notes
        -----

            
        """
        s = theta[-1]
        Rinv = self.R / s
        Ginv = self.update_gmat(theta, inverse=True)
        if self.rcov is None:
            RZ = self.Zs / s
            RX = self.X / s
            Ry = self.y / s
            ZtRZ = self.ZtZ / s
            XtRX = self.XtX / s
            ZtRX = self.ZtX / s
            ZtRy = self.Zty / s
        else:
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
        for key in (self.levels):
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
            g1 = Rinv.diagonal().sum() - (M.dot((RZ.T).dot(dR).dot(RZ))).diagonal().sum()
            g2 = Py.T.dot(Py)
            if reml:
                g3 = np.trace(XtWX_inv.dot(WX.T.dot(WX)))
            else:
                g3 = 0
            gi = g1 - g2 - g3
            grad.append(gi)
        grad = np.concatenate(grad)
        grad = _check_shape(np.array(grad))
        return grad
    
    def hessian(self, theta, reml=True, use_sw=False):
        """
        Parameters
        ----------
        theta: array_like
            The original parameterization of the components
        
        Returns
        -------
        H: array_like
            The hessian of the log likelihood with respect to the covariance
            parameterization
        
        Notes
        -----
        This function has the infrastructure to support more complex residual
        covariances that are yet to be implemented.  

        """
        Ginv = self.update_gmat(theta, inverse=True)
        Rinv = self.R / theta[-1]
        RZ = Rinv.dot(self.Zs)
        Q = Ginv + self.Zs.T.dot(RZ)
        M = cholesky(Q).inv()
        W = Rinv - RZ.dot(M).dot(RZ.T)

        WZ = W.dot(self.Zs)
        WX = W.dot(self.X)
        XtWX = WX.T.dot(self.X)
        ZtWX = self.Zs.T.dot(WX)
        U = np.linalg.solve(XtWX, WX.T)
        ZtP = WZ.T - ZtWX.dot(np.linalg.solve(XtWX, WX.T))
        ZtPZ = self.Zs.T.dot(ZtP.T)
        Py = W.dot(self.y) - WX.dot(U.dot(self.y))
        ZtPy = self.Zs.T.dot(Py)
        PPy = W.dot(Py) - WX.dot(U.dot(Py))
        ZtPPy =  self.Zs.T.dot(PPy)
        H = np.zeros((len(self.theta), len(self.theta)))
        PJ, yPZJ, ZPJ = [], [], []
        ix = []
        for key in (self.levels):
            ind = self.jac_inds[key]
            ZtPZi = ZtPZ[ind]
            ZtPyi = ZtPy[ind]
            ZtPi = ZtP[ind]
            for i in range(len(self.g_derivs[key])):
                Gi = self.g_derivs[key][i]
                PJ.append(Gi.dot(ZtPZi))
                yPZJ.append(Gi.dot(ZtPyi))
                ZPJ.append((Gi.dot(ZtPi)).T)
                ix.append(ind)
            
        t_indices = list(zip(*np.triu_indices(len(self.theta)-1)))
        for i, j in t_indices:
            ZtPZij = ZtPZ[ix[i]][:, ix[j]]
            PJi, PJj = PJ[i][:, ix[j]], PJ[j][:, ix[i]]
            yPZJi, JjZPy = yPZJ[i], yPZJ[j]
            Hij = -np.einsum('ij,ji->', PJi, PJj)\
                    + (2 * (yPZJi.T.dot(ZtPZij)).dot(JjZPy))[0]
            H[i, j] = H[j, i] = Hij
        dR = self.g_derivs['resid'][0]
        dRZtP = (dR.dot(ZtP.T))
        for i in range(len(self.theta)-1):
            yPZJi = yPZJ[i]
            ZPJi = ZPJ[i]
            ZtPPyi = ZtPPy[ix[i]]
            H[i, -1] = H[-1, i] = 2*yPZJi.T.dot(ZtPPyi) - np.einsum('ij,ji->', ZPJi.T, dRZtP[:, ix[i]])
        P = W - WX.dot(U)
        H[-1, -1] = Py.T.dot(PPy)*2 - np.einsum("ij,ji->", P, P)
        return H
    
    def update_chol(self, theta, inverse=False):
        """
        Parameters
        ----------
        theta: array_like
            array containing the lower triangular components of the cholesky
            for each random effect covariance
            
        inverse: bool
        
        Returns
        -------
        L_dict: dict of array_like
            Dictionary whose keys and values correspond to level names
            and the corresponding cholesky of the level's random effects 
            covariance
            
        """
        L_dict = {}
        for key in self.levels:
            theta_i = theta[self.indices['theta'][key]]
            L_i = invech_chol(theta_i)
            L_dict[key] = L_i
        return L_dict
    
    def dg_dchol(self, L_dict):
        """
        
        Parameters
        ----------
        
        L_dict: dict of array_like
            Dictionary whose keys and values correspond to level names
            and the corresponding cholesky of the level's random effects 
            covariance
        
        
        Returns
        -------
        
        Jf: dict of array_like
            For each level contains the derivative of the cholesky parameters
            with respect to the covariance
        
        Notes
        -----
        
        Function evaluates the derivative of the cholesky parameterization 
        with respect to the lower triangular components of the covariance
        
        """
        
        Jf = {}
        for key in self.levels:
            L = L_dict[key]
            E = self.elim_mats[key]
            N = self.symm_mats[key]
            I = self.iden_mats[key]
            Jf[key] = E.dot(N.dot(np.kron(L, I))).dot(E.T)
        return Jf
    
    def loglike_c(self, theta_chol, reml=True, use_sw=False):
        """
        Parameters
        ----------
        
        theta_chol: array_like
            The cholesky parameterization of the components
        
        Returns
        -------
        loglike: scalar
            Log likelihood of the model
        """
        theta = inverse_transform_theta(theta_chol.copy(), self.dims, self.indices)
        return self.loglike(theta, reml, use_sw)
    
    def gradient_c(self, theta_chol, reml=True, use_sw=False):
        """
        Parameters
        ----------
        
        theta_chol: array_like
            The cholesky parameterization of the components
        
        Returns
        -------
        gradient: array_like
            The gradient of the log likelihood with respect to the covariance
            parameterization
            
        """
        theta = inverse_transform_theta(theta_chol.copy(), self.dims, self.indices)
        return self.gradient(theta, reml, use_sw)
    
    
    def hessian_c(self, theta_chol, reml=True):
        """
        Parameters
        ----------
        
        theta_chol: array_like
            The cholesky parameterization of the components
        
        Returns
        -------
        hessian: array_like
            The hessian of the log likelihood with respect to the covariance
            parameterization
            
        """
        theta = inverse_transform_theta(theta_chol.copy(), self.dims, self.indices)
        return self.hessian(theta, reml)
    
    def gradient_chol(self, theta_chol, reml=True, use_sw=False):
        """
        Parameters
        ----------
        
        theta_chol: array_like
            The cholesky parameterization of the components
        
        Returns
        -------
        gradient: array_like
            The gradient of the log likelihood with respect to the cholesky
            parameterization
            
        """
        L_dict = self.update_chol(theta_chol)
        Jf_dict = self.dg_dchol(L_dict)
        Jg = self.gradient_c(theta_chol, reml, use_sw)
        Jf = sp.linalg.block_diag(*Jf_dict.values()) 
        Jf = np.pad(Jf, [[0, 1]])
        Jf[-1, -1] =  np.exp(theta_chol[-1])
        return Jg.dot(Jf)
    
    def hessian_chol(self, theta_chol, reml=True):
        """
        Parameters
        ----------
        
        theta_chol: array_like
            The cholesky parameterization of the components
        
        Returns
        -------
        hessian: array_like
            The hessian of the log likelihood with respect to the cholesky
            parameterization
            
        """
        L_dict = self.update_chol(theta_chol)
        Jf_dict = self.dg_dchol(L_dict)
        Hq = self.hessian_c(theta_chol, reml)
        Jg = self.gradient_c(theta_chol, reml)
        Hf = self.d2g_dchol
        Jf = sp.linalg.block_diag(*Jf_dict.values()) 
        Jf = np.pad(Jf, [[0, 1]])
        Jf[-1, -1] = np.exp(theta_chol[-1])
        A = Jf.T.dot(Hq).dot(Jf)  
        B = np.zeros_like(Hq)
        
        for key in self.levels:
            ix = self.indices['theta'][key]
            Jg_i = Jg[ix]
            Hf_i = Hf[key]
            C = np.einsum('i,ijk->jk', Jg_i, Hf_i)  
            B[ix, ix[:, None]] += C
        B[-1, -1] = Jg[-1] * np.exp(theta_chol[-1])
        H = A + B
        return H
    
    def _compute_effects(self, theta=None):
        """

        Parameters
        ----------
        theta : ndarray, optional
            Model parameters in the covariance form

        Returns
        -------
        beta : ndarray
            Fixed effects estimated at theta.
        XtViX_inv : ndarray
            Fixed effects covariance matrix.
        u : ndarray
            Random effect estimate at theta.
        G : csc_matrix
            Random effects covariance matrix.
        R : dia_matrix
            Matrix of residual covariance.
        V : csc_matrix
            Model covariance matrix given fixed effects.

        """
        theta = self.theta if theta is None else theta
        Ginv = self.update_gmat(theta, inverse=True)
        M = self.update_mme(Ginv, theta[-1])
        XZy = self.XZ.T.dot(self.y) / theta[-1]
        chol_fac = cholesky(M[:-1, :-1].tocsc())
        betau = chol_fac.solve_A(XZy)
        u = betau[self.X.shape[1]:].reshape(-1)
        beta = betau[:self.X.shape[1]].reshape(-1)
        
        Rinv = self.R / theta[-1]
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
        """

        Parameters
        ----------
        use_grad : bool, optional
            If true, the analytic gradient is used during optimization.
            The default is True.
        use_hess : bool, optional
            If true, the analytic hessian is used during optimization.
            The default is False.
        approx_hess: bool, optional
            If true, uses the gradient to approximate the hessian
        opt_kws : dict, optional
            Dictionary of options to use in scipy.optimize.minimize.
            The default is {}.

        Returns
        -------
        None.

        """
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
                hess = None
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
        """

        Parameters
        ----------
        use_grad : bool, optional
            If true and analytic_se is False, the gradient is used in the
            numerical approximation of the hessian. The default is True.
        analytic_se : bool, optional
            If true, then the hessian is used to compute standard errors.
            The default is False.

        Returns
        -------
        None.

        """
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
        """
        Parameters
        ----------
        X : ndarray, optional
            Model matrix for fixed effects. The default is None.
        Z : ndarray, optional
            Model matrix from random effects. The default is None.

        Returns
        -------
        yhat : ndarray
            Model predictions evaluated at X and Z.

        """
        if X is None:
            X = self.X
        if Z is None:
            Z = self.Z
        yhat = X.dot(self.beta)+Z.dot(self.u)
        return yhat
    
    def fit(self, reml=True, use_grad=True, use_hess=False, approx_hess=False,
            analytic_se=False, opt_kws={}):
        """
        

        Parameters
        ----------
        use_grad : bool, optional
            If true, the analytic gradient is used during optimization.
            The default is True.
        use_hess : bool, optional
            If true, the analytic hessian is used during optimization.
            The default is False.
        approx_hess: bool, optional
            If true, uses the gradient to approximate the hessian
        analytic_se : bool, optional
            If true, then the hessian is used to compute standard errors.
            The default is False.
        opt_kws : dict, optional
            Dictionary of options to use in scipy.optimize.minimize.
            The default is {}.

        Returns
        -------
        None.

        """
        theta, theta_chol, optimizer = self._optimize(reml, use_grad, use_hess, 
                                                      approx_hess, opt_kws)
        self._post_fit(theta, theta_chol, optimizer, reml, use_grad, 
                       analytic_se)
        param_names = list(self.fe_vars)
        for level in self.levels:
            for i, j in list(zip(*np.triu_indices(self.dims[level]['n_vars']))):
                param_names.append(f"{level}:G[{i}][{j}]")
        param_names.append("resid_cov")
        self.param_names = param_names
        res = np.vstack((self.params, self.se_params)).T
        res = pd.DataFrame(res, index=param_names, columns=['estimate', 'SE'])
        res['t'] = res['estimate'] / res['SE']
        res['p'] = sp.stats.t(self.X.shape[0]-self.X.shape[1]).sf(np.abs(res['t']))
        self.res = res
    
    
    def _restricted_ll_grad(self, theta_chol_f, free_ix, theta_chol_r, reml=True):
        theta_chol_r[free_ix] = theta_chol_f
        ll = self.loglike_c(theta_chol_r.copy(), reml)
        g = self.gradient_chol(theta_chol_r.copy(), reml)[free_ix]
        return ll, g
    
    def profile(self, n_points=40, par_ind=None, reml=True):
        par_ind = np.ones_like(self.theta_chol) if par_ind is None else par_ind
        theta_chol = self.theta_chol.copy()
        n_theta = len(theta_chol)
        
 
        llmax = self.loglike(self.theta.copy())
        free_ix = np.ones_like(theta_chol, dtype=bool)
        
        Hchol = so_gc_cd(self.gradient_chol, theta_chol, args=(reml,))
        se_chol = np.diag(np.linalg.inv(Hchol/2.0))**0.5
        thetas, zetas = np.zeros((n_theta*n_points, n_theta)), np.zeros(n_theta*n_points)
        k = 0
        pbar = tqdm.tqdm(total=n_theta*n_points, smoothing=0.001)
        for i in range(n_theta):
            free_ix[i] = False
            t_mle = theta_chol[i]
            theta_chol_r = theta_chol.copy()
            if self.bounds[i][0]==0:
                lb = np.maximum(0.01, t_mle-4.5*se_chol[i])
            else:
                lb = t_mle - 4.5 * se_chol[i]
            ub = t_mle + 4.5 * se_chol[i]
            tspace = np.linspace(lb, ub, n_points)
            for t0 in tspace:
                theta_chol_r = theta_chol.copy()
                theta_chol_r[~free_ix] = t0
                theta_chol_f = theta_chol[free_ix]
                func = lambda x : self._restricted_ll_grad(x, free_ix, theta_chol_r,
                                                           reml)
                bounds = np.array(self.bounds)[free_ix].tolist()
                opt = sp.optimize.minimize(func, theta_chol_f, jac=True,
                                           bounds=bounds,
                                           method='trust-constr')
                theta_chol_f = opt.x
                theta_chol_r[free_ix] = theta_chol_f
                LR = 2.0 * (opt.fun - llmax)
                zeta = np.sqrt(LR) * np.sign(t0 - theta_chol[~free_ix])
                zetas[k] = zeta
                thetas[k] = theta_chol_r
                k+=1
                pbar.update(1)
            free_ix[i] = True
        pbar.close()
        ix = np.repeat(np.arange(n_theta), n_points)
        return thetas, zetas, ix
    
    def plot_profile(self, n_points=40, par_ind=None, reml=True, quantiles=None):
        if quantiles is None:
            quantiles = [0.001, 0.05, 1, 5, 10, 20, 50, 80, 90, 95, 99, 99.5, 99.999]   
        thetas, zetas, ix = self.profile(n_points, par_ind, reml)
        n_thetas = thetas.shape[1]
        q = sp.stats.norm(0, 1).ppf(np.array(quantiles)/100)
        fig, axes = plt.subplots(figsize=(14, 4), ncols=n_thetas, sharey=True)
        plt.subplots_adjust(wspace=0.05, left=0.05, right=0.95)
        for i in range(n_thetas):
            ax = axes[i]
            x = thetas[ix==i, i]
            y = zetas[ix==i]
            trunc = (y>-5)&(y<5)
            x, y = x[trunc], y[trunc]
            f_interp = sp.interpolate.interp1d(y, x, fill_value="extrapolate")
            xq = f_interp(q)
            ax.plot(x,y)
            ax.set_xlim(x.min(), x.max())
            ax.axhline(0, color='k')
            for a, b in list(zip(xq, q)):
                ax.plot((a, a), (0, b), color='k')
        ax.set_ylim(-5, 5)
        return thetas, zetas, ix, fig, ax

 

class WLMM:
    
    def __init__(self, formula, data, weights=None, fixed_resid_cov=False):
        """
        Parameters
        ----------
        formula : string
            lme4 style formula with random effects specified by terms in 
            parentheses with a bar
            
        data : dataframe
            Dataframe containing data.  Missing values should be dropped 
            manually before passing the dataframe.
            
        weights : ndarray, optional
            Array of model weights. The default is None, which sets the
            weights to one internally.

        Returns
        -------
        None.

        """
        if weights is None:
            weights = np.eye(len(data))
        self.weights = sps.csc_matrix(weights)
        self.weights_inv = sps.csc_matrix(np.linalg.inv(weights))
      
        indices = {}
        X, Z, y, dims, levels, fe_vars = construct_model_matrices(formula, data, return_fe=True)
        theta, theta_indices = make_theta(dims)
        indices['theta'] = theta_indices
    
        G, g_indices = make_gcov(theta, indices, dims)
    
        indices['g'] = g_indices
    
    
        XZ, Xty, Zty, yty = np.hstack([X, Z]), X.T.dot(y), Z.T.dot(y), y.T.dot(y)
        XZ = sp.sparse.csc_matrix(XZ)
        C, m = sps.csc_matrix(XZ.T.dot(XZ)), sps.csc_matrix(np.vstack([Xty, Zty]))
        M = sps.bmat([[C, m], [m.T, yty]])
        M = M.tocsc()
        self.fe_vars = fe_vars
        self.X, self.Z, self.y, self.dims, self.levels = X, Z, y, dims, levels
        self.XZ, self.Xty, self.Zty, self.yty = XZ, Xty, Zty, yty
        self.C, self.m, self.M = C, m, M
        self.theta, self.theta_chol = theta, transform_theta(theta, dims, indices)
        self.G = G
        self.indices = indices
        self.R = sps.eye(Z.shape[0])
        self.Zs = sps.csc_matrix(Z)
        self.g_derivs, self.jac_inds = get_jacmats(self.Zs, self.dims, 
                                                   self.indices['theta'],
                                                   self.indices['g'], self.theta)
        self.t_indices = list(zip(*np.triu_indices(len(theta))))
        self.elim_mats, self.symm_mats, self.iden_mats = {}, {}, {}
        self.d2g_dchol = {}
        for key in self.levels:
            p = self.dims[key]['n_vars']
            self.elim_mats[key] = lmat(p).A
            self.symm_mats[key] = nmat(p).A
            self.iden_mats[key] = np.eye(p)
            self.d2g_dchol[key] = get_d2_chol(self.dims[key])
        self.bounds = [(0, None) if x==1 else (None, None) for x in self.theta[:-1]]+[(None, None)]
        self.bounds_2 = [(1e-6, None) if x==1 else (None, None) for x in self.theta[:-1]]+[(None, None)]
        self.zero_mat = sp.sparse.eye(self.X.shape[1])*0.0
        self.zero_mat2 = sp.sparse.eye(1)*0.0
        self.rcov = self.weights
    
        self.fixed_resid_cov = fixed_resid_cov

       
    def update_mme(self, Ginv, Rinv):
        """
        Parameters
        ----------
        Ginv: sparse matrix
             scipy sparse matrix with inverse covariance block diagonal
            
        s: float
            resid covariance
        
        Returns
        -------
        M: sparse matrix
            updated mixed model matrix
            
        """
        if type(Rinv) in [float, int, np.float64, np.float32, np.float16,
                          np.int, np.int16, np.int32, np.int64]:
            M = self.M.copy()/Rinv
        else:
            RZX = Rinv.dot(self.XZ)
            C = sps.csc_matrix(RZX.T.dot(self.XZ))
            Ry = Rinv.dot(self.y)
            m = sps.csc_matrix(np.vstack([self.X.T.dot(Ry), self.Zs.T.dot(Ry)]))
            M = sps.bmat([[C, m], [m.T, self.y.T.dot(Ry)]]).tocsc()
        Omega = sp.sparse.block_diag([self.zero_mat, Ginv, self.zero_mat2])
        M+=Omega
        return M
    
    def update_gmat(self, theta, inverse=False):
        """
        Parameters
        ----------
        theta: ndarray
             covariance parameters on the original scale
            
        inverse: bool
            whether or not to inverse G
        
        Returns
        -------
        G: sparse matrix
            updated random effects covariance
            
        """
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
        
    def loglike(self, theta, reml=True, use_sw=False, use_sparse=True):
        """
        Parameters
        ----------
        
        theta: array_like
            The original parameterization of the model parameters
        
        Returns
        -------
        loglike: scalar
            Log likelihood of the model
        """
        s = 1.0 if self.fixed_resid_cov else theta[-1]
        Rinv = self.weights_inv.dot(self.R / s).dot(self.weights_inv)
        Ginv = self.update_gmat(theta, inverse=True)
        M = self.update_mme(Ginv, Rinv)
        if (M.nnz / np.product(M.shape) < 0.05) and use_sparse:
            L = cholesky(M.tocsc()).L().A
        else:
            L = np.linalg.cholesky(M.A)
        ytPy = np.diag(L)[-1]**2
        logdetG = lndet_gmat(theta, self.dims, self.indices)
        logdetR = np.log(theta[-1]) * self.Z.shape[0]
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

    def vinvcrossprod(self, X, theta):
        """

        Parameters
        ----------
        X : ndarray
            Array with first dimension equal to number of observations.
        theta : ndarray
            covariance parameters.

        Returns
        -------
        XtVX : ndarray
            X' V^{-1} X.

        """
        s = 1.0 if self.fixed_resid_cov else theta[-1]
        Rinv = self.weights_inv.dot(self.R / s).dot(self.weights_inv)
        Ginv = self.update_gmat(theta, inverse=True)
        RZ = Rinv.dot(self.Zs)
        Q = Ginv + self.Zs.T.dot(RZ)
        M = cholesky(Q).inv()
        XtRX = X.T.dot(Rinv.dot(X)) 
        XtRZ = X.T.dot(Rinv.dot(self.Z)) 
        XtVX = XtRX - XtRZ.dot(M.dot(XtRZ.T))
        return XtVX

    
    def gradient(self, theta, reml=True, use_sw=False):
        """
        Parameters
        ----------
        theta: array_like
            The original parameterization of the components
        
        Returns
        -------
        gradient: array_like
            The gradient of the log likelihood with respect to the covariance
            parameterization
        
        Notes
        -----

            
        """
        s = 1.0 if self.fixed_resid_cov else theta[-1]
        Rinv = self.weights_inv.dot(self.R / s).dot(self.weights_inv)
        Ginv = self.update_gmat(theta, inverse=True)

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
        for key in (self.levels):
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
            g1 = Rinv.diagonal().sum() - (M.dot((RZ.T).dot(dR).dot(RZ))).diagonal().sum()
            g2 = Py.T.dot(Py)
            if reml:
                g3 = np.trace(XtWX_inv.dot(WX.T.dot(WX)))
            else:
                g3 = 0
            gi = g1 - g2 - g3
            grad.append(gi)
        grad = np.concatenate(grad)
        grad = _check_shape(np.array(grad))
        return grad
    
    def hessian(self, theta, reml=True, use_sw=False):
        """
        Parameters
        ----------
        theta: array_like
            The original parameterization of the components
        
        Returns
        -------
        H: array_like
            The hessian of the log likelihood with respect to the covariance
            parameterization
        
        Notes
        -----
        This function has the infrastructure to support more complex residual
        covariances that are yet to be implemented.  

        """
        s = 1.0 if self.fixed_resid_cov else theta[-1]
        Rinv = self.weights_inv.dot(self.R / s).dot(self.weights_inv)
        Ginv = self.update_gmat(theta, inverse=True)

        RZ = Rinv.dot(self.Zs)
        Q = Ginv + self.Zs.T.dot(RZ)
        M = cholesky(Q).inv()
        W = Rinv - RZ.dot(M).dot(RZ.T)

        WZ = W.dot(self.Zs)
        WX = W.dot(self.X)
        XtWX = WX.T.dot(self.X)
        ZtWX = self.Zs.T.dot(WX)
        U = np.linalg.solve(XtWX, WX.T)
        ZtP = WZ.T - ZtWX.dot(np.linalg.solve(XtWX, WX.T))
        ZtPZ = self.Zs.T.dot(ZtP.T)
        Py = W.dot(self.y) - WX.dot(U.dot(self.y))
        ZtPy = self.Zs.T.dot(Py)
        PPy = W.dot(Py) - WX.dot(U.dot(Py))
        ZtPPy =  self.Zs.T.dot(PPy)
        H = np.zeros((len(self.theta), len(self.theta)))
        PJ, yPZJ, ZPJ = [], [], []
        ix = []
        for key in (self.levels):
            ind = self.jac_inds[key]
            ZtPZi = ZtPZ[ind]
            ZtPyi = ZtPy[ind]
            ZtPi = ZtP[ind]
            for i in range(len(self.g_derivs[key])):
                Gi = self.g_derivs[key][i]
                PJ.append(Gi.dot(ZtPZi))
                yPZJ.append(Gi.dot(ZtPyi))
                ZPJ.append((Gi.dot(ZtPi)).T)
                ix.append(ind)
            
        t_indices = list(zip(*np.triu_indices(len(self.theta)-1)))
        for i, j in t_indices:
            ZtPZij = ZtPZ[ix[i]][:, ix[j]]
            PJi, PJj = PJ[i][:, ix[j]], PJ[j][:, ix[i]]
            yPZJi, JjZPy = yPZJ[i], yPZJ[j]
            Hij = -np.einsum('ij,ji->', PJi, PJj)\
                    + (2 * (yPZJi.T.dot(ZtPZij)).dot(JjZPy))[0]
            H[i, j] = H[j, i] = Hij
        dR = self.g_derivs['resid'][0]
        dRZtP = (dR.dot(ZtP.T))
        for i in range(len(self.theta)-1):
            yPZJi = yPZJ[i]
            ZPJi = ZPJ[i]
            ZtPPyi = ZtPPy[ix[i]]
            H[i, -1] = H[-1, i] = 2*yPZJi.T.dot(ZtPPyi) - np.einsum('ij,ji->', ZPJi.T, dRZtP[:, ix[i]])
        P = W - WX.dot(U)
        H[-1, -1] = Py.T.dot(PPy)*2 - np.einsum("ij,ji->", P, P)
        return H
    
    def update_chol(self, theta, inverse=False):
        """
        Parameters
        ----------
        theta: array_like
            array containing the lower triangular components of the cholesky
            for each random effect covariance
            
        inverse: bool
        
        Returns
        -------
        L_dict: dict of array_like
            Dictionary whose keys and values correspond to level names
            and the corresponding cholesky of the level's random effects 
            covariance
            
        """
        L_dict = {}
        for key in self.levels:
            theta_i = theta[self.indices['theta'][key]]
            L_i = invech_chol(theta_i)
            L_dict[key] = L_i
        return L_dict
    
    def dg_dchol(self, L_dict):
        """
        
        Parameters
        ----------
        
        L_dict: dict of array_like
            Dictionary whose keys and values correspond to level names
            and the corresponding cholesky of the level's random effects 
            covariance
        
        
        Returns
        -------
        
        Jf: dict of array_like
            For each level contains the derivative of the cholesky parameters
            with respect to the covariance
        
        Notes
        -----
        
        Function evaluates the derivative of the cholesky parameterization 
        with respect to the lower triangular components of the covariance
        
        """
        
        Jf = {}
        for key in self.levels:
            L = L_dict[key]
            E = self.elim_mats[key]
            N = self.symm_mats[key]
            I = self.iden_mats[key]
            Jf[key] = E.dot(N.dot(np.kron(L, I))).dot(E.T)
        return Jf
    
    def loglike_c(self, theta_chol, reml=True, use_sw=False):
        """
        Parameters
        ----------
        
        theta_chol: array_like
            The cholesky parameterization of the components
        
        Returns
        -------
        loglike: scalar
            Log likelihood of the model
        """
        theta = inverse_transform_theta(theta_chol.copy(), self.dims, self.indices)
        return self.loglike(theta, reml, use_sw)
    
    def gradient_c(self, theta_chol, reml=True, use_sw=False):
        """
        Parameters
        ----------
        
        theta_chol: array_like
            The cholesky parameterization of the components
        
        Returns
        -------
        gradient: array_like
            The gradient of the log likelihood with respect to the covariance
            parameterization
            
        """
        theta = inverse_transform_theta(theta_chol.copy(), self.dims, self.indices)
        return self.gradient(theta, reml, use_sw)
    
    
    def hessian_c(self, theta_chol, reml=True):
        """
        Parameters
        ----------
        
        theta_chol: array_like
            The cholesky parameterization of the components
        
        Returns
        -------
        hessian: array_like
            The hessian of the log likelihood with respect to the covariance
            parameterization
            
        """
        theta = inverse_transform_theta(theta_chol.copy(), self.dims, self.indices)
        return self.hessian(theta, reml)
    
    def gradient_chol(self, theta_chol, reml=True, use_sw=False):
        """
        Parameters
        ----------
        
        theta_chol: array_like
            The cholesky parameterization of the components
        
        Returns
        -------
        gradient: array_like
            The gradient of the log likelihood with respect to the cholesky
            parameterization
            
        """
        L_dict = self.update_chol(theta_chol)
        Jf_dict = self.dg_dchol(L_dict)
        Jg = self.gradient_c(theta_chol, reml, use_sw)
        Jf = sp.linalg.block_diag(*Jf_dict.values()) 
        Jf = np.pad(Jf, [[0, 1]])
        Jf[-1, -1] =  np.exp(theta_chol[-1])
        return Jg.dot(Jf)
    
    def hessian_chol(self, theta_chol, reml=True):
        """
        Parameters
        ----------
        
        theta_chol: array_like
            The cholesky parameterization of the components
        
        Returns
        -------
        hessian: array_like
            The hessian of the log likelihood with respect to the cholesky
            parameterization
            
        """
        L_dict = self.update_chol(theta_chol)
        Jf_dict = self.dg_dchol(L_dict)
        Hq = self.hessian_c(theta_chol, reml)
        Jg = self.gradient_c(theta_chol, reml)
        Hf = self.d2g_dchol
        Jf = sp.linalg.block_diag(*Jf_dict.values()) 
        Jf = np.pad(Jf, [[0, 1]])
        Jf[-1, -1] = np.exp(theta_chol[-1])
        A = Jf.T.dot(Hq).dot(Jf)  
        B = np.zeros_like(Hq)
        
        for key in self.levels:
            ix = self.indices['theta'][key]
            Jg_i = Jg[ix]
            Hf_i = Hf[key]
            C = np.einsum('i,ijk->jk', Jg_i, Hf_i)  
            B[ix, ix[:, None]] += C
        B[-1, -1] = Jg[-1] * np.exp(theta_chol[-1])
        H = A + B
        return H
    
    def _compute_effects(self, theta=None):
        """

        Parameters
        ----------
        theta : ndarray, optional
            Model parameters in the covariance form

        Returns
        -------
        beta : ndarray
            Fixed effects estimated at theta.
        XtViX_inv : ndarray
            Fixed effects covariance matrix.
        u : ndarray
            Random effect estimate at theta.
        G : csc_matrix
            Random effects covariance matrix.
        R : dia_matrix
            Matrix of residual covariance.
        V : csc_matrix
            Model covariance matrix given fixed effects.

        """
        s = 1.0 if self.fixed_resid_cov else theta[-1]
        Rinv = self.weights_inv.dot(self.R / s).dot(self.weights_inv)
        theta = self.theta if theta is None else theta
        Ginv = self.update_gmat(theta, inverse=True)
        M = self.update_mme(Ginv, Rinv)
        XZy = self.XZ.T.dot(Rinv.dot(self.y))
        chol_fac = cholesky(M[:-1, :-1].tocsc())
        betau = chol_fac.solve_A(XZy)
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
        """

        Parameters
        ----------
        use_grad : bool, optional
            If true, the analytic gradient is used during optimization.
            The default is True.
        use_hess : bool, optional
            If true, the analytic hessian is used during optimization.
            The default is False.
        approx_hess: bool, optional
            If true, uses the gradient to approximate the hessian
        opt_kws : dict, optional
            Dictionary of options to use in scipy.optimize.minimize.
            The default is {}.

        Returns
        -------
        None.

        """
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
                hess = None
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
        """

        Parameters
        ----------
        use_grad : bool, optional
            If true and analytic_se is False, the gradient is used in the
            numerical approximation of the hessian. The default is True.
        analytic_se : bool, optional
            If true, then the hessian is used to compute standard errors.
            The default is False.

        Returns
        -------
        None.

        """
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
        """
        Parameters
        ----------
        X : ndarray, optional
            Model matrix for fixed effects. The default is None.
        Z : ndarray, optional
            Model matrix from random effects. The default is None.

        Returns
        -------
        yhat : ndarray
            Model predictions evaluated at X and Z.

        """
        if X is None:
            X = self.X
        if Z is None:
            Z = self.Z
        yhat = X.dot(self.beta)+Z.dot(self.u)
        return yhat
    
    def fit(self, reml=True, use_grad=True, use_hess=False, approx_hess=False,
            analytic_se=False, opt_kws={}):
        """
        

        Parameters
        ----------
        use_grad : bool, optional
            If true, the analytic gradient is used during optimization.
            The default is True.
        use_hess : bool, optional
            If true, the analytic hessian is used during optimization.
            The default is False.
        approx_hess: bool, optional
            If true, uses the gradient to approximate the hessian
        analytic_se : bool, optional
            If true, then the hessian is used to compute standard errors.
            The default is False.
        opt_kws : dict, optional
            Dictionary of options to use in scipy.optimize.minimize.
            The default is {}.

        Returns
        -------
        None.

        """
        theta, theta_chol, optimizer = self._optimize(reml, use_grad, use_hess, 
                                                      approx_hess, opt_kws)
        self._post_fit(theta, theta_chol, optimizer, reml, use_grad, 
                       analytic_se)
        param_names = list(self.fe_vars)
        for level in self.levels:
            for i, j in list(zip(*np.triu_indices(self.dims[level]['n_vars']))):
                param_names.append(f"{level}:G[{i}][{j}]")
        param_names.append("resid_cov")
        self.param_names = param_names
        res = np.vstack((self.params, self.se_params)).T
        res = pd.DataFrame(res, index=param_names, columns=['estimate', 'SE'])
        res['t'] = res['estimate'] / res['SE']
        res['p'] = sp.stats.t(self.X.shape[0]-self.X.shape[1]).sf(np.abs(res['t']))
        self.res = res
    
    
    def _restricted_ll_grad(self, theta_chol_f, free_ix, theta_chol_r, reml=True):
        theta_chol_r[free_ix] = theta_chol_f
        ll = self.loglike_c(theta_chol_r.copy(), reml)
        g = self.gradient_chol(theta_chol_r.copy(), reml)[free_ix]
        return ll, g
    
    def profile(self, n_points=40, par_ind=None, reml=True):
        par_ind = np.ones_like(self.theta_chol) if par_ind is None else par_ind
        theta_chol = self.theta_chol.copy()
        n_theta = len(theta_chol)
        
 
        llmax = self.loglike(self.theta.copy())
        free_ix = np.ones_like(theta_chol, dtype=bool)
        
        Hchol = so_gc_cd(self.gradient_chol, theta_chol, args=(reml,))
        se_chol = np.diag(np.linalg.inv(Hchol/2.0))**0.5
        thetas, zetas = np.zeros((n_theta*n_points, n_theta)), np.zeros(n_theta*n_points)
        k = 0
        pbar = tqdm.tqdm(total=n_theta*n_points, smoothing=0.001)
        for i in range(n_theta):
            free_ix[i] = False
            t_mle = theta_chol[i]
            theta_chol_r = theta_chol.copy()
            if self.bounds[i][0]==0:
                lb = np.maximum(0.01, t_mle-4.5*se_chol[i])
            else:
                lb = t_mle - 4.5 * se_chol[i]
            ub = t_mle + 4.5 * se_chol[i]
            tspace = np.linspace(lb, ub, n_points)
            for t0 in tspace:
                theta_chol_r = theta_chol.copy()
                theta_chol_r[~free_ix] = t0
                theta_chol_f = theta_chol[free_ix]
                func = lambda x : self._restricted_ll_grad(x, free_ix, theta_chol_r,
                                                           reml)
                bounds = np.array(self.bounds)[free_ix].tolist()
                opt = sp.optimize.minimize(func, theta_chol_f, jac=True,
                                           bounds=bounds,
                                           method='trust-constr')
                theta_chol_f = opt.x
                theta_chol_r[free_ix] = theta_chol_f
                LR = 2.0 * (opt.fun - llmax)
                zeta = np.sqrt(LR) * np.sign(t0 - theta_chol[~free_ix])
                zetas[k] = zeta
                thetas[k] = theta_chol_r
                k+=1
                pbar.update(1)
            free_ix[i] = True
        pbar.close()
        ix = np.repeat(np.arange(n_theta), n_points)
        return thetas, zetas, ix
    
    def plot_profile(self, n_points=40, par_ind=None, reml=True, quantiles=None):
        if quantiles is None:
            quantiles = [0.001, 0.05, 1, 5, 10, 20, 50, 80, 90, 95, 99, 99.5, 99.999]   
        thetas, zetas, ix = self.profile(n_points, par_ind, reml)
        n_thetas = thetas.shape[1]
        q = sp.stats.norm(0, 1).ppf(np.array(quantiles)/100)
        fig, axes = plt.subplots(figsize=(14, 4), ncols=n_thetas, sharey=True)
        plt.subplots_adjust(wspace=0.05, left=0.05, right=0.95)
        for i in range(n_thetas):
            ax = axes[i]
            x = thetas[ix==i, i]
            y = zetas[ix==i]
            trunc = (y>-5)&(y<5)
            x, y = x[trunc], y[trunc]
            f_interp = sp.interpolate.interp1d(y, x, fill_value="extrapolate")
            xq = f_interp(q)
            ax.plot(x,y)
            ax.set_xlim(x.min(), x.max())
            ax.axhline(0, color='k')
            for a, b in list(zip(xq, q)):
                ax.plot((a, a), (0, b), color='k')
        ax.set_ylim(-5, 5)
        return thetas, zetas, ix, fig, ax


    
class GLMM(WLMM):
    '''
    Currently an ineffecient implementation of a GLMM, mostly done 
    for fun.  A variety of implementations for GLMMs have been proposed in the
    literature, and a variety of names have been used to refer to each model;
    the implementation here is based of off linearization using a taylor
    approximation of the error (assumed to be gaussian) around the current
    estimates of fixed and random effects.  This type of approach may be 
    referred to as penalized quasi-likelihood, or pseudo-likelihood, and 
    may be abbreviated PQL, REPL, RPL, or RQL.

    '''
    def __init__(self, formula, data, weights=None, fam=None):
        super().__init__(formula=formula, data=data, weights=weights)
        if isinstance(fam, ExponentialFamily) == False:
            fam = fam()
            
        self.f = fam
        self.theta_init = self.theta.copy()
        self.y_original = self.y.copy()
        self.non_continuous = [isinstance(self.f, Binomial),
                               isinstance(self.f, NegativeBinomial),
                               isinstance(self.f, Poisson)]
        if np.any(self.non_continuous):
            self.bounds = self.bounds[:-1]+[(0, 0)]
            self.fix_resid_cov=True
        self.theta, self.theta_chol, self.optimizer = self._optimize()
        self.beta, _, self.u = self._compute_effects(self.theta)
        if isinstance(self.f, Binomial):
            self.u /= np.linalg.norm(self.u)
        self._nfixed_params = self.X.shape[1]
        self._n_obs = self.X.shape[0]
        self._n_cov_params = len(self.bounds)
        self._df1 = self._n_obs - self._nfixed_params
        self._df2 = self._n_obs - self._nfixed_params - self._n_cov_params - 1
        self._ll_const = self._df1 / 2 * np.log(2*np.pi)
        
    
    def _update_model(self, W, nu):
        nu = _check_shape(nu, 2)
        self.weights = sps.csc_matrix(W)
        self.weights_inv = sps.csc_matrix(np.diag(1.0/np.diag((W))))
        self.y = nu
        self.Xty = self.X.T.dot(nu)
        self.Zty = self.Z.T.dot(nu)
        self.theta = self.theta_init
        self.yty = nu.T.dot(nu)
        
        
    
    def _get_pseudovar(self):
        eta = self.predict()
        mu = self.f.inv_link(eta)
        var_mu = _check_shape(self.f.var_func(mu=mu), 1)
        gp = self.f.dlink(mu)
        nu = eta + gp * (_check_shape(self.y_original, 1) - mu)
        W = np.diag(np.sqrt(var_mu * (self.f.dlink(mu)**2)))
        return W, nu

    def fit(self, n_iters=200, tol=1e-3, optimizer_kwargs={}, verbose_outer=True):
        theta, theta_chol, optimizer = self.theta, self.theta_chol, self.optimizer
        fit_hist = {}
        for i in range(n_iters):
            W, nu = self._get_pseudovar()
            self._update_model(W, nu)
            theta_new, theta_chol_new, optimizer_new = self._optimize(**optimizer_kwargs)
            tvar = (np.linalg.norm(theta)+np.linalg.norm(theta_new))
            eps = np.linalg.norm(theta - theta_new) / tvar
            fit_hist[i] = dict(param_change=eps, theta=theta_new, nu=nu)
            if verbose_outer:
                print(eps)
            if eps < tol:
                break
            theta, theta_chol, optimizer = theta_new, theta_chol_new, optimizer_new
            self.beta, _, self.u = self._compute_effects(theta)
        self._post_fit(theta, theta_chol, optimizer)
        self.res = get_param_table(self.params, self.se_params, 
                                   self.X.shape[0]-len(self.params))
        
        
        eta_fe = self.X.dot(self.beta)
        eta = self.X.dot(self.beta)+self.Z.dot(self.u)
        mu = self.f.inv_link(eta)
        gp = self.f.dlink(mu)
        var_mu  =  _check_shape(self.f.var_func(mu=mu), 1)
        r_eta_fe = _check_shape(self.y, 1) - eta_fe

        generalized_chi2 = self.vinvcrossprod(r_eta_fe, theta)
        resids_raw_linear = _check_shape(self.y, 1) - eta
        resids_raw_mean = _check_shape(self.y_original, 1) - mu
        
        s = 1.0 if self.fixed_resid_cov else theta[-1]
        
        R = self.weights.dot(self.R * s).dot(self.weights)
        var_pearson_linear = R.diagonal() / gp**2
        var_pearson_mean = var_mu
        
        resids_pearson_linear = resids_raw_linear / np.sqrt(var_pearson_linear)
        resids_pearson_mean = resids_raw_mean / np.sqrt(var_pearson_mean)
        
        pll = self.loglike(self.theta) / -2.0 - self._ll_const
        aicc = -2 * pll + 2 * self._n_cov_params  * self._df1 / self._df2
        bic = -2 * pll + self._n_cov_params * np.log(self._df1)
        self.sumstats = dict(generalized_chi2=generalized_chi2,
                             pseudo_loglike=pll,
                             AICC=aicc,
                             BIC=bic)
        self.resids = dict(resids_raw_linear=resids_raw_linear,
                           resids_raw_mean=resids_raw_mean,
                           resids_pearson_linear=resids_pearson_linear,
                           resids_pearson_mean=resids_pearson_mean)
        param_names = list(self.fe_vars)
        for level in self.levels:
            for i, j in list(zip(*np.triu_indices(self.dims[level]['n_vars']))):
                param_names.append(f"{level}:G[{i}][{j}]")
        param_names.append("resid_cov")
        self.param_names = param_names
        self.res.index = param_names
 
"""       
from pystats.utilities.random_corr import vine_corr
from pystats.tests.test_data import generate_data
from pylmm.pylmm.lmm import LME
from pylmm.pylmm.glmm import WLME, GLMM


from pystats.utilities import numerical_derivs


    
 

np.set_printoptions(precision=3, suppress=True, linewidth=200)
formula = "y~1+x1+x2+(1+x3|id1)+(1+x4|id2)"
model_dict = {}
model_dict['gcov'] = {'id1':invech(np.array([2., 0.4, 2.])),
                      'id2':invech(np.array([2.,-0.4, 2.]))}

model_dict['ginfo'] = {'id1':dict(n_grp=200, n_per=10),
                       'id2':dict(n_grp=400, n_per=5)}
 
model_dict['mu'] = np.zeros(4)
model_dict['vcov'] = vine_corr(4, 20)
model_dict['beta'] = np.array([1, -1, 1])
model_dict['n_obs'] = 2000
data, formula = generate_data(formula, model_dict, r=0.6**0.5)




model_original = LME(formula, data)
model_cholesky = LME3(formula, data)
model_original._fit()
model_cholesky._fit(opt_kws=dict(verbose=3))
model_cholesky._post_fit()

model_original.se_params
model_cholesky.se_params




"""

