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
from scipy.linalg.lapack import dtrtri
from .ranef_terms import RandomEffects, RandomEffectTerm
from ..utilities.linalg_operations import invech_chol, invech, vech
from ..utilities.numerical_derivs import fo_fc_cd, so_fc_cd, so_gc_cd
from ..utilities.formula import parse_random_effects

from sksparse.cholmod import cholesky, cholesky_AAt
import pandas as pd


def sparsity(sparse_arr):
    return sparse_arr.nnz / np.prod(sparse_arr.shape)

class LMM(object):
    
    def __init__(self, formula, data):
        model_info = parse_random_effects(formula)
        y_vars = model_info["y_vars"]
        fe_form = model_info["fe_form"]
        groups = model_info["re_terms"]
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
        self.n = X.shape[0]
        self.X, self.Z, self.y = X, Z, y
        self.random_effects = random_effects
        self.fe_vars, self.y_vars = fe_vars, y_vars
        self.fe_form, self.groups = fe_form, groups
        self.fe_vars = fe_vars
        self._model_info = model_info
        self.M = M.tocsc()
        self.C = C
        self.G = random_effects.G
        self.L = random_effects.L
        self.levels = random_effects.levels
        self.g_inds = random_effects.g_inds
        self.l_inds = random_effects.l_inds
        self.t_inds = random_effects.t_inds
        self.ng = random_effects.group_sizes
        self.zero_mat1, self.zero_mat2 = zero_mat1, zero_mat2
        self.ZtZ = ZtZ
        self.Zty = ytZ.T
        self.ZtX = ZtX
        self.Xty = ytX.T
        self.R = sps.diags([np.ones((self.n,))], [0])
        self.dR = sps.eye(self.n)
        self.jac_inds = random_effects.jac_inds
        self.theta = random_effects.theta
        self.n_par = random_effects.n_par
        self.H = random_effects.H
        self.elim_mats = random_effects.elim_mats
        self.symm_mats = random_effects.symm_mats
        self.iden_mats = random_effects.iden_mats
        self.d2g_dchol = random_effects.d2g_dchol
        self.n_fe = self.p = self.X.shape[1]
        self.bounds = [(0, None) if x==1 else (None, None) for x in self.theta[:-1]]+[(None, None)]
        
            
    def transform_theta(self, theta):
        for i in range(self.levels):
            G = invech(theta[self.t_inds[i]])
            L = np.linalg.cholesky(G)
            theta[self.t_inds[i]] = vech(L)
        theta[-1] = np.log(theta[-1])
        return theta
        
    def inverse_transform_theta(self, theta):
        for i in range(self.levels): 
            L = invech_chol(theta[self.t_inds[i]])
            G = L.dot(L.T)
            theta[self.t_inds[i]] = vech(G)
        theta[-1] = np.exp(theta[-1])
        return theta
    
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
    
    def _update_mme(self, Ginv, theta):
        M =  self.M.copy()/theta[-1]
        C = sp.sparse.block_diag([self.zero_mat1, Ginv, self.zero_mat2])
        M+=C
        return M
    
    def update_mme2(self, Ginv, theta):
        M =  self.M.copy()/theta[-1]
        M[self.n_fe:-1, self.n_fe:-1] +=Ginv
        return M
    
    def update_mme(self, Ginv, theta):
        M = self.M.copy()
        M.data /=theta[-1]
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
    
    def pls(self, theta, w=None):
        W = self.R if w is None else sps.diags([w], [0])
        L = self.update_lmat(theta)
        X, Z, y = self.X, self.Z, self.y
        ZtW, WX, Wy = Z.T.dot(W), W.dot(X), W.dot(y)
        XtWX = WX.T.dot(WX)
        XtWy = WX.T.dot(Wy)
        ZtWX = ZtW.dot(WX)
        ZtWy = ZtW.dot(Wy)
        LtZtW = (L.T.dot(ZtW)).tocsc()
        Lfactor = cholesky_AAt(LtZtW, beta=1)
        cu    = Lfactor.solve_L(L.T.dot(ZtWy), use_LDLt_decomposition=False)
        RZX   = Lfactor.solve_L(L.T.dot(ZtWX), use_LDLt_decomposition=False)
        RXtRX = XtWX - RZX.T.dot(RZX)
        b = np.linalg.solve(RXtRX, XtWy - RZX.T.dot(cu))
        v = Lfactor.solve_Lt(cu - RZX.dot(b), use_LDLt_decomposition=False)
        u = L.dot(v)
        mu = X.dot(b) + Z.dot(u)
        wtres = W.dot(y - mu)
        pwrss = np.sum(wtres**2) + np.sum(v**2)
        ld = Lfactor.slogdet()[1] + np.linalg.slogdet(RXtRX)[1]
        n, p = X.shape
        dev = self.n * np.log(2 * np.pi * theta[-1]) + ld + pwrss / theta[-1]        
        return dev, b, u
    
    def dg_dchol(self, L_dict):
        Jf = {}
        for i in range(self.levels):
            L = L_dict[i]
            E = self.elim_mats[i]
            N = self.symm_mats[i]
            I = self.iden_mats[i]
            Jf[i] = E.dot(N.dot(np.kron(L, I))).dot(E.T)
        return Jf
    
    def loglike_c(self, theta_chol, reml=True, use_sw=False):
        theta = self.inverse_transform_theta(theta_chol.copy())
        return self.loglike(theta, reml, use_sw)
    
    def gradient_c(self, theta_chol, reml=True, use_sw=False):
        theta = self.inverse_transform_theta(theta_chol.copy())
        return self.gradient(theta, reml, use_sw)
    
    
    def hessian_c(self, theta_chol, reml=True):
        theta = self.inverse_transform_theta(theta_chol.copy())
        return self.hessian(theta, reml)
    
    def gradient_chol(self, theta_chol, reml=True, use_sw=False):
        L_dict = self.update_chol(theta_chol)
        Jf_dict = self.dg_dchol(L_dict)
        Jg = self.gradient_c(theta_chol, reml, use_sw)
        Jf = sp.linalg.block_diag(*Jf_dict.values()) 
        Jf = np.pad(Jf, [[0, 1]])
        Jf[-1, -1] =  np.exp(theta_chol[-1])
        return Jg.dot(Jf)
    
    def hessian_chol(self, theta_chol, reml=True):
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
        
        for i in range(self.levels):
            ix = self.t_inds[i]
            Jg_i = Jg[ix]
            Hf_i = Hf[i]
            C = np.einsum('i,ijk->jk', Jg_i, Hf_i)  
            B[ix, ix[:, None]] += C
        B[-1, -1] = Jg[-1] * np.exp(theta_chol[-1])
        H = A + B
        return H
    
    def _gradient_old(self, theta, reml=True, use_sw=False):
        s = theta[-1]
        W = self.R / s
        Ginv = self.update_gmat(theta, inverse=True)
        X, Z, y = self.X, self.Z, self.y
        RZ, RX, Ry = W.dot(Z), W.dot(X), W.dot(y)
        ZtRZ, XtRX, ZtRX, ZtRy = RZ.T.dot(Z), X.T.dot(RX), RZ.T.dot(X), RZ.T.dot(y)
    
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
        Py = Vy - WX.dot(U.dot(y))
        ZtPy = Z.T.dot(Py)
        grad = []
        for i in range(self.levels):
            ind = self.jac_inds[i]
            ZtWZi = ZtWZ[ind][:, ind]
            ZtWXi = ZtWX[ind]
            ZtPyi = ZtPy[ind]
            for dGdi in self.random_effects.terms[i].G_deriv:
                g1 = dGdi.dot(ZtWZi).diagonal().sum() 
                g2 = ZtPyi.T.dot(dGdi.dot(ZtPyi))
                if reml:
                    g3 = np.trace(XtWX_inv.dot(ZtWXi.T.dot(dGdi.dot(ZtWXi))))
                else:
                    g3 = 0
                gi = g1 - g2 - g3
                grad.append(gi)

        for dR in [self.dR]:
            g1 = W.diagonal().sum() - (M.dot((RZ.T).dot(dR).dot(RZ))).diagonal().sum()
            g2 = Py.T.dot(Py)
            if reml:
                g3 = np.trace(XtWX_inv.dot(WX.T.dot(WX)))
            else:
                g3 = 0
            gi = g1 - g2 - g3
            grad.append(gi)
        grad = np.concatenate(grad)
        grad = np.asarray(grad).reshape(-1)
        return grad
    
    def _gradient_dense(self, theta, reml=True, use_ws=False):
        s = theta[-1]
        W = self.R / s
        Ginv = self.update_gmat(theta, inverse=True)
        X, Z, y = self.X, self.Z, self.y
        RZ, RX, Ry = W.dot(Z), W.dot(X), W.dot(y)
        ZtRZ, XtRX, ZtRX, ZtRy = RZ.T.dot(Z), X.T.dot(RX), RZ.T.dot(X), RZ.T.dot(y)
    
        Q = Ginv + ZtRZ
        Q_chol = sp.linalg.cholesky(Q.toarray(), lower=True)
        Q_cinv, _ = dtrtri(Q_chol, lower=1)
        M = Q_cinv.T.dot(Q_cinv)
        
        ZtRZ = ZtRZ.toarray() 
        ZtWZ = ZtRZ - ZtRZ.dot(M).dot(ZtRZ)
                    
        MZtRX = M.dot(ZtRX)
        
        XtWX = XtRX - ZtRX.T.dot(MZtRX)
        XtWX_inv = np.linalg.inv(XtWX)
        ZtWX = ZtRX - ZtRZ.dot(MZtRX)
        WX = RX - RZ.dot(MZtRX)
        U = XtWX_inv.dot(WX.T)
        Vy = Ry - RZ.dot(M.dot(ZtRy))
        Py = Vy - WX.dot(U.dot(y))
        ZtPy = Z.T.dot(Py)
        grad = []
        for i in range(self.levels):
            ind = self.jac_inds[i]
            ZtWZi = ZtWZ[ind][:, ind]
            ZtWXi = ZtWX[ind]
            ZtPyi = ZtPy[ind]
            for dGdi in self.random_effects.terms[i].G_deriv:
                g1 = dGdi.dot(ZtWZi).diagonal().sum() 
                g2 = ZtPyi.T.dot(dGdi.dot(ZtPyi))
                if reml:
                    g3 = np.trace(XtWX_inv.dot(ZtWXi.T.dot(dGdi.dot(ZtWXi))))
                else:
                    g3 = 0
                gi = g1 - g2 - g3
                grad.append(gi)

        for dR in [self.dR]:
            RZt_dR_RZ = RZ.T.dot(dR).dot(RZ)
            g1 = W.diagonal().sum() - (RZt_dR_RZ.T.dot(M.T)).diagonal().sum()
            g2 = Py.T.dot(Py)
            if reml:
                g3 = np.trace(XtWX_inv.dot(WX.T.dot(WX)))
            else:
                g3 = 0
            gi = g1 - g2 - g3
            grad.append(gi)
        grad = np.concatenate(grad)
        grad = np.asarray(grad).reshape(-1)
        return grad
    
    def gradient(self, theta, reml=True, use_sw=False, assume_sparse_inverse=True):
        s = theta[-1]
        W = self.R / s
        Ginv = self.update_gmat(theta, inverse=True)
        X, Z, y = self.X, self.Z, self.y
        RZ, RX, Ry = W.dot(Z), W.dot(X), W.dot(y)
        ZtRZ, XtRX, ZtRX, ZtRy = RZ.T.dot(Z), X.T.dot(RX), RZ.T.dot(X), RZ.T.dot(y)
        Q = Ginv + ZtRZ
        ztrz_sparse = sparsity(ZtRZ) < 0.05
        if not ztrz_sparse:
            ZtRZ = ZtRZ.toarray()
        
        if assume_sparse_inverse and ztrz_sparse:
            M = cholesky(Q).inv()
            m_sparse = sparsity(M) < 0.05
            if not m_sparse:
                M = M.toarray()
        else:
            Q_chol = sp.linalg.cholesky(Q.toarray(), lower=True, check_finite=False)
            Q_cinv, _ = dtrtri(Q_chol, lower=1)
            M = Q_cinv.T.dot(Q_cinv)
            m_sparse = False
        
        if m_sparse:
            if ztrz_sparse:
                ZtWZ = ZtRZ - ZtRZ.dot(M).dot(ZtRZ)
            else:
                ZtWZ = ZtRZ - ZtRZ.dot(M.dot(ZtRZ))
        else:
            if ztrz_sparse:
                ZtWZ = ZtRZ - (ZtRZ.T.dot((ZtRZ.dot(M)).T)).T
            else:                
                ZtWZ = ZtRZ - ZtRZ.dot(M).dot(ZtRZ)
                    
        MZtRX = M.dot(ZtRX)
        
        XtWX = XtRX - ZtRX.T.dot(MZtRX)
        XtWX_inv = np.linalg.inv(XtWX)
        ZtWX = ZtRX - ZtRZ.dot(MZtRX)
        WX = RX - RZ.dot(MZtRX)
        U = XtWX_inv.dot(WX.T)
        Vy = Ry - RZ.dot(M.dot(ZtRy))
        Py = Vy - WX.dot(U.dot(y))
        ZtPy = Z.T.dot(Py)
        grad = []
        for i in range(self.levels):
            ind = self.jac_inds[i]
            ZtWZi = ZtWZ[ind][:, ind]
            ZtWXi = ZtWX[ind]
            ZtPyi = ZtPy[ind]
            for dGdi in self.random_effects.terms[i].G_deriv:
                g1 = dGdi.dot(ZtWZi).diagonal().sum() 
                g2 = ZtPyi.T.dot(dGdi.dot(ZtPyi))
                if reml:
                    g3 = np.trace(XtWX_inv.dot(ZtWXi.T.dot(dGdi.dot(ZtWXi))))
                else:
                    g3 = 0
                gi = g1 - g2 - g3
                grad.append(gi)

        for dR in [self.dR]:
            RZt_dR_RZ = RZ.T.dot(dR).dot(RZ)
            if sparsity(RZt_dR_RZ) < 0.05:
                if m_sparse:
                    g1 = W.diagonal().sum() - (M.dot(RZt_dR_RZ)).diagonal().sum()
                else:
                    g1 = W.diagonal().sum() - (RZt_dR_RZ.T.dot(M.T)).diagonal().sum()
            else:
                RZt_dR_RZ = RZt_dR_RZ.toarray()
                g1 = W.diagonal().sum() - (M.dot(RZt_dR_RZ)).diagonal().sum()
            g2 = Py.T.dot(Py)
            if reml:
                g3 = np.trace(XtWX_inv.dot(WX.T.dot(WX)))
            else:
                g3 = 0
            gi = g1 - g2 - g3
            grad.append(gi)
        grad = np.concatenate(grad)
        grad = np.asarray(grad).reshape(-1)
        return grad
    
    def hessian(self, theta, reml=True, use_sw=False):
        s = theta[-1]
        R = self.R / s
        Ginv = self.update_gmat(theta, inverse=True)
        X, Z, y = self.X, self.Z, self.y
        RZ = R.dot(Z)
        ZtRZ = RZ.T.dot(Z)
    
        Q = ZtRZ + Ginv
        M = cholesky(Q).inv()
        W = R - RZ.dot(M).dot(RZ.T)
        WZ = W.dot(Z)
        WX = W.dot(X)
        XtWX = WX.T.dot(X)
        ZtWX = Z.T.dot(WX)
        U = np.linalg.solve(XtWX, WX.T)
        ZtP = WZ.T - ZtWX.dot(np.linalg.solve(XtWX, WX.T))
        ZtPZ = Z.T.dot(ZtP.T)
        Py = W.dot(y) - WX.dot(U.dot(y))
        ZtPy = Z.T.dot(Py)
        PPy = W.dot(Py) - WX.dot(U.dot(Py))
        ZtPPy =  Z.T.dot(PPy)
        H = self.H * 0.0
        PJ, yPZJ, ZPJ = [], [], []
        ix = []
        for i in range(self.levels):
            ind = self.jac_inds[i]
            ZtPZi = ZtPZ[ind]
            ZtPyi = ZtPy[ind]
            ZtPi = ZtP[ind]
            for dGdi in self.random_effects.terms[i].G_deriv:
                PJ.append(dGdi.dot(ZtPZi))
                yPZJ.append(dGdi.dot(ZtPyi))
                ZPJ.append((dGdi.dot(ZtPi)).T)
                ix.append(ind)
            
        t_indices = list(zip(*np.triu_indices(len(self.theta)-1)))
        for i, j in t_indices:
            ZtPZij = ZtPZ[ix[i]][:, ix[j]]
            PJi, PJj = PJ[i][:, ix[j]], PJ[j][:, ix[i]]
            yPZJi, JjZPy = yPZJ[i], yPZJ[j]
            Hij = -np.einsum('ij,ji->', PJi, PJj)\
                    + (2 * (yPZJi.T.dot(ZtPZij)).dot(JjZPy))[0]
            H[i, j] = H[j, i] = Hij
        dR = self.dR
        dRZtP = (dR.dot(ZtP.T))
        for i in range(len(self.theta)-1):
            yPZJi = yPZJ[i]
            ZPJi = ZPJ[i]
            ZtPPyi = ZtPPy[ix[i]]
            H[i, -1] = H[-1, i] = 2*yPZJi.T.dot(ZtPPyi) - np.einsum('ij,ji->', ZPJi.T, dRZtP[:, ix[i]])
        P = W - WX.dot(U)
        H[-1, -1] = Py.T.dot(PPy)*2 - np.einsum("ij,ji->", P, P)
        return H
    
    
    def vinvcrossprod(self, A, B, theta):
        Rinv = self.R / theta[-1]
        Ginv = self.update_gmat(theta, inverse=True)
        RZ = Rinv.dot(self.Z)
        Q = Ginv + self.Z.T.dot(RZ)
        M = cholesky(Q).inv()
        AtRB = ((Rinv.dot(B)).T.dot(A)).T 
        AtRZ = (RZ.T.dot(A)).T
        ZtRB = RZ.T.dot(B)
        AtVB = AtRB - (M.dot(ZtRB)).T.dot(AtRZ.T).T
        return AtVB
        
    def _compute_effects(self, theta=None):
        theta = self.theta if theta is None else theta
        Ginv = self.update_gmat(theta, inverse=True)
        M = self.update_mme(Ginv, theta)
        XZy = np.r_[self.Xty.A, self.Zty.A].flatten() / theta[-1]
        chol_fac = cholesky(M[:-1, :-1].tocsc())
        betau = chol_fac.solve_A(XZy)
        u = betau[self.X.shape[1]:].reshape(-1)
        beta = betau[:self.X.shape[1]].reshape(-1)
        
        Rinv = self.R / theta[-1]
        RZ = Rinv.dot(self.Z)
        Q = Ginv + self.Z.T.dot(RZ)
        M = cholesky(Q).inv()
        XtRinvX = self.X.T.dot(Rinv.dot(self.X)) 
        XtRinvZ = (RZ.T.dot(self.X)).T
        XtVinvX = XtRinvX - XtRinvZ.dot(M.dot(XtRinvZ.T))
        XtVinvX_inv = np.linalg.inv(XtVinvX)
        return beta, XtVinvX_inv, u
     
    def update_chol(self, theta, inverse=False):
        L_dict = {}
        for i in range(self.levels):
            theta_i = theta[self.t_inds[i]]
            L_i = invech_chol(theta_i)
            L_dict[i] = L_i
        return L_dict
    
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
        theta = self.inverse_transform_theta(theta_chol.copy())
        return theta, theta_chol, optimizer
        
    def _post_fit(self, theta, theta_chol, optimizer, reml=True,
                  use_grad=True, analytic_se=False):

        beta, XtWX_inv, u = self._compute_effects(theta)
        params = np.concatenate([beta, theta])
        re_covs, re_corrs = {}, {}
        for i in range(self.levels):
            re_covs[i] = invech(theta[self.t_inds[i]].copy())
            C = re_covs[i]
            v = np.diag(np.sqrt(1/np.diag(C)))
            re_corrs[i] = v.dot(C).dot(v)
        
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
            analytic_se=False, adjusted_pvals=True, opt_kws={}):
    
        theta, theta_chol, optimizer = self._optimize(reml, use_grad, use_hess, 
                                                      approx_hess, opt_kws)
        self._post_fit(theta, theta_chol, optimizer, reml, use_grad, 
                       analytic_se)
        param_names = list(self.fe_vars)
        for k in range(self.levels):
            group = self.random_effects.terms[k].gr_form
            p = self.random_effects.terms[k].n_rvars
            for i, j in list(zip(*np.triu_indices(p))):
                param_names.append(f"{group}:G[{i}][{j}]")
        param_names.append("resid_cov")
        self.param_names = param_names
        res = np.vstack((self.params, self.se_params)).T
        res = pd.DataFrame(res, index=param_names, columns=['estimate', 'SE'])
        res['t'] = res['estimate'] / res['SE']
        res['p'] = sp.stats.t(self.X.shape[0]-self.X.shape[1]).sf(np.abs(res['t']))
        res['degfree'] = self.X.shape[0] - self.X.shape[1]
        if adjusted_pvals:
            L = np.eye(self.X.shape[1])
            L_list = [L[[i]] for i in range(self.X.shape[1])]
            adj_table = pd.DataFrame(self.approx_degfree(L_list), index=self.fe_vars)
            res.loc[self.fe_vars, 't'] = adj_table['F']**0.5
            res.loc[self.fe_vars, 'degfree'] = adj_table['df2']
            res.loc[self.fe_vars, 'p'] = adj_table['p']
        self.res = res
        
    def approx_degfree(self, L_list=None, theta=None, beta=None, method='satterthwaite'):
        L_list = [np.eye(self.X.shape[1])] if L_list is None else L_list
        theta = self.theta if theta is None else theta
        beta = self.beta if beta is None else beta
        C = np.linalg.inv(self.vinvcrossprod(self.X, self.X, theta))
        Vtheta = np.linalg.inv(so_gc_cd(self.gradient, theta))
        J = []
        for i in range(self.levels):
            ind = self.jac_inds[i]
            XtVZ = self.vinvcrossprod(self.X, self.Z[:, ind], theta)
            CXtVZ = C.dot(XtVZ)
            for dGdi in self.random_effects.terms[i].G_deriv:
                dC = CXtVZ.dot(dGdi.dot(CXtVZ.T))
                J.append(dC)
        XtVi = self.vinvcrossprod(self.X, self.R.copy(), theta)
        CXtVi = C.dot(XtVi)
        J.append(CXtVi.dot(CXtVi.T))
        res = []
        for L in L_list:
            u, Q = np.linalg.eigh(L.dot(C).dot(L.T))
            order = np.argsort(u)[::-1]
            u, Q = u[order], Q[:, order]
            q = np.linalg.matrix_rank(L)
            P = Q.T.dot(L)
            t2 = (P.dot(beta))**2 / u
            f = np.sum(t2) / q
            D = []
            for i in range(q):
                x = P[i]
                D.append([np.dot(x, Ji).dot(x) for Ji in J])
            D = np.asarray(D)
            nu_d = np.array([D[i].T.dot(Vtheta).dot(D[i]) for  i in range(q)])
            nu_m = u**2 / nu_d
            E = np.sum(nu_m[nu_m>2] / (nu_m[nu_m>2] - 2.0))
            nu = 2.0 * E / (E - q)
            res.append(dict(F=f, df1=q, df2=nu, p=sp.stats.f(q, nu).sf(f)))
        return res
    


        
        
