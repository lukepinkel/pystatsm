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
from sksparse.cholmod import cholesky, cholesky_AAt


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
        theta.append(np.ones(1))
        t_inds.append(np.arange(t_offset, t_offset+1))
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
        self.n = X.shape[0]
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
        self.W = sps.diags([np.ones((self.n,))], [0])
        self.dR = sps.eye(self.n)
        self.jac_inds = random_effects.jac_inds
        self.theta = random_effects.theta
        self.n_par = len(self.theta)
        self.H = np.zeros((self.n_par, self.n_par))
        self.transforms = random_effects.transforms
        
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
    
    def pls(self, theta, w=None):
        W = self.W if w is None else sps.diags([w], [0])
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
    
    def gradient(self, theta, reml=True, use_sw=False):
        s = theta[-1]
        W = self.W / s
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
    
    def hessian(self, theta, reml=True, use_sw=False):
        s = theta[-1]
        R = self.W / s
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
    
    def loglike_grad(self, x):
        theta = np.zeros_like(x)
        theta[-1] = np.exp(x[-1])
        dtheta_dx = []
        for i in range(self.levels):
            ix = self.t_inds[i]
            transform = self.transforms[i]
            theta[ix] = transform._rvs(x[ix])
            dtheta_dx.append(transform._jac_rvs(x[ix]))
        dtheta_dx.append(np.atleast_2d(np.exp(x[-1])))
        df_dtheta = self.gradient(theta)
        dtheta_dx = sp.linalg.block_diag(*dtheta_dx)
        df_dx = df_dtheta.dot(dtheta_dx)
        f = self.loglike(theta)
        return f, df_dx
    
    def _fwd_transform(self, theta):
        x = np.zeros_like(theta)
        x[-1] = np.log(theta[-1])
        for i in range(self.levels):
            ix = self.t_inds[i]
            transform = self.transforms[i]
            x[ix] = transform._fwd(theta[ix])
        return x
    
    def _optimize(self):
        x = self._fwd_transform(self.theta)
        opt = sp.optimize.minimize(self.loglike_grad, x, jac=True, method="trust-constr",
                                   options=dict(verbose=3, gtol=1e-6, xtol=1e-6))
        self.opt=opt
    
        
    
    


        
        