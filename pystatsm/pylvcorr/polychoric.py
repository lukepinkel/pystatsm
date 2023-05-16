#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 13 18:04:42 2023

@author: lukepinkel
"""
import pandas as pd
import numpy as np
import scipy as sp
import scipy.stats
from ..utilities import func_utils
from ..utilities import param_transforms
from ..utilities.indexing_utils import (inv_tril_indices, generate_indices)
from ..utilities.func_utils import (dbinorm_cdf_du, d2binorm_cdf_du2,
                                    d2binorm_cdf_duv, d2binorm_cdf_dur)





class Polychor(object):
    
    def __init__(self, data):
        self.data = data
        self.X = data.values.astype(int)
        self.X = self.X - np.min(self.X, axis=0)
        self.n, self.p = self.X.shape
        self.N_cats = np.array([len(np.unique(self.X[:, i])) for i in range(self.p)]).astype(int)
        self.transform_list = [param_transforms.OrderedTransform,
                               param_transforms.OrderedTransform,
                               param_transforms.TanhTransform]
        self.tanh_transform = param_transforms.TanhTransform()
        self.initialize_vars()
        self.process_data()
        self.initialize_interactions()

    def initialize_vars(self):
        self.uniq = {}
        self.catt = {}
        self.cprp = {}
        self.tauc = {}
        self.tauo = {}
        self.ncat = {}
    
    def process_data(self):
        self.n, self.p = self.X.shape
        self.uniq = {}
        self.catt = {}
        self.cprp = {}
        self.tauc = {}
        self.tauo = {}
        self.ncat = {}
        for i in range(self.p):
            self.uniq[i], self.catt[i] = np.unique(self.X[:, i], return_counts=True)
            self.cprp[i] = (np.cumsum(self.catt[i])/np.sum(self.catt[i]))[:-1]
            self.tauc[i] = sp.special.ndtri(self.cprp[i])
            self.tauo[i] = np.r_[-1e10, self.tauc[i], 1e10]
        self.ncat = np.array([len(self.catt[i]) for i in range(self.p)])
    
    def initialize_interactions(self):
        p = self.p
        p2 = int(p * (p - 1) // 2)
        p3 = int(p2 * (p2 + 1) // 2)
        inds1 = generate_indices((p,)*2, first_indices_change_fastest=False, 
                                 ascending=False, strict=True)
        inds2 = generate_indices((p2,)*2, first_indices_change_fastest=False,
                                 ascending=False, strict=False)
        inds3 = []
        for i in range(p3):
            inds3.append(inds1[inds2[i][0]]+inds1[inds2[i][1]])
        self.inds1, self.inds2, self.inds3 = inds1, inds2, inds3
        counts = {}
        props = {}
        indices = {}
        params = {}
        trns = {}
        for i, (i1, i2) in enumerate(self.inds1):
            counts[i] , props[i], indices[i], params[i], trns[i] = self._xtabs(i1, i2)
        self.counts = counts
        self.props = props
        self.indices = indices
        self.params = params
        self.trns = trns
        self.II_ind = inv_tril_indices(p, -1)
        self.p2 = p2
        self.p3 = p3
        
    def _xtabs(self, i1, i2):
        _, counts = sp.stats.contingency.crosstab(self.X[:, i1], self.X[:, i2])
        ni1, ni2 = counts.shape
        mi1, mi2 = ni1 - 1, ni2 - 1
        n_obs = np.sum(counts)
        params = np.r_[self.tauc[i1], self.tauc[i2],
                       np.corrcoef(self.X[:, i1], self.X[:, i2])[0, 1]]
        props = counts / n_obs
        
        ind_ni1_ni2 = np.meshgrid(np.arange(ni1), np.arange(ni2), indexing='ij')
        ind_mi1_ni2 = np.meshgrid(np.arange(mi1), np.arange(ni2), indexing='ij')
        ind_mi2_ni1 = np.meshgrid(np.arange(mi2), np.arange(ni1), indexing='ij')
        ind_mi1_mi2 = np.meshgrid(np.arange(mi1), np.arange(mi2), indexing='ij')
        
        indices = {"prob":ind_ni1_ni2,
                    "taui1":ind_mi1_ni2,
                    "taui2":ind_mi2_ni1,
                    "tau_i1i2":ind_mi1_mi2}
        transform = param_transforms.CombinedTransform(self.transform_list,
                                                       [mi1, mi2, 1])
        return counts, props, indices, params, transform

    def unpack_params_full(self, params, i1, i2, padval=1e12):
        ti1 = np.r_[-padval, params[:self.ncat[i1]-1], padval]
        ti2 = np.r_[-padval, params[self.ncat[i1]-1:self.ncat[i1]-1+self.ncat[i2]-1], 1e10]
        ri1i2 = params[-1]
        return ti1, ti2, ri1i2
    
    def predicted_probs(self, i1, i2, params=None, r=None):
        if params is not None:
           ti1, ti2, r = self.unpack_params_full(params, i1, i2)
        else:
            ti1, ti2 = self.tauo[i1], self.tauo[i2]
        vech_ind = self.II_ind[i1, i2]
        i, j = self.indices[vech_ind]["prob"]
        p = func_utils.binorm_cdf_region((ti1[i], ti2[j]), (ti1[i+1], ti2[j+1]), r)
        return p
    
    def loglike(self, params, i1, i2):
        ti1, ti2, r = self.unpack_params_full(params, i1, i2)
        vech_ind = self.II_ind[i1, i2]
        counts = self.counts[vech_ind]
        i, j = self.indices[vech_ind]["prob"]
        p = func_utils.binorm_cdf_region((ti1[i], ti2[j]), (ti1[i+1], ti2[j+1]), r)
        p = np.maximum(p, 1e-16)
        ll = -np.sum(counts * np.log(p))
        return ll
    
    def qloglike(self, r, i1, i2):
        ti1, ti2 = self.tauo[i1], self.tauo[i2]
        vech_ind = self.II_ind[i1, i2]
        counts = self.counts[vech_ind]
        i, j = self.indices[vech_ind]["prob"]
        p = func_utils.binorm_cdf_region((ti1[i], ti2[j]), (ti1[i+1], ti2[j+1]), r)
        p = np.maximum(p, 1e-16)
        ll = -np.sum(counts * np.log(p))
        return ll
    
    def dprob_dparams(self, params, i1, i2, order=1):
        ti1, ti2, r = self.unpack_params_full(params, i1, i2)
        ni1, ni2 = self.ncat[i1], self.ncat[i2]
        mi1, mi2 = ni1 - 1, ni2 - 1
        vech_ind = self.II_ind[i1, i2]
        indices = self.indices[vech_ind]
        dp_dt1 = np.zeros((ni1, ni2, mi1))
        dp_dt2 = np.zeros((ni1, ni2, mi2))
        
        i, j = indices["prob"]
        p = func_utils.binorm_cdf_region((ti1[i], ti2[j]), (ti1[i+1], ti2[j+1]), r)
        dp_dr = func_utils.binorm_pdf_region((ti1[i], ti2[j]), (ti1[i+1], ti2[j+1]), r)
        
        k, j = indices["taui1"]
        t1, t2, t3 = ti1[k+1], ti2[j], ti2[j+1]
        tmp1 = dbinorm_cdf_du(t1, t3, r) - dbinorm_cdf_du(t1, t2, r)
        dp_dt1[k, j, k] =  tmp1
        dp_dt1[k+1, j, k] = -tmp1
        k, i = indices["taui2"]
    
        t1, t2, t3 = ti2[k+1], ti1[i], ti1[i+1]
        
        tmp1 = dbinorm_cdf_du(t1, t3, r) - dbinorm_cdf_du(t1, t2, r)
        dp_dt2[i, k, k] =  tmp1
        dp_dt2[i, k+1, k] = -tmp1
        
        dp_dparams = np.concatenate([dp_dt1, dp_dt2, dp_dr[:, :, None]], axis=-1)
        
        if order < 2:
            return p, dp_dparams

        d2p_dr_dr = np.zeros((ni1, ni2, 1, 1))
        d2p_dt1_dt1 =  np.zeros((ni1, ni2, mi1, mi1))
        d2p_dt2_dt2 =  np.zeros((ni1, ni2, mi2, mi2))
        d2p_dt1_dt2 =  np.zeros((ni1, ni2, mi1, mi2))
        d2p_dt1_dr  =  np.zeros((ni1, ni2, mi1, 1))
        d2p_dt2_dr  =  np.zeros((ni1, ni2, mi2, 1))
        
        i, j = indices["prob"]
        d2p_dr_dr[i, j, 0, 0] = func_utils.dbinorm_pdf_region((ti1[i], ti2[j]), (ti1[i+1], ti2[j+1]), r)
        
        k, j =  indices["taui1"]
        t1, t2, t3 = ti1[k+1], ti2[j], ti2[j+1]
        tmp1 = d2binorm_cdf_du2(t1, t3, r) - d2binorm_cdf_du2(t1, t2, r)
        tmp2 = d2binorm_cdf_dur(t1, t3, r) - d2binorm_cdf_dur(t1, t2, r)
        d2p_dt1_dt1[k, j, k, k] = tmp1
        d2p_dt1_dt1[k+1, j, k, k] = -tmp1
        d2p_dt1_dr[k, j, k, 0] = tmp2
        d2p_dt1_dr[k+1, j, k, 0] = -tmp2
        
        k, i = indices["taui2"]
        t1, t2, t3 = ti2[k+1], ti1[i], ti1[i+1]
        tmp1 = d2binorm_cdf_du2(t1, t3, r) - d2binorm_cdf_du2(t1, t2, r)
        tmp2 = d2binorm_cdf_dur(t1, t3, r) - d2binorm_cdf_dur(t1, t2, r)
        d2p_dt2_dt2[i, k, k, k]   = tmp1
        d2p_dt2_dt2[i, k+1, k, k] = -tmp1
        d2p_dt2_dr[i, k, k, 0]    = tmp2
        d2p_dt2_dr[i, k+1, k, 0]  = -tmp2

        k, j = indices["tau_i1i2"]
        t1, t2 = ti1[k+1], ti2[j+1]
        tmp1 = d2binorm_cdf_duv(t1, t2, r) 
        d2p_dt1_dt2[k, j, k, j] = d2p_dt1_dt2[k+1, j+1, k, j] = tmp1
        d2p_dt1_dt2[k+1, j, k, j] = d2p_dt1_dt2[k, j+1, k, j] = -tmp1
        
        d2p_dt2_dt1 = np.swapaxes(d2p_dt1_dt2, 2, 3)
        d2p_dr_dt1 = np.swapaxes(d2p_dt1_dr, 2, 3)
        d2p_dr_dt2 = np.swapaxes(d2p_dt2_dr, 2, 3)
        d2p_dparams2 = np.block([[d2p_dt1_dt1, d2p_dt1_dt2,  d2p_dt1_dr],
                        [d2p_dt2_dt1, d2p_dt2_dt2,  d2p_dt2_dr],
                        [d2p_dr_dt1,  d2p_dr_dt2,   d2p_dr_dr]])
        
        return p, dp_dparams, d2p_dparams2
    
    def dprob_dr(self, r, i1, i2, order=1):
        ti1, ti2 = self.tauo[i1], self.tauo[i2]
        vech_ind = self.II_ind[i1, i2]
        i, j = self.indices[vech_ind]["prob"]
        t1, t2, t3, t4 = ti1[i], ti2[j], ti1[i+1], ti2[j+1]
        p = func_utils.binorm_cdf_region((t1, t2), (t3, t4), r)
        p = np.maximum(p, 1e-16)
        dp_dr = func_utils.binorm_pdf_region((t1, t2), (t3, t4), r)
        if order<2:
            return p, dp_dr
        d2p_dr_dr = func_utils.dbinorm_pdf_region((t1, t2), (t3, t4), r)
        return p, dp_dr, d2p_dr_dr
    
    
    def gradient(self, params, i1, i2):
        prob, dprob = self.dprob_dparams(params, i1, i2, order=1)
        prob = np.maximum(prob, 1e-16)
        counts = self.counts[self.II_ind[i1, i2]]
        g = -np.einsum("ij,ijl->l", counts / prob, dprob)
        return g
    
    
    def qgradient(self, r, i1, i2):
        prob, dprob = self.dprob_dr(r, i1, i2, order=1)
        prob = np.maximum(prob, 1e-16)
        counts = self.counts[self.II_ind[i1, i2]]
        g = -np.einsum("ij,ij->", counts / prob, dprob)
        return g

    def qhessian(self, r, i1, i2):
        prob, dp, dp2 = self.dprob_dr(r, i1, i2, order=2)
        prob = np.maximum(prob, 1e-16)
        counts = self.counts[self.II_ind[i1, i2]]
        u = counts / prob
        v = counts / np.maximum(prob**2, 1e-16)
        H = np.einsum("ij,ij,ij->", dp, v, dp) - np.einsum("ij,ij->", u, dp2)
        return H
    
    def hessian(self, params, i1, i2):
        prob, dp, dp2 = self.dprob_dparams(params, i1, i2, order=2)
        prob = np.maximum(prob, 1e-16)
        counts = self.counts[self.II_ind[i1, i2]]
        u = counts / prob
        v = counts / np.maximum(prob**2, 1e-16)
        H = np.einsum("ijl,ij,ijk->lk",dp, v, dp) - np.einsum("ij,ijkl->kl", u, dp2)
        return H
    
    
    def loglike_transformed(self, params, i1, i2):
        vech_ind = self.II_ind[i1, i2]
        x = self.trns[vech_ind].rvs(params.copy())
        ll = self.loglike(x, i1, i2)
        return ll
    
    def gradient_transformed(self, params, i1, i2):
        vech_ind = self.II_ind[i1, i2]
        x = self.trns[vech_ind].rvs(params.copy())
        J = self.trns[vech_ind].jac_rvs(params.copy())
        g = self.gradient(x, i1, i2)
        g = np.dot(g, J)
        return g
    
    def hessian_transformed(self, params, i1, i2):
        vech_ind = self.II_ind[i1, i2]
        x = self.trns[vech_ind].rvs(params)
        J = self.trns[vech_ind].jac_rvs(params)
        D = self.trns[vech_ind].hess_rvs(params)
        g = self.gradient(x, i1, i2)
        H = self.hessian(x, i1, i2)    
        H = J.T.dot(H).dot(J) + np.einsum("ijk,i->jk", D, g)
        return H
    
    def qloglike_transformed(self, r, i1, i2):
        x = self.tanh_transform.rvs(r)
        ll = np.atleast_1d(self.qloglike(x, i1, i2))
        return ll

    def qgradient_transformed(self, r, i1, i2):
        x = self.tanh_transform.rvs(r)
        J = self.tanh_transform.jac_rvs(r)
        g = self.qgradient(x, i1, i2)
        g = np.reshape(np.dot(g, J), -1)
        return g


    def qhessian_transformed(self, r, i1, i2):
        x = self.tanh_transform.rvs(r)
        J = self.tanh_transform.jac_rvs(r)
        D = self.tanh_transform.hess_rvs(r)
        g = self.qgradient(x, i1, i2)
        H = self.qhessian(x, i1, i2)    
        H = np.atleast_2d(H*J**2 + D*g)
        return H
    
    def _fit_fml(self, i1, i2, opt_kws=None):
        default_opt_kws = dict(method="trust-constr")
        opt_kws = func_utils.handle_default_kws(opt_kws, default_opt_kws)
        func = lambda x: self.loglike_transformed(x, i1, i2)
        grad = lambda x: self.gradient_transformed(x, i1, i2)
        hess = lambda x: self.hessian_transformed(x, i1, i2)
        vech_ind = self.II_ind[i1, i2]
        y = self.trns[vech_ind].fwd(self.params[vech_ind])
        opt_res = sp.optimize.minimize(func, y, jac=grad, hess=hess, **opt_kws)
        param = self.trns[vech_ind].rvs(opt_res.x)
        return opt_res, param
    
    def _fit_qml(self, i1, i2, opt_kws=None):
        default_opt_kws = dict(method="trust-constr")
        opt_kws = func_utils.handle_default_kws(opt_kws, default_opt_kws)
        func = lambda x: self.qloglike_transformed(x, i1, i2)
        grad = lambda x: self.qgradient_transformed(x, i1, i2)
        hess = lambda x: self.qhessian_transformed(x, i1, i2)
        vech_ind = self.II_ind[i1, i2]
        y = self.tanh_transform.fwd(self.params[vech_ind][-1])
        opt_res = sp.optimize.minimize(func, y, jac=grad, hess=hess, **opt_kws)
        param =self.tanh_transform.rvs(opt_res.x)
        return opt_res, param

    
    def _fit(self, i1, i2, method="twostep", opt_kws=None):
        i = self.II_ind[i1, i2]
        if method.lower() in ["two-step", "twostep", "two step", "qml"]:
            self.opt_res[i], self.params[i][-1] = self._fit_qml(i1, i2, opt_kws)
        elif method.lower() in ["one-step", "onestep", "one step", "ml", "fml"]:
            self.opt_res[i], self.params[i] = self._fit_fml(i1, i2, opt_kws)
    
    def _get_acm_comp(self, n, xi, xj, xk, xl, gij, gkl, wij, wkl):
        v = np.sum(gij[xi, xj] * gkl[xk, xl]) / n
        v = v - wij * wkl
        return v

    def get_acov(self):
        arr = self.X
        A_mats, B_mats, G_mats, W = {}, {}, {}, {}
        
        for i in range(self.p):
            ni = self.ncat[i] 
            ti = self.tauo[i]
            rind, cind = np.diag_indices(ni-1)
            Ai = np.zeros((ni, ni-1))
            phi = func_utils.norm_pdf(ti[1:-1])
            Ai[(rind, cind)] = phi
            Ai[(rind+1, cind)] = -phi
            p = func_utils.norm_cdf(ti[1:])-func_utils.norm_cdf(ti[:-1]) 
            A_mats[i] = Ai
            ADAi = np.dot(Ai.T * 1/p, Ai) 
            Bi = np.linalg.solve(ADAi, Ai.T * 1 / p)
            B_mats[i] = Bi
        
        for counter, (i1, i2) in enumerate(self.inds1):
            params = self.params[counter]
            counts = self.counts[counter]
            prob, dprob1, dprob2 = self.dprob_dparams(params, i1, i2, order=2)
            ni1, ni2 = self.ncat[[i1, i2]]-1
            mi1, mi2 = self.ncat[[i1, i2]]-1
            dp_dti1 = dprob1[:, :, :mi1]
            dp_dti2 = dprob1[:, :, mi1:mi1+mi2]
            dp_dr = dprob1[:, :, -1]
            bi1 = -np.einsum("ij,ijk->k", dp_dr / prob, dp_dti1)
            bi2 = -np.einsum("ij,ijk->k", dp_dr / prob, dp_dti2)
            Bi1, Bi2 = B_mats[i1], B_mats[i2]
            alpha= dp_dr / prob
            D = (np.sum((dp_dr**2) / prob))
            G = alpha + np.dot(bi1, Bi1)[:, None] + np.dot(bi2, Bi2)[None]
            G = G / D
            W[counter] = np.einsum("ij,ij->", G, counts)
            G_mats[counter] = G

        Acov = np.zeros((self.p2,self.p2))
        for counter, (a, b, c, d) in enumerate(self.inds3):
            ab, cd = self.inds2[counter]
            Acov[ab, cd] = self._get_acm_comp(self.n, arr[:, a], arr[:, b],
                                              arr[:, c], arr[:, d],
                                              G_mats[ab], G_mats[cd], 
                                              W[ab], W[cd])
            Acov[cd, ab] = Acov[ab, cd]
        return Acov
           
            
    def fit(self, method="twostep", opt_kws=None):
        self.opt_res = {}
        for ii in range(self.p2):
            self._fit(*self.inds1[ii], method=method, opt_kws=opt_kws)
        self.rhos = np.array([self.params[i][-1] for i in range(len(self.params))])
        self.acov = self.get_acov()

def threshold_dict_to_array(t_dict):
    p = len(t_dict)
    nt_extended = np.zeros(p, dtype=int)
    nmax = max([len(x) for x in t_dict.values()])
    t_arr = np.zeros((p, nmax))
    for i, (key, val) in enumerate(t_dict.items()):
        nt_extended[i] = len(val)
        t_arr[i, :nt_extended[i]] = val
    return t_arr



