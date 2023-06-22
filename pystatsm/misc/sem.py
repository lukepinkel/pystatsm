#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 07:59:44 2023

@author: lukepinkel
"""

import numpy as np
import scipy as sp
import pandas as pd

from ..utilities.linalg_operations import  _vech, _vec, _invec, _invech
from ..utilities.func_utils import handle_default_kws, triangular_number
from .derivatives import _dloglike_mu, _d2loglike_mu, _dsigma_mu, _d2sigma_mu #from .cov_derivatives import _d2sigma, _dsigma, _dloglike, _d2loglike
from .model_spec import ModelSpecification


pd.set_option("mode.chained_assignment", None)



        
class SEM(ModelSpecification):
    
    def __init__(self, formula, data, group_col=None, model_spec_kws=None, group_kws=None):
        default_model_spec_kws = dict(extension_kws=dict(fix_lv_var=False))
        model_spec_kws = handle_default_kws(model_spec_kws, default_model_spec_kws)
        group_kws = dict(shared=[True]*6) if group_kws is None else group_kws
        super().__init__(formula, data, group_col, **group_kws)
        bounds = self.free_df[["lb", "ub"]]
        bounds = bounds.values
        self.bounds = [tuple(x) for x in bounds.tolist()]
        self.bounds_theta = [tuple(x) for x in bounds[self._first_locs]]
        self.p_templates = self.p_templates
        self.group_col = group_col
        self.indexer = self.indexers[0]
        self.p2, self.q2 = triangular_number(self.p), triangular_number(self.q)
        self.nf = len(self.indexer.flat_indices)
        self.nf2 = triangular_number(self.nf)
        self.nt = len(self.indexer.first_locs)
        self.make_derivative_matrices()
        self.theta = self.transform_free_to_theta(self.free)
        self.n_par = len(self.p_templates[0])
        self.ll_const =  1.8378770664093453 * self.p
        self.gsizes = np.array(list(self.model_data.n_obs.values()))
        self.gweights = self.gsizes / np.sum(self.gsizes)
        self.n_obs = np.sum(self.gsizes)

    def make_derivative_matrices(self):
        self.indexer.create_derivative_arrays([(0, 0), (1, 0), (2, 0), (1 ,1), (2, 1), (5, 0), (5, 1)])
        self.dA = self.indexer.dA
        self.n_group_theta = len(self._first_locs)
        self.dSm = np.zeros((self.n_groups, self.p2+self.p, self.nf))
        self.d2Sm = np.zeros((self.n_groups, self.p2+self.p, self.nf, self.nf))
        self.m_size = self.indexer.block_sizes                    
        self.m_kind = self.indexer.block_indices                 
        self.d2_kind = self.indexer.block_pair_types             
        self.d2_inds = self.indexer.colex_descending_inds  
        self.J_theta = sp.sparse.csc_array(
            (np.ones(self.nf), (np.arange(self.nf), self.indexer.unique_locs)), 
             shape=(self.nf, self.nt))
        s, r = np.triu_indices(self.p, k=0)
        self._vech_inds = r+s*self.p
        self.unique_locs = self.indexer.unique_locs
        self.free_names = self.free_df["label"]
        self.theta_names = self.free_df.iloc[self._first_locs]["label"]
        self._grad_kws = dict(dA=self.dA, m_size=self.m_size, m_type=self.m_kind, 
                              vind=self._vech_inds, n=self.nf, p2=self.p2)
        self._hess_kws = dict(dA=self.dA, m_size=self.m_size, d2_inds=self.d2_inds,
                              first_deriv_type=self.m_kind,  second_deriv_type=self.d2_kind,
                              n=self.nf,  vech_inds=self._vech_inds)
            

    
    def func(self, theta, per_group=False):
        free = self.transform_theta_to_free(theta)
        if per_group:
            f = np.zeros(self.n_groups)
            if np.iscomplexobj(theta):
                f = f.astype(complex)
        else:
            f = 0.0
        for i in range(self.n_groups):
            group_free = self.transform_free_to_group_free(free, i)
            par = self.free_to_par(group_free, i)
            mats = self.par_to_model_mats(par, i)
            Sigma, mu = self._implied_cov_mean(*mats)
            r = (self.model_data.sample_mean[i]-mu).flatten()
            V = np.linalg.inv(Sigma)
            trSV = np.trace(V.dot(self.model_data.sample_cov[i]))
            rVr = np.dot(r.T.dot(V), r) 
            if np.any(np.iscomplex(Sigma)):
                s, lnd = np.linalg.slogdet(Sigma)
                lndS = np.log(s)+lnd
            else:
                s, lndS = np.linalg.slogdet(Sigma)
            fi = (rVr + lndS + trSV + self.model_data.const[i]) * self.gweights[i]
            if (s==-1) or (fi < -1):
                fi += np.inf
            if per_group:
                f[i] = fi
            else:
                f += fi
        return f
    
    def gradient(self, theta, per_group=False):
        free = self.transform_theta_to_free(theta)
        if per_group:
            g = np.zeros((self.n_groups, self.n_total_free))
        else:
            g = np.zeros(self.n_total_free)
        for i in range(self.n_groups):
            group_free = self.transform_free_to_group_free(free, i)
            par = self.free_to_par(group_free, i)
            L, B, F, P, a, b = self.par_to_model_mats(par, i)
            B = np.linalg.inv(np.eye(B.shape[0]) - B)
            LB = np.dot(L, B)
            mu = (a+LB.dot(b.T).T).reshape(-1)
            a, b = a.flatten(), b.flatten()
            Sigma = LB.dot(F).dot(LB.T) + P
            R = self.model_data.sample_cov[i] - Sigma
            Sinv = np.linalg.inv(Sigma)
            VRV = Sinv.dot(R).dot(Sinv)
            rtV = (self.model_data.sample_mean[i].flatten() - mu).dot(Sinv)
            gi = np.zeros(self.nf)
            kws = self._grad_kws
            gi = _dloglike_mu(gi, L, B, F, b, a,VRV, rtV, **kws) * self.gweights[i]
            gi = self.jac_group_free_to_free(gi, i)
            if per_group:
                g[i] = gi
            else:
                g = g + gi
        g = self.transform_free_to_theta(g)
        return g
    
    def gradient2(self, theta, per_group=False):
        free = self.transform_theta_to_free(theta)
        if per_group:
            g = np.zeros((self.n_groups, self.n_total_free))
        else:
            g = np.zeros(self.n_total_free)
        for i in range(self.n_groups):
            group_free = self.transform_free_to_group_free(free, i)
            par = self.free_to_par(group_free, i)
            L, B, F, P, a, b = self.par_to_model_mats(par, i)
            B = np.linalg.inv(np.eye(B.shape[0]) - B)
            LB = np.dot(L, B)
        
            mu = (a+LB.dot(b.T).T).reshape(-1)
            a, b = a.flatten(), b.flatten()
            Bb = B.dot(b)
            S =  self.model_data.sample_cov[i]
            Sigma = LB.dot(F).dot(LB.T) + P
            r = (self.model_data.sample_mean[i]-mu).flatten()
            V = np.linalg.inv(Sigma)
            VRV = V.dot(S+np.outer(r, r) - Sigma).dot(V)
            Vr = np.dot(V, r)
    
            VrbtBt = np.outer(Vr, Bb)
            T1 = LB.dot(F.dot(B.T))
            dL = -2.0 * (VRV.dot(T1) + VrbtBt)
            dB = -2.0 * (LB.T.dot(VRV).dot(T1) + LB.T.dot(VrbtBt))
            dF = -LB.T.dot(VRV).dot(LB)
            dP = -VRV
            da = -2 * Vr
            db = -2 * np.dot(LB.T, Vr)
            ind = self.indexers[i]
            gi = np.zeros(self.nf)
            rows, cols = ind.row_indices,  ind.col_indices
            sl = ind.slices_nonzero
            gi[sl[0]] = dL[rows[sl[0]], cols[sl[0]]]
            gi[sl[1]] = dB[rows[sl[1]], cols[sl[1]]]
            gi[sl[2]] = dF[rows[sl[2]], cols[sl[2]]]
            gi[sl[2]] = gi[sl[2]] * (1+1*(rows[sl[2]]!=cols[sl[2]]))
            gi[sl[3]] = dP[rows[sl[3]], cols[sl[3]]]
            gi[sl[3]] = gi[sl[3]] * (1+1*(rows[sl[3]]!=cols[sl[3]]))
            gi[sl[4]] = da[cols[sl[4]]]
            gi[sl[5]] = db[cols[sl[5]]]
            gi = gi * self.gweights[i]
            gi = self.jac_group_free_to_free(gi, i)
            if per_group:
                g[i] = gi
            else:
                g = g + gi
        g = self.transform_free_to_theta.dot(g)
        return g

    def hessian(self, theta, per_group=False, method=0):
        free = self.transform_theta_to_free(theta)
        if per_group:
            H = np.zeros((self.n_groups, self.n_total_free, self.n_total_free))
        else:
            H = np.zeros((self.n_total_free, self.n_total_free))
        for i in range(self.n_groups):
            group_free = self.transform_free_to_group_free(free, i)
            par = self.free_to_par(group_free, i)
            L, B, F, P, a, b = self.par_to_model_mats(par, i)
            B = np.linalg.inv(np.eye(B.shape[0]) - B)   
            LB = np.dot(L, B)
            mu = (a+LB.dot(b.T).T).reshape(-1)
            a, b = a.flatten(), b.flatten()
            kws = self._hess_kws
            Sigma = LB.dot(F).dot(LB.T) + P
            S = self.model_data.sample_cov[i]
            R = S - Sigma
            Sinv = np.linalg.inv(Sigma)
            VRV = Sinv.dot(R).dot(Sinv)
            V = np.linalg.inv(Sigma)
            VRV = V.dot(R).dot(V)
            rtV = (self.model_data.sample_mean[i].flatten() - mu).dot(V)
            Hi = np.zeros((self.nf,)*2)
            d1Sm = np.zeros((self.nf, self.p, self.p+1))
            Hi = _d2loglike_mu(H=Hi, d1Sm=d1Sm, L=L, B=B, F=F, P=P, a=a, b=b,VRV=VRV, rtV=rtV, 
                               V=V,  **kws) 
            Hi = Hi * self.gweights[i]
            Hi = self.jac_group_free_to_free(Hi, i, axes=(0, 1))
            if per_group:
                H[i] = Hi
            else:
                H = H + Hi
        H = self.jac_group_free_to_theta(H, axes=(0, 1))
        return H

    def dsigma_mu(self, theta):
        free = self.transform_theta_to_free(theta)
        dSm = self.dSm.copy()
        for i in range(self.n_groups):
            group_free = self.transform_free_to_group_free(free, i)
            par = self.free_to_par(group_free, i)
            L, B, F, P, a, b = self.par_to_model_mats(par, i)
            B = np.linalg.inv(np.eye(B.shape[0])-B)
            kws =  dict(dA=self.dA, m_size=self.m_size, m_type=self.m_kind,
                        vind=self._vech_inds, n=self.nf, p2=self.p2)
            a, b = a.flatten(), b.flatten()
            dSm[i]  = _dsigma_mu(dSm[i], L, B, F, b, a,**kws)
        return dSm

    def d2sigma_mu(self, theta):
        free =self.transform_theta_to_free(theta)
        d2Sm = self.d2Sm.copy()
        for i in range(self.n_groups):
            group_free = self.transform_free_to_group_free(free, i)
            par = self.free_to_par(group_free, i)
            L, B, F, P, a, b = self.par_to_model_mats(par, i)
            B = np.linalg.inv(np.eye(B.shape[0])-B)
            a, b = a.flatten(), b.flatten()
            kws =  dict(dA=self.dA, m_size=self.m_size,  m_type=self.d2_kind, 
                        d2_inds=self.d2_inds, vind=self._vech_inds, n=self.nf2, 
                        p2=self.p2)
            d2Sm[i] = _d2sigma_mu(d2Sm[i], L, B, F, b, a,**kws)
        return d2Sm
            



    def _implied_cov_mean(self, L, B, F, P, a, b):
        B = np.linalg.inv(np.eye(B.shape[0])-B)
        LB = np.dot(L, B)
        Sigma = LB.dot(F).dot(LB.T) + P
        mu = (a+LB.dot(b.T).T).reshape(-1)
        return Sigma, mu
    
    def implied_cov_mean(self, theta):
        free = self.transform_theta_to_free(theta)
        Sigma = np.zeros((self.n_groups, self.p, self.p))
        mu = np.zeros((self.n_groups, self.p))
        for i in range(self.n_groups):
            group_free = self.transform_free_to_group_free(free, i)
            par = self.free_to_par(group_free, i)
            mats = self.par_to_model_mats(par, i)
            Sigma[i], mu[i] = self._implied_cov_mean(*mats)
        return Sigma, mu
    
    def implied_sample_stats(self, theta):
        free = self.transform_theta_to_free(theta)
        Sigmamu = np.zeros((self.n_groups, self.p2+self.p))
        if np.iscomplexobj(theta):
            Sigmamu = Sigmamu.astype(complex)
        for i in range(self.n_groups):
            group_free = self.transform_free_to_group_free(free, i)
            par = self.free_to_par(group_free, i)
            L, B, F, P, a, b = self.par_to_model_mats(par, i)
            B = np.linalg.inv(np.eye(B.shape[0])-B)
            LB = np.dot(L, B)
            Sigma = LB.dot(F).dot(LB.T) + P
            mu = (a+LB.dot(b.T).T).reshape(-1)
            s, mu = _vech(Sigma), mu.flatten()
            Sigmamu[i, :self.p2] = s
            Sigmamu[i, self.p2:] = mu
        return Sigmamu
    
    def loglike(self, theta, level="sample"):
        free = self.transform_theta_to_free(theta)
        if level == "group":
            f = np.zeros(self.n_groups)
        elif level == "observation" or level == "sample":
            f = np.zeros(self.n_obs)
        if np.iscomplexobj(theta):
            f = f.astype(complex)
        for i in range(self.n_groups):
            ix =  self.model_data.group_indices[i]
            group_free = self.transform_free_to_group_free(free, i)
            par = self.free_to_par(group_free, i)
            mats = self.par_to_model_mats(par, i)
            Sigma, mu = self._implied_cov_mean(*mats)
            L = sp.linalg.cholesky(Sigma)
            Y = self.model_data.data[ix] - mu
            Z = sp.linalg.solve_triangular(L, Y.T, trans=1).T #np.dot(X, np.linalg.inv(L.T))
            t1 = 2.0 * np.log(np.diag(L)).sum() 
            t2 = np.sum(Z**2, axis=1)
            ll = (t1 + t2 + self.ll_const) / 2
            if level == "group":
                f[i] = np.sum(ll)
            else:
                f[ix] = ll
        if level == "sample":
            f = np.sum(f)
        return f
    
    
    def _fit(self, theta_init=None, minimize_kws=None, minimize_options=None, use_hess=False):
        bounds = self.bounds_theta
        func = self.func
        grad = self.gradient
        if use_hess:
            hess = self.hessian
        else:
            hess = None
        theta = self.theta if theta_init is None else theta_init
        default_minimize_options = dict(initial_tr_radius=1.0, verbose=3)
        minimize_options = handle_default_kws(minimize_options, default_minimize_options)
                
        default_minimize_kws = dict(method="trust-constr", options=minimize_options)
        minimize_kws = handle_default_kws(minimize_kws, default_minimize_kws)
        res = sp.optimize.minimize(func, x0=theta, jac=grad, hess=hess,  
                                   bounds=bounds,  **minimize_kws)
        return res
    
    def fit(self,  theta_init=None, minimize_kws=None, minimize_options=None, use_hess=False):
        res = self._fit(theta_init=theta_init, minimize_kws=minimize_kws, minimize_options=minimize_options, use_hess=use_hess)
        if np.linalg.norm(res.grad)>1e16:
            if minimize_options is None:
                minimize_options = {}
            minimize_options["initial_tr_radius"]=0.01
            res = self._fit(minimize_kws=minimize_kws, minimize_options=minimize_options, use_hess=use_hess)
        self.opt_res = res
        self.theta = res.x
        self.free = self.transform_theta_to_free(self.theta)
        self.res = pd.DataFrame(res.x, index=self.theta_names, 
                                columns=["estimate"])
        self.res["se"] = np.sqrt(np.diag(np.linalg.inv(self.hessian(self.theta)*self.n_obs/2)))
        self.res_free = pd.DataFrame(self.transform_theta_to_free(self.res.values), 
                                     index=self.free_names, columns=self.res.columns)
        mats = {}
        free = self.transform_theta_to_free(self.theta)
        for i in range(self.n_groups):
            group_free = self.transform_free_to_group_free(free, i)
            par = self.free_to_par(group_free, i)
            mlist = self.par_to_model_mats(par, i)
            mats[i] = {}
            for j,  mat in enumerate(mlist):
                mats[i][j] = pd.DataFrame(mat, index=self.mat_rows[j],
                                          columns=self.mat_cols[j])
        self.mats = mats
        
