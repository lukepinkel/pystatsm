#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 22:04:01 2022

@author: lukepinkel
"""
import tqdm
import numpy as np
import scipy as sp
import scipy.linalg
import pandas as pd
from .efa import FactorAnalysis
from .align_loadings import align_model_matrices 
from ..utilities.random import r_lkj, exact_rmvnorm
from ..utilities.func_utils import handle_default_kws
from ..utilities.linalg_operations import vec, invecl, vecl
from ..utilities.data_utils import _csd

class FactorModelSim(object):
    
    def __init__(self, n_vars=12, n_facs=3, L=None, Phi=None, 
                 Psi=None, rng=None, seed=None):
        self.rng = np.random.default_rng(seed) if rng is None else rng
        self.n_vars = n_vars
        self.n_facs = n_facs
        self.L = L
        self.Phi = Phi
        self.Psi = Psi
        
    def _generate_loadings(self, random=False, r_kws=None):
        L = np.zeros((self.n_vars, self.n_facs))
        inds = list(zip(np.array_split(np.arange(self.n_vars), self.n_facs), np.arange(self.n_facs)))
        default_r_kws = dict(low=0.4, high=0.9)
        r_kws = {} if r_kws is None else r_kws
        r_kws = {**default_r_kws, **r_kws}
        for rows, col in inds:
            s =len(rows)
            if random:
                u =self.rng.uniform(size=s, **r_kws)
                u = np.sort(u)[::-1]
            else:
                u = np.linspace(1.0, 0.4, s)
            L[rows, col] = u
        return L
    
    def _generate_factor_corr(self, random=False, r_kws=None, rho=0.5):
        default_r_kws = dict(eta=2.0)
        r_kws = {} if r_kws is None else r_kws
        r_kws = {**default_r_kws, **r_kws}
        if random:
            Phi = r_lkj(n=1, dim=self.n_facs, rng=self.rng, **r_kws)
        else:
            Phi = rho**sp.linalg.toeplitz(np.arange(self.n_facs))
            s = (-1)**np.arange(self.n_facs).reshape(-1, 1)
            Phi = s*Phi*s.T
        return Phi
    
    def _generate_residual_cov(self, random=True, dist=None):
        if random:
            if dist is None:
                dist = lambda size: self.rng.uniform(low=0.3, high=0.7, size=size)
            psi = dist(self.n_vars)
        else:
            psi = np.linspace(0.3, 0.7, self.n_vars)
        Psi = np.diag(psi)
        return Psi
    
    def simulate_cov(self, loadings_kws=None, factor_corr_kws=None,
                     residual_cov_kws=None):
        loadings_kws = {} if loadings_kws is None else loadings_kws
        factor_corr_kws = {} if factor_corr_kws is None else factor_corr_kws
        residual_cov_kws = {} if residual_cov_kws is None else residual_cov_kws
        
        self.L = self._generate_loadings(**loadings_kws) if self.L is None else self.L
        self.Phi = self._generate_factor_corr(**factor_corr_kws) if self.Phi is None else self.Phi
        self.Psi = self._generate_residual_cov(**residual_cov_kws) if self.Psi is None else self.Psi
        self.C = sp.linalg.block_diag(self.Phi, self.Psi)

        self.Sigma = self.L.dot(self.Phi).dot(self.L.T) + self.Psi
        
    def simulate_data(self, n_obs=1000, exact=True):
        if exact:
            Z = exact_rmvnorm(n=n_obs, S=self.C)
        else:
            Z = self.rng.multivariate_normal(mean=np.zeros(self.n_vars+self.n_facs),
                                             cov=self.C, size=n_obs)
        X = Z[:, :self.n_facs].dot(self.L.T) + Z[:, self.n_facs:]
        return Z, X

        
    

def cov(arr):
    arr = arr - np.mean(arr, axis=0)
    S = np.dot(arr.T, arr) / len(arr)
    return S

class FactorEstimateSim(object):
    
    def __init__(self, p=9, m=3, sim_kws=None, cov_kws=None, data_kws=None):
        sim_kws = handle_default_kws(sim_kws, {})
        cov_kws = handle_default_kws(cov_kws, {})
        data_kws = handle_default_kws(data_kws, dict(exact=False))
            
        model_sim = FactorModelSim(n_vars=p, n_facs=m, **sim_kws)
        model_sim.simulate_cov(**cov_kws)
        self.model_sim = model_sim
        self.p, self.m = p, m
    
    def align_model(self, model):
        L, L_se, Phi, Phi_se = model.L, model.L_se, model.Phi, model.Phi_se
        L, L_se, Phi, Phi_se, _, _ = align_model_matrices(self.model_sim.L, L, Phi, 
                                                    L_se, Phi_se)
        Psi, Psi_se = np.diag(model.Psi), model.psi_se
        if model.rotation_type == "oblique":
            params = np.concatenate([vec(L), vecl(Phi), Psi])
            params_se = np.concatenate([vec(L_se), vecl(Phi_se), Psi_se])
        else:
            params = np.concatenate([vec(L), Psi])
            params_se = np.concatenate([vec(L_se), Psi_se])
        return params, params_se

    
    def simulate(self, n=100, n_obs=200, model_kws=None):
        model_kws = handle_default_kws(model_kws, dict(n_factors=self.m, 
                                                       rotation_method="varimax",
                                                       rotation_type="oblique"))
        pbar = tqdm.tqdm(total=n, smoothing=0.001)
        Z, X = self.model_sim.simulate_data(n_obs=n_obs, exact=False)
        model = FactorAnalysis(X, **model_kws)
        model.fit()
        model.rho_init = np.log(np.diag(self.model_sim.Psi))
        params = np.zeros((n, len(model.res)))
        chi2_table = np.zeros((n, 2, 3))
        ifi = np.zeros((n, 5, 1))
        mfi = np.zeros((n, 7, 1))
        params_se = np.zeros_like(params)
        cov_mats = np.zeros((n, self.p, self.p))
        for i in range(n):
            Z, X = self.model_sim.simulate_data(n_obs=n_obs, exact=False)
            model.S = cov(X)#model._process_data(X, None, X.shape[0])
            cov_mats[i] = model.S.copy()
            model.fit()
            params[i], params_se[i] = self.align_model(model)
            chi2_table[i] = model.chi2_table
            ifi[i] = model.incrimental_fit_indices
            mfi[i] = model.misc_fit_indices
            pbar.update(1)
        pbar.close()
        if model.rotation_type == "oblique":
            sim_pars = np.concatenate([vec(self.model_sim.L), vecl(self.model_sim.Phi), 
                                       np.diag(self.model_sim.Psi)])
        else:
            sim_pars = np.concatenate([vec(self.model_sim.L), np.diag(self.model_sim.Psi)])
        self.params, self.params_se =params, params_se
        res = np.vstack((params.mean(axis=0),
                         sim_pars,
                         params.std(axis=0),
                         params_se.mean(axis=0))).T
        self.res = pd.DataFrame(res, index=model.res.index, columns=["Mean", "Generative Values", "SE", "SEM"])
        self.res["ratio"] = self.res["SE"] / self.res["SEM"]
        self.cov_mats = cov_mats
        self.chi2_table = chi2_table
        self.ifi = ifi
        self.mfi = mfi

            
        
        
        
    
    
            
            
            