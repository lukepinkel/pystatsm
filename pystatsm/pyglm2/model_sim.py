#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 06:53:41 2023

@author: lukepinkel
"""

import re
import patsy
import numpy as np                                          
import pandas as pd                                          
from ..utilities import random
from ..utilities import func_utils
from ..utilities import cov_utils 

np.set_printoptions(suppress=True)
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%f' % x)

seed = 23644
rng = np.random.default_rng(seed)



class LinearModelSim(object):
    
    def __init__(self, n_obs=1000, n_var=10, formula=None, cat_terms=None,
                 x_vars=None, nlevels=None, corr_kws=None, coef_kws=None):

        self.cat_terms = [] if cat_terms is None else cat_terms
        self.x_vars = [f"x{i}" for i in range(n_var)] if x_vars is None else x_vars
        self.formula = "+".join(self.x_vars) if formula is None else formula
        self.nlevels = {} if nlevels is None else nlevels
        for cat in self.cat_terms:
            if cat not in self.nlevels.keys():
                self.nlevels[cat] = 2
        self.nlevels = nlevels
        self.n_obs = n_obs
        self.n_var = n_var
        self.cat_inds = [self.x_vars.index(x) for x in self.cat_terms]
        corr_kws = func_utils.handle_default_kws(corr_kws, {})
        self.corr = self._make_corr_matrix(self.n_var, **corr_kws)
        self.Z1 = self._make_underlying_matrix(self.n_obs, self.corr)
        self.Z2 = self.Z1.copy()
        
        for i in self.cat_inds:
            self.Z2[:, i] = self._make_categorical(self.Z1[:, i], self.nlevels[self.x_vars[i]])
        
        self.data = pd.DataFrame(self.Z2, columns=self.x_vars)
        self.X = patsy.dmatrix(self.formula, data=self.data, return_type="dataframe")
        self.n_coef = self.X.shape[1]
        self.design_info = self.X.design_info
        self.term_names = self.design_info.term_names
        self.term_name_slices = self.design_info.term_name_slices
        self.factor_infos = self.design_info.factor_infos
        self.terms = self.design_info.terms
        self.term_class = self._classify_terms(self.n_coef, self.terms, self.factor_infos,
                                               self.term_name_slices)
        coef_kws = func_utils.handle_default_kws(coef_kws, {})
        self.beta = self.make_coefs(**coef_kws)
        self.linpred = self.X.dot(self.beta)
        self.linpred_var = np.var(self.linpred)
        
    @staticmethod
    def _make_corr_matrix(n_var, corr_method=cov_utils.get_eig_corr, corr_method_kws=None, 
                          eig_kws=None):
        if n_var>1:
            corr_method_kws = func_utils.handle_default_kws(corr_method_kws, {"n_var":n_var})
            if corr_method == cov_utils.get_eig_corr:
                default_eig_kws = {"n_var":n_var, "p_eff":0.5, "a":1.0, "b":0.1, "c":0.5}
                eig_kws = func_utils.handle_default_kws(eig_kws, default_eig_kws)
                corr_method_kws["u"] = cov_utils.get_eigvals(**eig_kws)
            R = corr_method(**corr_method_kws)
        else:
            R = np.eye(1)
        return R

    
    @staticmethod
    def _make_underlying_matrix(n_obs, corr, mat_kws=None):
        mat_kws = func_utils.handle_default_kws(mat_kws, {"n":n_obs})
        X = random.exact_rmvnorm(corr, **mat_kws)
        return X
    
    @staticmethod
    def _make_categorical(x, nlevels):
        return func_utils.quantile_cut(x, nlevels)
    
    @staticmethod
    def _classify_terms(n_coef, terms, factor_infos, term_name_slices):
        term_class = np.zeros(n_coef, dtype=int)
        k = 0
        for term in terms:
            term_name = term.name()
            term_slice = term_name_slices[term_name]
            numerical = np.all([factor_infos[factor].type=="numerical" 
                                for factor in term.factors])
            if numerical:
                term_class[term_slice] = -1
            else:
                term_class[term_slice] = k
            k += 1
        return term_class
    
    @staticmethod
    def _get_var(linpred_var, rsquared):
        resid_scale = np.sqrt((1.0 - rsquared) / rsquared * linpred_var)
        return resid_scale
    
    @staticmethod
    def _simulate_y(loc, scale, dist=rng.normal):
        y = dist(loc=loc, scale=scale)
        return y
    
    @staticmethod
    def _make_coef(n_var, p_nnz=0.5):
        n_nnz = max(int(p_nnz * n_var), 1)
        beta = np.zeros(n_var)
        i_nnz = rng.choice(np.arange(n_var), size=n_nnz, replace=False)
        b_nnz = np.zeros(n_nnz)
        b_nnz[:(n_nnz // 2)] = -1
        b_nnz[(n_nnz // 2):] = 1
        beta[i_nnz] = b_nnz
        return beta
    
    @staticmethod
    def _make_coef_cat(n_var):
        if n_var>1:
            coef = np.zeros(n_var)
            n = n_var // 2
            m = n_var - n
            coef[:n] = np.linspace(0, -1, n+1)[1:][::-1]
            coef[n:] = np.linspace(0, 1, m+1)[1:]
            coef[:n] -= (coef[:n].sum()+coef[n:].sum()) / n
        else:
            coef = np.ones(n_var)
        return coef
    
    @staticmethod
    def parse_formula(formula):
        if formula.find("~")!=-1:
            _, formula = re.split("[~]", formula)
        terms = re.split("[+]", formula)
        x_vars = set([y for x in terms for y in re.split("[:]", x)])
        x_vars = set([y for x in x_vars for y in re.split("[*]", x)])
        for match in re.findall("C\([^)]+\)", formula):
          x_vars.remove(match)
        cat_terms = re.findall("(?<=C[(])(.*?)(?=[)])", formula)
        for x in cat_terms:
            x_vars.add(x)
        x_vars = sorted(x_vars)
        cat_terms = cat_terms
        if "1" in x_vars:
            x_vars.remove("1")
        return x_vars, cat_terms
    
    @classmethod
    def from_formula(cls, formula, n_obs=10_000, nlevels=None, **kws):
        x_vars, cat_terms = cls.parse_formula(formula)
        n_var = len(x_vars)
        return cls(n_obs, n_var, formula, cat_terms, x_vars, nlevels, **kws)
    
    def make_coefs(self, p_nnz=0.5, zero_intercept=True):
        beta = np.zeros(self.n_coef)
        u = np.unique(self.term_class)
        if not hasattr(p_nnz, "__len__"):
            p_nnz = (p_nnz,)*len(u)
        for i, c in enumerate(u):
            ix = self.term_class==c
            n_var = int(np.sum(ix))
            if c==-1:
                beta[ix] = self._make_coef(n_var, p_nnz[i])
            else:
                beta[ix] = self._make_coef_cat(n_var)
        if zero_intercept:
            beta[self.term_name_slices["Intercept"]] = 0.0
        return beta
    
    
    def rescale_coefs(self, s):
        self.beta = self.beta / np.sqrt(self.linpred_var) * s
        self.linpred = self.X.dot(self.beta)
        self.linpred_var = np.var(self.linpred)
    
    
    
    
    
    
    
 