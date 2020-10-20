#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 19:52:41 2020

@author: lukepinkel
"""

import re
import numpy as np
import scipy as sp
import scipy.stats
import pandas as pd # analysis:ignore
from ..utilities.random_corr import multi_rand, vine_corr
from ..utilities.linalg_operations import invech
from ..pylmm.model_matrices import construct_model_matrices


def replace_duplicate_operators(match):
    return match.group()[-1:]


def parse_vars(formula, model_dict):
    n_obs = model_dict['n_obs']
    matches = re.findall("\([^)]+[|][^)]+\)", formula)
    groups = [re.search("\(([^)]+)\|([^)]+)\)", x).groups() for x in matches]
    frm = formula
    for x in matches:
        frm = frm.replace(x, "")
    fe_form = re.sub("(\+|\-)(\+|\-)+", replace_duplicate_operators, frm)
    re_form, re_groupings = list(zip(*groups))
    re_form, re_groupings = set(re_form), set(re_groupings)

    yvars, fe_form = re.split("[~]", fe_form)
    fe_form = re.sub("\+$", "", fe_form)
    
    fixed_vars = re.split("[(\+|\-)]", fe_form)
    randm_vars = [re.split("[(\+|\-)]", x) for x in re_form]
    randm_vars = set([x for y in randm_vars for x in y])
    vars_ = sorted(randm_vars.union(fixed_vars).union(re_groupings))
    cont_vars = sorted(randm_vars.union(fixed_vars))
    if str(1) in cont_vars:
        cont_vars.remove(str(1))
        
    vars_ = sorted(set(cont_vars).union(re_groupings))
    if str(1) in vars_:
        vars_.remove(str(1))
        
    df = pd.DataFrame(np.zeros((n_obs, len(vars_))), columns=vars_)
    df[yvars] = 0
    return df, re_groupings, cont_vars

def _generate_model(df, formula, re_groupings, cont_vars, model_dict, r=0.5):
    beta =  model_dict['beta']
    gcov = model_dict['gcov']
    ginfo = model_dict['ginfo']
    mu = model_dict['mu']
    n_obs = model_dict['n_obs']
    vcov = model_dict['vcov']

    for x in re_groupings:
        n_grp, n_per = ginfo[x]['n_grp'], ginfo[x]['n_per']
        df[x] = np.kron(np.arange(n_grp), np.ones(n_per))


    df[list(cont_vars)] = sp.stats.multivariate_normal(mu, vcov).rvs(n_obs)
    X, Z, y, dims = construct_model_matrices(formula, data=df)
    U = []
    for x in re_groupings:
        n_grp = ginfo[x]['n_grp'],
        Gi = gcov[x]
        globals()["Gi"]=Gi
        globals()["n_grp"]=n_grp
        if len(Gi)==1:
            Ui = np.random.normal(0, 1, size=n_grp)
            Ui -= Ui.mean()
            Ui /= Ui.std()
            Ui *= np.sqrt(Gi[0])
        else:
            Ui = multi_rand(Gi, *n_grp).flatten()
        U.append(Ui)
    u = np.concatenate(U)
    eta = X.dot(beta)+Z.dot(u)
    eta_var = eta.var()
    rsq = r**2
    df['y'] = sp.stats.norm(eta, np.sqrt((1-rsq)/rsq*eta_var)).rvs()
    return df, u, eta
       
def generate_data(formula, model_dict, r=0.5):
    df, re_groupings, cont_vars = parse_vars(formula, model_dict)
    df, u, eta = _generate_model(df, formula, re_groupings, cont_vars,  model_dict, r)
    return df, formula, u, eta
    

class SimulatedGLMM:
    
    def __init__(self, n_grp, n_per, r=0.9):
        formula = "y~1+x1+x2+x3+(1+x4|id1)"
        gcov = invech(np.array([0.8, -0.4, 0.8]))
        beta = np.array([0.25, -0.25, 0.5, -0.5])
        model_dict = {}
        model_dict['gcov'] = {'id1':gcov}
        model_dict['ginfo'] = {'id1':dict(n_grp=n_grp, n_per=n_per)}
        model_dict['mu'] = np.zeros(4)
        model_dict['vcov'] = vine_corr(4, 20)
        model_dict['beta'] = beta
        model_dict['n_obs'] = int(n_grp * n_per)
        self.formula = formula
        self.model_dict = model_dict
        self.r = r
        self.df, _, self.u, self.eta = generate_data(formula, model_dict, r)
        self.df = self.df.rename(columns=dict(y="eta"))
        self.df["mu"] = np.exp(self.df["eta"]) / (1 + np.exp(self.df["eta"]))
        self.df["y"] = sp.stats.binom(n=1, p=self.df["mu"]).rvs()
        
    

class SimulatedGLMM2:
    
    def __init__(self, formula, n_grp, n_per, gcov, beta, mu, vcov, r=0.9):
        model_dict = {}
        model_dict['gcov'] = {'id1':gcov}
        model_dict['ginfo'] = {'id1':dict(n_grp=n_grp, n_per=n_per)}
        model_dict['mu'] = mu
        model_dict['vcov'] = vcov
        model_dict['beta'] = beta
        model_dict['n_obs'] = int(n_grp * n_per)
        self.formula = formula
        self.model_dict = model_dict
        self.r = r
        self.df, _, self.u, self.eta = generate_data(formula, model_dict, r)
        self.df = self.df.rename(columns=dict(y="eta"))
        self.df["mu"] = np.exp(self.df["eta"]) / (1 + np.exp(self.df["eta"]))
       
    
        
            
        


        







