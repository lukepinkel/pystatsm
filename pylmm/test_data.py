#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 16 22:22:58 2020

@author: lukepinkel
"""
import re
import numpy as np  # analysis:ignore
import scipy as sp  # analysis:ignore
import scipy.stats  # analysis:ignore
import pandas as pd # analysis:ignore
from .model_matrices import construct_model_matrices

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

def _generate_model(df, formula, re_groupings, cont_vars, model_dict, r=0.5, 
                    rng=None):
    if rng is None:
        rng = np.random.default_rng()
        
    beta =  model_dict['beta']
    gcov = model_dict['gcov']
    ginfo = model_dict['ginfo']
    mu = model_dict['mu']
    n_obs = model_dict['n_obs']
    vcov = model_dict['vcov']

    for x in re_groupings:
        n_grp, n_per = ginfo[x]['n_grp'], ginfo[x]['n_per']
        df[x] = np.kron(np.arange(n_grp), np.ones(n_per))


    df[list(cont_vars)] = rng.multivariate_normal(mu, vcov, size=n_obs)
    X, Z, y, dims = construct_model_matrices(formula, data=df)
    U = []
    for x in re_groupings:
        n_grp = ginfo[x]['n_grp'],
        Gi = gcov[x]
        Ui = rng.multivariate_normal(np.zeros(Gi.shape[1]), Gi, 
                                     size=n_grp).flatten()
        U.append(Ui)
    u = np.concatenate(U)
    eta = X.dot(beta)+Z.dot(u)
    eta_var = eta.var()
    rsq = r**2
    df['y'] = rng.normal(eta, np.sqrt((1-rsq)/rsq*eta_var))
    return df
       
def generate_data(formula, model_dict, r=0.5, rng=None):
    df, re_groupings, cont_vars = parse_vars(formula, model_dict)
    df = _generate_model(df, formula, re_groupings, cont_vars,  model_dict, r,
                         rng=rng)
    return df, formula
    
      












                  