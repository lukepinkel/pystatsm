#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 07:56:38 2023

@author: lukepinkel
"""

import re
import numpy as np
import pandas as pd

class FormulaParser:
    def __init__(self, formulas):
        self._formulas = formulas
        self._param_list = self.parse_formula(formulas)
        self._param_df = pd.DataFrame(self._param_list)

    @staticmethod  
    def parse_formula(formulas):
        formulas = re.sub(r'\s*#.*', '', formulas)
        parameters= []
        equations = formulas.strip().split('\n')
        for eq in equations:
            if eq.strip():
                parameters = FormulaParser.unpack_equation(eq ,parameters)
        return parameters
        
    @staticmethod  
    def get_var_pair(ls, rs, rel):
        comps = rs.strip().split('*')
        ls = ls.strip()
        if len(comps) > 1:
            name = comps[1].strip()
            mod = comps[0].strip()
            try:
                fixedval = float(mod)
                fixed = True
                label = None
            except ValueError:
                fixedval = None
                fixed = False
                label = mod
        else:
            mod = None
            fixed = False
            name = comps[0].strip()
            label= None#f"{ls}{rel}{name}"
            fixedval = None
        row = {"lhs":ls,"rel":rel ,"rhs":name, "mod":mod,
               "label":label, "fixedval":fixedval, 
               "fixed":fixed}
        return row
     
    @staticmethod        
    def unpack_equation(eq, parameters):
        if "=~" in eq:
            rel = "=~"
        elif "~~" in eq:
            rel = "~~"
        else:
            rel = "~"
        lhss, rhss = eq.split(rel)
        for ls in lhss.split('+'):
            for rs in rhss.split('+'):
                row = FormulaParser.get_var_pair(ls, rs, rel)
                parameters.append(row)
        return parameters
    
    @staticmethod
    def classify_variables(param_df):
        one_set = set(["1"])
        measurement_mask = param_df["rel"] == "=~"
        regressions_mask  = ((param_df["rel"] == "~") & (param_df["rhs"]!="1"))
        all_var_names = set(param_df[["lhs", "rhs"]].values.flatten()) - one_set
        nob_var_names = set(param_df.loc[measurement_mask, "lhs"]) #unobserved variables i.e. lv
        obs_var_names = all_var_names - nob_var_names              #observed variables i.e. ov
        ind_var_names = set(param_df.loc[measurement_mask, "rhs"]) #indicator variables observed or unobserved v_inds
        end_var_names = set(param_df.loc[regressions_mask, "lhs"]) #
        exo_var_names = set(param_df.loc[regressions_mask, "rhs"]) - one_set
        reg_var_names = set.union(end_var_names, exo_var_names)
        lvo_var_names = reg_var_names - nob_var_names
        lav_var_names = lvo_var_names | nob_var_names
        lox_var_names = lav_var_names - (nob_var_names | end_var_names | ind_var_names) #observed exog variables in structural part of model i.e.latent observed
        loy_var_names = lav_var_names - (nob_var_names | exo_var_names | ind_var_names) #observed exog variables in structural part of model i.e.latent observed
        
        #lvo_var_names - the rows and cols of Lambda with identity
        #obs_var_names - rows of Lambda
        #lav_var_names - cols of lambda and row/col of beta and phi
        #lox_var_names - fix Phi and beta at sample stats
        #loy_var_names and lox_var_names should partition lvo_var_names
        names = dict(all=all_var_names, nob=nob_var_names, 
                     obs=obs_var_names, ind=ind_var_names, 
                     end=end_var_names, exo=exo_var_names, 
                     reg=reg_var_names, lvo=lvo_var_names, 
                     lav=lav_var_names, lox=lox_var_names, 
                     loy=loy_var_names)
        return names
        
