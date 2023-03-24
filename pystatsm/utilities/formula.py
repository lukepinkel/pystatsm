#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 05:54:38 2022

@author: lukepinkel
"""

import re
import patsy
import numpy as np
import pandas as pd
from patsy import highlevel

def find_smooth_terms(formula):
    return re.findall("(?<=s[(])(.*?)(?=[)])", formula)

def find_ranef_terms(formula):
    return re.findall("\([^)]+[|][^)]+\)", formula)

def replace_duplicate_operators(match):
    return match.group()[-1:]

        

def parse_random_effects(formula):
    matches = re.findall("\([^)]+[|][^)]+\)", formula)
    re_terms = [re.search("\(([^)]+)\|([^)]+)\)", x).groups() for x in matches]
    frm = formula
    for x in matches:
        frm = frm.replace(x, "")
    fe_form = re.sub("(\+|\-)(\+|\-)+", replace_duplicate_operators, frm)
    yvars, fe_form = re.split("[~]", fe_form)
    fe_form = re.sub("\+$", "", fe_form)
    y_vars = re.split(",", re.sub("\(|\)", "", yvars))
    y_vars = [x.strip() for x in y_vars]
    re_forms, re_groupings = list(zip(*re_terms))
    x_vars = re.split("[+]", fe_form)
    for re_form in re_forms:
        x_vars.extend(re.split("[+]", re_form))

    x_vars = list(dict.fromkeys(x_vars))
    consts = ["0", "1"]
    x_vars = [x for x in x_vars if x not in consts]
    model_info = dict(y_vars=y_vars, x_vars=x_vars, fe_form=fe_form,
                      re_terms=re_terms, re_forms=re_forms,
                      re_groupings=re_groupings)
    return model_info


def get_parametric_formula(formula):
    tmp = re.findall("s[(].*?[)]", formula)
    frm = formula[:-1]+formula[-1:]
    for x in tmp:
        frm = frm.replace(x, "")
    frm = re.sub("(\+|\-)(\+|\-)+", replace_duplicate_operators, frm)
    frm = re.sub("\+$", "", frm)
    return frm

def parse_smooth_term(term):
    arg_order = ['x', 'df', 'kind', 'by']
    tokens = [t.strip() for t in term.split(',')]
    smooth_info = dict(x=None, df=10, kind='cr', by=None)
    for i, token in enumerate(tokens):
        if token.find('=')!=-1:
            key, val = token.split('=')
            smooth_info[key.strip()] = val.strip()
        else:
            key, val = arg_order[i], token.strip()
            smooth_info[key] = val
    smooth_info['df'] = int(smooth_info['df'])
    smooth_info['kind'] = smooth_info['kind'].replace("'", "").replace("'", "")
    return smooth_info

def parse_smooths(formula):
    smooths = []
    smooth_terms = re.findall("(?<=s[(])(.*?)(?=[)])", formula)
    for term in smooth_terms:
        smooths.append(parse_smooth_term(term))
    fe_form = get_parametric_formula(formula)
    y_vars, fe_form = re.split("[~]", fe_form)
    x_vars = re.split("[+]", fe_form)
    consts = ["0", "1"]
    x_vars = [x for x in x_vars if x not in consts]
    model_info = dict(y_vars=y_vars, x_vars=x_vars, fe_form=fe_form,
                      smooths=smooths)
    return model_info


def design_matrices(formula_like, data={}, eval_env=0, NA_action="drop",
                    return_type="dataframe"):
    eval_env = highlevel.EvalEnvironment.capture(eval_env, reference=1)
    (lhs, rhs) = highlevel._do_highlevel_design(formula_like, data=data,
                                   return_type=return_type,
                                   eval_env=eval_env,
                                   NA_action=NA_action)
    return lhs, rhs






