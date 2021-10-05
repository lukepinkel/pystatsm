# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 16:58:31 2021

@author: lukepinkel
"""

import re
import patsy
import numpy as np
from ..utilities.splines import (_get_crsplines, _get_bsplines, _get_ccsplines,
                                 crspline_basis, bspline_basis, ccspline_basis,
                                 get_penalty_scale, absorb_constraints)

def parse_smooths(smoother_formula, data):
    smooths = {}
    smooth_terms = re.findall("(?<=s[(])(.*?)(?=[)])", smoother_formula)
    arg_order = ['x', 'df', 'kind', 'by']
    for term in smooth_terms:
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
        smooth_info['kind'] = smooth_info['kind'].replace("'", "")
        var = smooth_info['x']
        smooth_info['x'] = data[smooth_info['x']].values
        smooth_info['kind'] = smooth_info['kind'].replace("'", "")
        if smooth_info['by'] is not None:
            by_design_mat = patsy.dmatrix(f"C({smooth_info['by']})-1", data, 
                                          return_type='dataframe',
                                          eval_env=0)
            finfo = by_design_mat.design_info.factor_infos
            cats = finfo[list(finfo.keys())[0]].categories
            smooth_info['by'] = dict(by_vals=by_design_mat.values,
                                     by_cats=cats)
        smooths[var] = smooth_info
    return smooths

def get_parametric_formula(formula):
    tmp = re.findall("s[(].*?[)]", formula)
    frm = formula[:-1]+formula[-1:]
    for x in tmp:
        frm = frm.replace(x, "")
    frm = re.sub("(\+|\-)(\+|\-)+", replace_duplicate_operators, frm)
    frm = re.sub("\+$", "", frm)
    return frm

def replace_duplicate_operators(match):
    return match.group()[-1:]

def get_smooth(x, df=10, kind="cr", by=None):
    methods = {"cr":_get_crsplines, "cc":_get_bsplines, "bs":_get_ccsplines}
    X, S, knots, fkws = methods[kind](x, df)
    sc = get_penalty_scale(X, S)
    q, _ = np.linalg.qr(X.mean(axis=0).reshape(-1, 1), mode='complete')
    X, S = absorb_constraints(q, X=X, S=S)
    S = S / sc
    smooth_list = []
    if by is not None:
        for i in range(by['by_vals'].shape[1]):
            Xi = X * by['by_vals'][:, [i]]
            x0 = x*by['by_vals'][:,i]
            smooth_list.append(dict(X=Xi, S=S, knots=knots, kind=kind, 
                                        q=q, sc=sc, fkws=fkws, x0=x0,
                                        xm=x0[x0!=0],
                                        by_cat=by['by_cats'][i]))
    else:
        smooth_list = [dict(X=X, S=S, knots=knots, kind=kind, q=q, sc=sc, 
                            fkws=fkws, x0=x, xm=None, by_cat=None)]
    return smooth_list

def get_smooth_terms(smooth_info, Xp):
    varnames, n_parametric = Xp.columns.tolist(), Xp.shape[1]
    smooths, n_smooth_terms, n_total_params = {}, 0, n_parametric
    for key, val in smooth_info.items():
        slist = get_smooth(**val)
        if len(slist)==1:
            smooths[key], = slist
            p_i = smooths[key]['X'].shape[1]
            varnames += [f"{key}{j}" for j in range(1, p_i+1)]
            n_total_params += p_i
            n_smooth_terms += 1
        else:
            for i, x in enumerate(slist):
                by_key = f"{key}_{x['by_cat']}"
                smooths[by_key] = x
                p_i = x['X'].shape[1]
                varnames += [f"{by_key}_{j}" for j in range(1, p_i+1)]
                n_total_params += p_i
                n_smooth_terms += 1
    return smooths, n_smooth_terms, n_total_params, varnames

def get_smooth_matrices(Xp, smooths, n_smooth_terms, n_total_params):
    X, ranks, ldS, start = [Xp], [], [], Xp.shape[1]
    S = np.zeros((n_smooth_terms, n_total_params, n_total_params))
    for i, (var, s) in enumerate(smooths.items()):
        p_i = s['X'].shape[1]
        Si = np.zeros((n_total_params, n_total_params))
        ix = np.arange(start, start+p_i)
        start += p_i
        Si[ix, ix.reshape(-1, 1)] = s['S']
        smooths[var]['ix'], smooths[var]['Si'] = ix, Si
        X.append(smooths[var]['X'])
        S[i] = Si
        ranks.append(np.linalg.matrix_rank(Si))
        u = np.linalg.eigvals(s['S'])
        ldS.append(np.log(u[u>np.finfo(float).eps]).sum())
    return X, S, ranks, ldS

