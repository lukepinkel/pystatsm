#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 07:57:34 2023

@author: lukepinkel
"""
import re
import numpy as np
import pandas as pd
from .formula_parser import FormulaParser
from ..utilities.indexing_utils import tril_indices
from ..utilities.linalg_operations import  _vech, _vec
from .model_mats import FlattenedIndicatorIndices, BlockFlattenedIndicatorIndices

def _default_sort_key(item):
    match = re.match(r"([a-zA-Z]+)(\d+)", item)
    if match:
        alphabetic_part = match.group(1)
        numeric_part = int(match.group(2))
    else:
        alphabetic_part = item
        numeric_part = 0
    return (alphabetic_part, numeric_part)    


class ParameterTable(object):
    matrix_names = ["L", "B", "F", "P", "a", "b"]
    matrix_order = dict(L=0, B=1, F=2, P=3, a=4, b=5)
    is_symmetric = {0:False, 1:False, 2:True, 3:True, 4:False, 5:False}
    is_vector = {0:False, 1:False, 2:False, 3:False, 4:True, 5:True}
    def __init__(self, formulas, sample_cov=None, sample_mean=None):
        param_df =  pd.DataFrame(FormulaParser.parse_formula(formulas))
        var_names =  FormulaParser.classify_variables(param_df)
        self.param_df = param_df
        self.var_names = var_names
        self.all_var_names = self.var_names["all"]
        ptable = ParameterTable.add_variances(self.param_df, self.var_names, sample_cov=sample_cov)
        ptable =  ParameterTable.add_covariances(ptable, var_names, sample_cov=sample_cov)
        ptable =  ParameterTable.add_means(ptable, var_names, sample_mean=sample_mean)
        lv_order= ParameterTable.default_sort(var_names["lav"], var_names)
        ov_order= ParameterTable.default_sort(var_names["obs"], var_names)

        lav_order =lv_order = dict(zip(lv_order, np.arange(len(lv_order))))
        obs_order = ov_order = dict(zip(ov_order, np.arange(len(ov_order))))
        
        ptable = ParameterTable.assign_matrices(ptable, var_names)
        ptable = ParameterTable.sort_table(ptable, ov_order, lv_order)
        ptable = ParameterTable.index_params(ptable)
        ptable = ParameterTable.add_bounds(ptable)
        self.ptable = ptable
        self.lav_order = lav_order
        self.obs_order = obs_order
        self.free_ix = ptable["free"]!=0
        self.ftable = ptable.loc[self.free_ix]
    
    @property
    def n_free(self):
        return self.ftable.groupby("mat")["ind"].agg("size").values.flatten()
    
    @staticmethod
    def check_missing_variances(param_df, vars_to_check):
        if type(vars_to_check) is not set:
            vars_to_check = set(vars_to_check)
        cov_ix = (param_df["rel"] == "~~")
        sym_ix = (param_df["lhs"] == param_df["rhs"])
        existing_vars = param_df.loc[cov_ix & sym_ix, ["lhs", "rhs"]]
        existing_var_set = set(existing_vars.values.flatten().tolist())
        vars_to_add = vars_to_check - existing_var_set
        return vars_to_add
    
    @staticmethod
    def check_missing_covs(param_df, vars_to_check):
        vars_to_check = np.asarray(sorted(vars_to_check, key=_default_sort_key))
        n = len(vars_to_check)
        lhs_ix, rhs_ix = tril_indices(n, -1)
        lhs, rhs = vars_to_check[lhs_ix], vars_to_check[rhs_ix]
        df = param_df.loc[param_df["rel"]=="~~"]
        pairs_to_add = []
        for x1, x2 in list(zip(lhs, rhs)):
            ix = (((df["lhs"] == x1) & (df["rhs"] == x2)) |
                  ((df["lhs"] == x2) & (df["rhs"] == x1)))
            if not np.any(ix):
                pairs_to_add.append((x1, x2))
        return pairs_to_add
    
    @staticmethod   
    def check_missing_means(param_df, vars_to_check):
        if type(vars_to_check) is not set:
            vars_to_check = set(vars_to_check)
        ix =  (param_df["rel"] == "~") & (param_df["rhs"] == "1")
        existing_vars = param_df.loc[ix, "lhs"]
        existing_var_set = set(existing_vars.values.flatten().tolist())
        vars_to_add = vars_to_check - existing_var_set
        return vars_to_add
    
    @staticmethod
    def add_parameters(params_df, pairs_to_add):
        list_of_param_dicts = params_df.to_dict(orient="records")
        for row in pairs_to_add: 
            row = {**dict(mode=None, fixed=False, fixed_val=np.nan), **row}
            list_of_param_dicts.append(row)
        params_df = pd.DataFrame(list_of_param_dicts)
        return params_df
        
    @staticmethod
    def add_variances(param_df, var_names, sample_cov=None, fix_lv_cov=False):
        all_var_names = var_names["all"]
        vars_to_add = ParameterTable.check_missing_variances(param_df, all_var_names)
        list_of_param_dicts = param_df.to_dict(orient="records")
        vars_to_add = ParameterTable.default_sort(vars_to_add, var_names)
        for var in vars_to_add:
            row = {"lhs":var, "rel":"~~", "rhs":var, "start":1.0, "fixed":False}
            if var in var_names["lox"]:
                row["fixed"] = True
                if sample_cov is not None:
                    row["fixedval"] = sample_cov.loc[var, var]
                else:
                    row["fixedval"] = np.inf
            if var in var_names["nob"] and fix_lv_cov:
                row["fixed"] = True
            list_of_param_dicts.append(row)
        param_df = pd.DataFrame(list_of_param_dicts)
        return param_df
    
    @staticmethod
    def default_sort(subset, var_names):
        g = []
        g.extend(sorted(subset & var_names["obs"] & var_names["ind"], key=_default_sort_key))
        g.extend(sorted(subset & var_names["nob"] & var_names["ind"], key=_default_sort_key))
        g.extend(sorted(subset & var_names["obs"] & var_names["end"], key=_default_sort_key))
        g.extend(sorted(subset & var_names["nob"] - (var_names["nob"] & var_names["ind"]), key=_default_sort_key))
        u = set(g)
        g.extend(sorted(subset - u, key=_default_sort_key))
        return g
    
    @staticmethod
    def add_covariances(param_df, var_names, sample_cov=None, lvx_cov=False, y_cov=True):
        list_of_param_dicts = param_df.to_dict(orient="records")
        lvx_names= var_names["exo"] & var_names["nob"]
        end_vars = ParameterTable.check_missing_covs(param_df, var_names["end"]-var_names["exo"])
        lox_vars = ParameterTable.check_missing_covs(param_df, var_names["lox"])
        lvx_vars = ParameterTable.check_missing_covs(param_df, lvx_names)
    
        for x1, x2 in lox_vars:
            row = {"lhs":x1, "rel":"~~", "rhs":x2, "start":0.0, "fixed":True}
            if sample_cov is not None:
                row["fixedval"] = sample_cov.loc[x1, x2]
            else:
                row["fixedval"] = np.inf
            list_of_param_dicts.append(row)
        if lvx_cov:
            for x1, x2 in lvx_vars:
                row = {"lhs":x1, "rel":"~~", "rhs":x2, "start":0.0, "fixed":False}
                list_of_param_dicts.append(row)
        if y_cov:
            for x1, x2 in end_vars:
                row = {"lhs":x1, "rel":"~~", "rhs":x2, "start":0.0, "fixed":False}
                list_of_param_dicts.append(row)
        param_df = pd.DataFrame(list_of_param_dicts)
        return param_df
    
    @staticmethod
    def add_means(param_df, var_names, sample_mean=None, fix_lv_mean=True):
        list_of_param_dicts = param_df.to_dict(orient="records")
        vars_to_add = ParameterTable.check_missing_means(param_df, var_names["all"])
        for var in vars_to_add:
            row = {"lhs":var, "rel":"~", "rhs":"1", "start":0.0, "fixed":False}
            if fix_lv_mean:
                if var in var_names["nob"]:
                    row["fixed"] = True
                    row["fixedval"] = 0.0
            if var in var_names["lox"]:
                if sample_mean is not None:
                    row["fixed"] = True
                    row["fixedval"] = sample_mean.loc[:, var][0]
            list_of_param_dicts.append(row)
        param_df = pd.DataFrame(list_of_param_dicts)
        return param_df

    
    @staticmethod
    def fix_first(param_df, var_names):
        ind1 = (param_df["rel"]=="=~") & (param_df["lhs"].isin(var_names["nob"]))
        ltable = param_df.ptable[ind1]
        ltable.groupby("lhs")
        for v in var_names["nob"]:
            ix = ltable["lhs"]==v
            if len(ltable.index[ix])>0:
                if ~np.any(ltable.loc[ix, "fixed"]):
                    param_df.loc[ltable.index[ix][0], "fixed"] = True
                    param_df.loc[ltable.index[ix][0], "fixedval"] = 1.0
        return param_df
    
    @staticmethod
    def assign_matrices(param_df, var_names):
        """
        Assign the parameters to the corresponding matrices based on the relation and 
        types of variables involved.
    
        Parameters
        ----------
        param_df : pandas.DataFrame
            The dataframe containing parameters and their details such as 'lhs', 'rhs', 'rel', 
            'mod', 'label', 'fixedval', and 'fixed'.
        var_names : dict
            A dictionary containing the observed variable names under 'obs' key and 
            latent variable names under 'lav' key.
        Returns
        -------
        param_df : pandas.DataFrame
            The updated dataframe where each parameter is assigned to a matrix represented by an 
            integer (0-5) in the 'mat' column. The 'r' and 'c' columns represent the row and 
            column placement of the parameter in its corresponding matrix.
        
        """
        obs_names = sorted(var_names["obs"], key=_default_sort_key)
        lav_names = sorted(var_names["lav"], key=_default_sort_key)
        
        mes = param_df["rel"]== "=~"
        reg = param_df["rel"]== "~"
        cov = param_df["rel"]== "~~" 
        mst = param_df["rhs"]=="1"
        
        ix = {}
        rvl = param_df["rhs"].isin(lav_names)
        rvo = param_df["rhs"].isin(obs_names)
        lvl = param_df["lhs"].isin(lav_names)
        lol = param_df["lhs"].isin(obs_names)

        rvb = rvl & rvo
        ix[0] = mes & ~rvl
        ix[1] = (mes & rvb) | (mes & ~rvo) | reg
        ix[2] = (cov & ~rvl) | (cov & lvl)
        ix[3] = cov & ~lvl
        ix[4] = lol & ~lvl & reg  & mst
        ix[5] = lvl & reg  & mst
        param_df["mat"] = 0
        param_df["mat"] = param_df["mat"].astype(int)
        for i in range(6):
            param_df.loc[ix[i], "mat"] = i
            if i==0:
                param_df.loc[ix[i], "r"] = param_df.loc[ix[i], "rhs"]
                param_df.loc[ix[i], "c"] = param_df.loc[ix[i], "lhs"]
            elif i<4:
                param_df.loc[ix[i], "r"] = param_df.loc[ix[i], "lhs"]
                param_df.loc[ix[i], "c"] = param_df.loc[ix[i], "rhs"]
            else:
                param_df.loc[ix[i], "c"] = param_df.loc[ix[i], "lhs"]
                param_df.loc[ix[i], "r"] = 0

            if i==1:
                j = (param_df["mat"]==1) & (param_df["rel"]=="=~")
                param_df.loc[j, "r"], param_df.loc[j, "c"] = param_df.loc[j, "c"],  param_df.loc[j, "r"]
        return param_df
        
    @staticmethod
    def map_rc(df, rmap, cmap):
        df["r"] = df["r"].map(rmap)
        df["c"] = df["c"].map(cmap)
        return df
    
    @staticmethod
    def sort_flat_representation(df, symmetric=False, vector=False):
        if symmetric:
            df = df.sort_values(["c", "r"])
            ix = df["r"] < df["c"]
            df.loc[ix, "r"], df.loc[ix, "c"] = df.loc[ix, "c"], df.loc[ix, "r"]
            df = df.sort_values(["c", "r"])
        elif vector:
            df = df.sort_values(["c"])
        else:
            df = df.sort_values(["c", "r"])
        return df

    @staticmethod
    def sort_table(param_df, obs_order, lav_order):
        mats = np.unique(param_df["mat"])
        mat_dict = {}
        mat_rc = {0:(obs_order, lav_order), 1:(lav_order, lav_order),
                  2:(lav_order, lav_order), 3:(obs_order, obs_order),
                  4:({"0":0, 0:0}, obs_order), 5:({"0":0, 0:0}, lav_order)}
        for i in sorted(mats):
            mat = param_df.loc[param_df["mat"] == i]
            mat = ParameterTable.map_rc(mat, *mat_rc[i])
            kws = dict(symmetric=ParameterTable.is_symmetric[i],
                       vector=ParameterTable.is_vector[i])
            mat = ParameterTable.sort_flat_representation(mat, **kws)
            mat_dict[i] = mat
        
        param_df = pd.concat([mat_dict[i] for i in sorted(mats)], axis=0)
        param_df = param_df.reset_index(drop=True)
        return param_df
    
    @staticmethod
    def add_bounds(param_dfs):
        ix = (param_dfs["lhs"]==param_dfs["rhs"]) & (param_dfs["rel"]=="~~")
        param_dfs["lb"] = None
        param_dfs.loc[ix, "lb"] = 0
        param_dfs.loc[~ix, "lb"] = None
        param_dfs["ub"]  = None
        return param_dfs
    
    @staticmethod
    def index_params(param_df):
        """
        Assign indices to the parameters in a SEM model.
    
        Parameters
        ----------
        param_df : pd.DataFrame
            The parameter DataFrame containing the 'lhs', 'rhs', 'rel', 'mod', 'label', 'fixedval', 'fixed' columns.
    
        Returns
        -------
        pd.DataFrame
            The updated DataFrame with new 'free' and 'ind' columns representing the indices of free parameters.
    
        Notes
        -----
        The function creates two new columns 'free' and 'ind' in the DataFrame.
        'free' assigns indices to free parameters. If a parameter has a label,
        it shares the same 'free' index with the first parameter that has the same label. 
        'ind' is another index for free parameters, starting from 0, running 
        through indices of free parameters without  accounting for the equality
        constrained ones with duplicate labels.
        """
        param_df["free"] = 0
        ix = ~param_df["fixed"]
        ix2 = ~param_df["label"].isnull()
        links = {}
        eqc = np.unique(param_df.loc[ix2, "label"] )
        for c in eqc:
            ixc = param_df["label"]==c
            index = param_df.loc[ixc, "label"].index.values
            i = np.min(index)
            links[c] = i, index
            ix2[i] = False
        ix = ix & ~ix2    
        n = len(param_df[ix])
        param_df.loc[ix, "free"] = np.arange(1, 1+n)
        for c in eqc:
            i, index = links[c]
            param_df.loc[index, "free"] = param_df.loc[i, "free"]
        param_df["ind"] = 0
        free_ix = param_df["free"]!=0
        param_df.loc[param_df["free"]!=0, "ind"] = np.arange(np.sum(free_ix))
        return param_df
    
    @staticmethod
    def construct_model_mats(param_df, var_names, lv_order, ov_order):
        p = len(var_names["obs"])
        q = len(var_names["lav"])
        lv_names = sorted(lv_order.keys(), key=lambda x: lv_order[x])
        ov_names = sorted(ov_order.keys(), key=lambda x: ov_order[x])
        mat_dims = {0:(p, q), 1:(q, q), 2:(q, q), 3:(p, p), 4:(1, p), 5:(1, q)}
        mat_rows = {0:ov_names, 1:lv_names, 2:lv_names, 3:ov_names, 4:["0"], 5:["0"]}
        mat_cols = {0:lv_names, 1:lv_names, 2:lv_names, 3:ov_names, 4:ov_names, 5:lv_names}
        free_mats, start_mats = {}, {}
        for i in np.unique(param_df["mat"]):
            subtable =  param_df.loc[param_df["mat"]==i]
            free_mat = np.zeros(mat_dims[i])
            start_mat = np.zeros(mat_dims[i])      
            free = subtable.loc[~subtable["fixed"]]
            fixed = subtable.loc[subtable["fixed"]]
            free_mat[(free["r"], free["c"])] = free["free"]
            free_mats[i] = pd.DataFrame(free_mat, index=mat_rows[i], columns=mat_cols[i])
            if i==2:
                start_mat = np.eye(mat_dims[i][0])
            elif i==3:
                start_mat = np.zeros(mat_dims[i])
                inx = [ov_order[x] for x in var_names["obs"]-var_names["lox"]]
                start_mat[inx, inx] = 1.0
            else:
                start_mat = np.zeros(mat_dims[i])
            start_mat[(fixed["r"], fixed["c"])] = fixed["fixedval"]
            start_mats[i] = pd.DataFrame(start_mat, index=mat_rows[i], columns=mat_cols[i])
            mat = start_mats[i].values
            if ParameterTable.is_symmetric[i]:
                v = _vech(mat)
            else:
                v = _vec(mat)
        p_template = []
        lvo = var_names["lvo"]
        for v in lvo:
            if (v in free_mats[0].index) and (v in free_mats[0].columns):
                start_mats[0].loc[v, v] = 1.0
                start_mats[3].loc[v, v] = 0.0
        indexing_objects = {}
        for i in range(6):
            mat = start_mats[i].values
            if ParameterTable.is_symmetric[i]:
                v = _vech(mat)
            else:
                v = _vec(mat)
            p_template.append(v)
            mat = free_mats[i]
            if type(mat) is pd.DataFrame:
                mat = mat.values
            indobj = FlattenedIndicatorIndices(mat, symmetric=ParameterTable.is_symmetric[i])
            indexing_objects[i] = indobj
        p_template = np.concatenate(p_template)
        indexer = BlockFlattenedIndicatorIndices([val for key, val in indexing_objects.items()])
        return p_template, indexer, mat_rows, mat_cols, mat_dims, free_mats

