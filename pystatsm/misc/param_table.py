#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 07:57:34 2023

@author: lukepinkel
"""
import re
import numpy as np
import scipy as sp
import pandas as pd

from .formula_parser import FormulaParser
from ..utilities import indexing_utils
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


def equality_constraint_mat(unique_locs):
    n = unique_locs.max()+1
    m = len(unique_locs)
    row = np.arange(m)
    col = unique_locs
    data = np.ones(m)
    arr = sp.sparse.csc_matrix((data, (row, col)), shape=(m, n))
    return arr

class BaseModel(object):
    matrix_names = ["L", "B", "F", "P", "a", "b"]
    matrix_order = dict(L=0, B=1, F=2, P=3, a=4, b=5)
    is_symmetric = {0:False, 1:False, 2:True, 3:True, 4:False, 5:False}
    is_vector = {0:False, 1:False, 2:False, 3:False, 4:True, 5:True}
    
    def __init__(self, formulas, sample_stats=None, var_order=None, 
                 n_groups=1):
        self.n_groups = n_groups
        self.init_formula_parser(formulas)
        self.init_parameter_table(var_order)
        self.extend_param_df()
        if sample_stats is not None:
            self.fix_sample_stats(sample_stats)

    def init_formula_parser(self, formulas):
        self.formula_parser = FormulaParser(formulas)
        self.var_names = self.formula_parser._var_names
        self.all_var_names = self.var_names["all"]
        self.param_df = self.formula_parser._param_df

    def init_parameter_table(self, var_order):
        self.process_parameter_table()
        self.set_ordering(var_order)
        self.sort_and_index_parameters()

    def process_parameter_table(self):
        self.add_variances(self.var_names)
        self.fix_first(self.var_names)
        self.add_covariances(self.var_names)
        self.add_means(self.var_names)

    def set_ordering(self, var_order):
        lav_order = self.default_sort(self.var_names["lav"], self.var_names)
        if var_order is None:
            ov_order = sorted(self.var_names["obs"], key=_default_sort_key)
        else:
            ov_order = sorted(self.var_names["obs"], key=lambda x:var_order[x])
        self.lav_order = dict(zip(lav_order, np.arange(len(lav_order))))
        self.obs_order = dict(zip(ov_order, np.arange(len(ov_order))))

    def sort_and_index_parameters(self):
        self.assign_matrices()
        self.sort_table()
        self.param_df = self.index_params(self.param_df)
        self.param_df = self.add_bounds(self.param_df)
        self.free_ix  = self.param_df["free"] != 0
        self.free_df  = self.param_df.loc[self.free_ix]
    
    def extend_param_df(self):
        self._param_df = self.param_df.copy()
        self._free_df = self.free_df.copy()
        self.param_df = pd.concat([self.param_df] * self.n_groups, keys=range(self.n_groups), names=['group'])
        self.param_df.reset_index(inplace=True)
        self.free_df = pd.concat([self.free_df] * self.n_groups, keys=range(self.n_groups), names=['group'])
        self.free_df.reset_index(inplace=True)
        label = self.free_df[["group", "lhs", "rel","rhs"]].astype(str).agg(' '.join, axis=1)
        self.free_df.loc[:, "label"] = self.free_df.loc[:, "label"].fillna(label)
    @property
    def n_free(self):
        return self.free_df.groupby("mat")["ind"].agg("size").values.flatten()

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
    def add_bounds(param_dfs):
        ix = (param_dfs["lhs"]==param_dfs["rhs"]) & (param_dfs["rel"]=="~~")
        param_dfs["lb"] = None
        param_dfs.loc[ix, "lb"] = 0
        param_dfs.loc[~ix, "lb"] = None
        param_dfs["ub"]  = None
        return param_dfs

    
    @staticmethod
    def index_params(param_df):
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
    
    
    def check_missing_variances(self, vars_to_check):
        param_df = self.param_df
        if type(vars_to_check) is not set:
            vars_to_check = set(vars_to_check)
        cov_ix = (param_df["rel"] == "~~")
        sym_ix = (param_df["lhs"] == param_df["rhs"])
        existing_vars = param_df.loc[cov_ix & sym_ix, ["lhs", "rhs"]]
        existing_var_set = set(existing_vars.values.flatten().tolist())
        vars_to_add = vars_to_check - existing_var_set
        return vars_to_add
    
    def check_missing_covs(self, vars_to_check, param_df=None):
        if param_df is None:
            param_df = self.param_df
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
    
    def check_missing_means(self, vars_to_check):
        param_df = self.param_df
        if type(vars_to_check) is not set:
            vars_to_check = set(vars_to_check)
        ix =  (param_df["rel"] == "~") & (param_df["rhs"] == "1")
        existing_vars = param_df.loc[ix, "lhs"]
        existing_var_set = set(existing_vars.values.flatten().tolist())
        vars_to_add = vars_to_check - existing_var_set
        return vars_to_add

    def add_variances(self, var_names, fix_lv_cov=False):
        param_df = self.param_df
        all_var_names = var_names["all"]
        vars_to_add = self.check_missing_variances(all_var_names)
        list_of_param_dicts = param_df.to_dict(orient="records")
        vars_to_add = self.default_sort(vars_to_add, var_names)
        for var in vars_to_add:
            row = {"lhs":var, "rel":"~~", "rhs":var, "start":1.0, "fixed":False}
            if var in var_names["lox"]:
                row["fixed"] = True
            elif var in var_names["nob"] and fix_lv_cov:
                row["fixed"] = True
            list_of_param_dicts.append(row)
        self.param_df = pd.DataFrame(list_of_param_dicts)
        
    def add_covariances(self, var_names, lvx_cov=True, y_cov=True):
        param_df = self.param_df
        list_of_param_dicts = param_df.to_dict(orient="records")
        lvx_names= var_names["nob"].difference(set.union(var_names["ind"], var_names["end"]))
        end_vars = self.check_missing_covs(var_names["enx"])
        lox_vars = self.check_missing_covs(var_names["lox"])
        lvx_vars = self.check_missing_covs(lvx_names)
    
        for x1, x2 in lox_vars:
            row = {"lhs":x1, "rel":"~~", "rhs":x2, "start":0.0, "fixed":True}
            list_of_param_dicts.append(row)
        if lvx_cov:
            for x1, x2 in lvx_vars:
                row = {"lhs":x1, "rel":"~~", "rhs":x2, "start":0.0, "fixed":False}
                list_of_param_dicts.append(row)
        if y_cov:
            for x1, x2 in end_vars:
                row = {"lhs":x1, "rel":"~~", "rhs":x2, "start":0.0, "fixed":False}
                list_of_param_dicts.append(row)
        self.param_df = pd.DataFrame(list_of_param_dicts)
    
    def add_means(self, var_names, fix_lv_mean=True):
        param_df = self.param_df
        list_of_param_dicts = param_df.to_dict(orient="records")
        vars_to_add = self.check_missing_means(var_names["all"])
        for var in vars_to_add:
            row = {"lhs":var, "rel":"~", "rhs":"1", "start":0.0, "fixed":False}
            if fix_lv_mean:
                if var in var_names["nob"]:
                    row["fixed"] = True
                    row["fixedval"] = 0.0
            if var in var_names["lox"]:
                    row["fixed"] = True
            list_of_param_dicts.append(row)
        self.param_df = pd.DataFrame(list_of_param_dicts)

    
    def fix_first(self, var_names):
        param_df = self.param_df
        ind1 = (param_df["rel"]=="=~") & (param_df["lhs"].isin(var_names["nob"]))
        ltable = param_df.loc[ind1]
        ltable.groupby("lhs")
        for v in var_names["nob"]:
            ix = ltable["lhs"]==v
            if len(ltable.index[ix])>0:
                if ~np.any(ltable.loc[ix, "fixed"]):
                    param_df.loc[ltable.index[ix][0], "fixed"] = True
                    param_df.loc[ltable.index[ix][0], "fixedval"] = 1.0
        self.param_df = param_df
    
    def assign_matrices(self):
        param_df, var_names = self.param_df, self.var_names
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
        self.param_df = param_df
        

    def sort_table(self):
        param_df = self.param_df
        obs_order, lav_order = self.obs_order, self.lav_order
        mats = np.unique(param_df["mat"])
        mat_dict = {}
        mat_rc = {0:(obs_order, lav_order), 1:(lav_order, lav_order),
                  2:(lav_order, lav_order), 3:(obs_order, obs_order),
                  4:({"0":0, 0:0}, obs_order), 5:({"0":0, 0:0}, lav_order)}
        for i in sorted(mats):
            mat = param_df.loc[param_df["mat"] == i]
            mat = BaseModel.map_rc(mat, *mat_rc[i])
            kws = dict(symmetric=BaseModel.is_symmetric[i],
                       vector=BaseModel.is_vector[i])
            mat = BaseModel.sort_flat_representation(mat, **kws)
            mat_dict[i] = mat
        
        param_df = pd.concat([mat_dict[i] for i in sorted(mats)], axis=0)
        self.param_df  = param_df.reset_index(drop=True)
    
    def fix_sample_stats(self, sample_stats):
        param_df = self.param_df
        fixed_df = param_df.loc[param_df["fixed"] & ~param_df["fixedval"].isnull()]
#        end_vars = self.check_missing_covs(self.var_names["enx"],fixed_df )
        lox_vars = self.check_missing_covs(self.var_names["lox"], fixed_df)
        lvx_vars = self.check_missing_covs(self.var_names["lvx"], fixed_df)
        is_cov = param_df["rel"] == "~~"
        is_reg=  param_df["rel"] == "~"
        for i in range(sample_stats.n_groups):
            ix_group = param_df["group"] == i
            covi = sample_stats.sample_cov_df[i]
            meani = sample_stats.sample_mean_df[i]
            for var in self.var_names["lox"]:
                ix_var =((param_df["lhs"] == var) &
                         (param_df["rhs"] == var) &
                         is_cov)
                ix = ix_var & ix_group
                if np.any(ix):
                    param_df.loc[ix, "fixedval"] = covi.loc[var, var]
                    param_df.loc[ix, "fixed"] = True
                ix_var = ((param_df["lhs"] == var) &
                          (param_df["rhs"] == "1") &
                          is_reg)
                ix = ix_var & ix_group
                if np.any(ix):
                    param_df.loc[ix, "fixedval"] = meani.loc[var]
                    param_df.loc[ix, "fixed"] = True
            for x1, x2 in lox_vars:
                ix_var = ((param_df["lhs"] == x1) & (param_df["rhs"] == x2)  |
                         (param_df["rhs"] == x1) & (param_df["lhs"] == x2)) 
                ix = is_cov & ix_var & ix_group
                if np.any(ix):
                    param_df.loc[ix, "fixedval"] = covi.loc[x1, x2]
                    param_df.loc[ix, "fixed"] = True
            for x1, x2 in lvx_vars:
                ix_var = ((param_df["lhs"] == x1) & (param_df["rhs"] == x2)  |
                         (param_df["rhs"] == x1) & (param_df["lhs"] == x2)) 
                # ix = is_cov & ix_var & ix_group
                # if np.any(ix):
                #     param_df.loc[ix, "fixedval"] = covi.loc[x1, x2]
                #     param_df.loc[ix, "fixed"] = True        
        self.param_df = param_df
        self.free_ix  = self.param_df["free"] != 0
        self.free_df  = self.param_df.loc[self.free_ix]
        label = self.free_df[["group", "lhs", "rel","rhs"]].astype(str).agg(' '.join, axis=1)
        self.free_df.loc[:, "label"] = self.free_df.loc[:, "label"].fillna(label)
        self.free_df = self.free_df.reset_index(drop=True)
        
    
            
    def prepare_matrices(self):
        param_df = self.param_df
        lv_order, ov_order = self.lav_order, self.obs_order
        p = len(self.var_names["obs"])
        q = len(self.var_names["lav"])
        lv_names = sorted(lv_order.keys(), key=lambda x: lv_order[x])
        ov_names = sorted(ov_order.keys(), key=lambda x: ov_order[x])
        mat_dims = {0:(p, q), 1:(q, q), 2:(q, q), 3:(p, p), 4:(1, p), 5:(1, q)}
        mat_rows = {0:ov_names, 1:lv_names, 2:lv_names, 3:ov_names, 4:["0"], 5:["0"]}
        mat_cols = {0:lv_names, 1:lv_names, 2:lv_names, 3:ov_names, 4:ov_names, 5:lv_names}
        free_mats, start_mats = {}, {}
        for j in range(self.n_groups):
            free_mats[j], start_mats[j] = {}, {}
            group_ix = param_df["group"]==j
            for i in range(6):
                mat_ix = param_df["mat"]==i
                ix = group_ix & mat_ix
                subtable = param_df.loc[ix]
                free_mat = np.zeros(mat_dims[i])
                start_mat = np.zeros(mat_dims[i])      
                free = subtable.loc[~subtable["fixed"]]
                fixed = subtable.loc[subtable["fixed"]]
                free_mat[(free["r"], free["c"])] = free["free"]
                free_mats[j][i] = pd.DataFrame(free_mat, index=mat_rows[i], columns=mat_cols[i])
                if i==2:
                    start_mat = np.eye(mat_dims[i][0])
                elif i==3:
                    start_mat = np.zeros(mat_dims[i])
                    inx = [ov_order[x] for x in self.var_names["onx"]]
                    start_mat[inx, inx] = 1.0
                else:
                    start_mat = np.zeros(mat_dims[i])
                start_mat[(fixed["r"], fixed["c"])] = fixed["fixedval"]
                start_mats[j][i] = pd.DataFrame(start_mat, index=mat_rows[i], columns=mat_cols[i])
        self.free_mats = free_mats
        self.start_mats = start_mats
        self.mat_rows = mat_rows
        self.mat_cols = mat_cols
        self.mat_dims = mat_dims
    
    def apply_fixed_and_free_values(self):
        lvo = self.var_names["lvo"]
        for v in lvo:
            if (v in self.free_mats[0][0].index) and (v in self.free_mats[0][0].columns):
                for j in range(self.n_groups):
                    self.start_mats[j][0].loc[v, v] = 1.0
                    self.start_mats[j][3].loc[v, v] = 0.0
        
    def flatten_matrices(self):
        self.indexers, self.p_templates, self.free_params = {}, {}, {}
        for j in range(self.n_groups):
            p_template, indices = [], []
            for i in range(6):
                mat = self.start_mats[j][i].values
                if BaseModel.is_symmetric[i]:
                    v = _vech(mat)
                else:
                    v = _vec(mat)
                p_template.append(v)
                mat = self.free_mats[j][i]
                if type(mat) is pd.DataFrame:
                    mat = mat.values
                indices.append(FlattenedIndicatorIndices(mat, symmetric=BaseModel.is_symmetric[i]))
            p_template = np.concatenate(p_template)
            indexer = BlockFlattenedIndicatorIndices(indices)
            self.p_templates[j] = p_template
            self.indexers[j] = indexer
            self.free_params[j] = self.p_templates[j][self.indexers[j].flat_indices]


    def construct_model_mats(self):
        self.prepare_matrices()
        self.apply_fixed_and_free_values()
        self.flatten_matrices()
        
    def reduce_parameters(self, shared):
        self.shared = shared
        ftable = self.free_df
        ix = ftable["mod"].isnull()
        ftable["label"]= ftable["mod"].copy() 
        # ftable["duplicate"] = False
        for i in range(6):
            #Identify parameters (in matrix i) without labels already specified particular in the formula
            ix = ftable["mat"]==i
            not_null = ~ftable.loc[ix, "label"].isnull()
            ix1 = ix & not_null
            #Either add a label that will be unique across groups by adding group id or shared across groups
            if self.shared[i]:
                label = ftable[["lhs", "rel","rhs"]].astype(str).agg(' '.join, axis=1)
            else:
                label = ftable[["group", "lhs", "rel","rhs"]].astype(str).agg(' '.join, axis=1)
                ftable.loc[ix1, "label"] = ftable.loc[ix1, ["group", "label"]].astype(str).agg(' '.join, axis=1)
            ftable.loc[ix, "label"] = ftable.loc[ix, "label"].fillna(label)
        
        unique_values, inverse_mapping, indices = indexing_utils.unique(ftable["label"])
        unique_labels = pd.Series(unique_values)
        label_to_ind = pd.Series(unique_labels.index, index=unique_labels.values)
        self.dfree_dtheta = equality_constraint_mat(indices)
        self._unique_locs, self._first_locs = inverse_mapping, indices
        self.free_df = ftable
        self.free_df["theta_index"] = ftable["label"].map(label_to_ind)
        self.free = self.free_df["start"].fillna(0).values
        self.dfree_dgroup = {}
        self.free_to_group_free = {}
        for i in range(self.n_groups):
            ix = self.free_df.loc[self.free_df["group"]==i, "theta_index"]
            cols = ix
            nrows = len(ix)
            ncols = self.free_df.shape[0]
            rows = np.arange(nrows)
            d = np.ones(nrows)
            self.free_to_group_free[i] = ix             
            self.dfree_dgroup[i] = sp.sparse.csc_array((d, (rows, cols)), shape=(nrows, ncols)).T
        self.n_total_free = len(self.free_df)  
     
 