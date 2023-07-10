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
from ..utilities.linalg_operations import _vech, _vec
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


class BaseModel:
    matrix_names = ["L", "B", "F", "P", "a", "b"]
    matrix_order = {"L": 0, "B": 1, "F": 2, "P": 3, "a": 4, "b": 5}
    is_symmetric = {0: False, 1: False, 2: True, 3: True, 4: False, 5: False}
    is_vector = {0: False, 1: False, 2: False, 3: False, 4: True, 5: True}

    def __init__(self, formulas, sample_stats=None, var_order=None,
                 n_groups=1, process_steps=4, **kwargs):
        self.kwargs = kwargs
        self.n_groups = n_groups
        self.process_steps = process_steps
        if self.process_steps >= 1:
            self.init_formula_parser(formulas)
        if self.process_steps >= 2:
            self.init_parameter_table(var_order)
        if self.process_steps >= 3:
            self.extend_param_df()
        if sample_stats is not None and self.process_steps >= 4:
            self.fix_sample_stats(sample_stats)

    def init_formula_parser(self, formulas):
        self.formula_parser = FormulaParser(formulas)
        self.var_names = self.formula_parser.var_names
        self.all_var_names = self.var_names["all"]
        self.param_df = self.formula_parser.param_df

    def init_parameter_table(self, var_order):
        self.process_parameter_table()
        self.set_ordering(var_order)
        self.sort_and_index_parameters()

    def process_parameter_table(self):
        fix_lv_cov = self.kwargs.get('fix_lv_cov', False)
        self.add_variances(fix_lv_cov=fix_lv_cov)
        self.fix_first()
        lvx_cov = self.kwargs.get('fix_lv_cov', True)
        y_cov = self.kwargs.get('y_cov', True)
        self.add_covariances(lvx_cov=lvx_cov, y_cov=y_cov)
        fix_lv_mean = self.kwargs.get('fix_lv_mean', True)
        self.add_means(fix_lv_mean=fix_lv_mean)

    def update_sample_stats(self, sample_stats):
        self.fix_sample_stats(sample_stats)

    def set_ordering(self, var_order):
        lav_order = self.default_sort(self.var_names["lav"], self.var_names)
        if var_order is None:
            ov_order = sorted(self.var_names["obs"], key=_default_sort_key)
        else:
            ov_order = sorted(
                self.var_names["obs"], key=lambda x: var_order[x])
        self.lav_order = dict(zip(lav_order, np.arange(len(lav_order))))
        self.obs_order = dict(zip(ov_order, np.arange(len(ov_order))))

    def sort_and_index_parameters(self):
        self.assign_matrices()
        self.sort_table()
        self.param_df = self.index_params(self.param_df)
        self.param_df = self.add_bounds(self.param_df)
        self.free_ix = self.param_df["free"] != 0
        self.free_df = self.param_df.loc[self.free_ix]

    def extend_param_df(self):
        self._param_df = self.param_df.copy()
        self._free_df = self.free_df.copy()
        self.param_df = pd.concat(
            [self.param_df] * self.n_groups, keys=range(self.n_groups), names=['group'])
        self.param_df.reset_index(inplace=True)
        self.free_df = pd.concat(
            [self.free_df] * self.n_groups, keys=range(self.n_groups), names=['group'])
        self.free_df.reset_index(inplace=True)
        label = self.free_df[["group", "lhs", "rel", "rhs"]].astype(
            str).agg(' '.join, axis=1)
        self.free_df.loc[:, "label"] = self.free_df.loc[:,
                                                        "label"].fillna(label)

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
        g.extend(
            sorted(subset & var_names["obs"] & var_names["ind"],
                   key=_default_sort_key))
        g.extend(
            sorted(subset & var_names["nob"] & var_names["ind"],
                   key=_default_sort_key))
        g.extend(
            sorted(subset & var_names["obs"] & var_names["end"],
                   key=_default_sort_key))
        g.extend(sorted(
            subset & var_names["nob"] - (var_names["nob"] & var_names["ind"]),
            key=_default_sort_key))
        u = set(g)
        g.extend(sorted(subset - u, key=_default_sort_key))
        return g

    @staticmethod
    def add_bounds(param_dfs):
        ix = (param_dfs["lhs"] == param_dfs["rhs"]) & (
            param_dfs["rel"] == "~~")
        param_dfs["lb"] = None
        param_dfs.loc[ix, "lb"] = 0
        param_dfs.loc[~ix, "lb"] = None
        param_dfs["ub"] = None
        return param_dfs

    @staticmethod
    def index_params(param_df):
        param_df["free"] = 0
        ix = ~param_df["fixed"]
        ix2 = ~param_df["label"].isnull()
        links = {}
        eqc = np.unique(param_df.loc[ix2, "label"])
        for c in eqc:
            ixc = param_df["label"] == c
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
        free_ix = param_df["free"] != 0
        param_df.loc[param_df["free"] != 0, "ind"] = np.arange(np.sum(free_ix))
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
        vars_to_check = np.asarray(
            sorted(vars_to_check, key=_default_sort_key))
        n = len(vars_to_check)
        lhs_ix, rhs_ix = tril_indices(n, -1)
        lhs, rhs = vars_to_check[lhs_ix], vars_to_check[rhs_ix]
        df = param_df.loc[param_df["rel"] == "~~"]
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
        ix = (param_df["rel"] == "~") & (param_df["rhs"] == "1")
        existing_vars = param_df.loc[ix, "lhs"]
        existing_var_set = set(existing_vars.values.flatten().tolist())
        vars_to_add = vars_to_check - existing_var_set
        return vars_to_add

    def add_variances(self, fix_lv_cov=False):
        var_names, param_df = self.var_names, self.param_df
        all_var_names = var_names["all"]
        vars_to_add = self.check_missing_variances(all_var_names)
        list_of_param_dicts = param_df.to_dict(orient="records")
        vars_to_add = self.default_sort(vars_to_add, var_names)
        for var in vars_to_add:
            row = {"lhs": var, "rel": "~~", "rhs": var,
                   "start": 1.0, "fixed": False}
            if var in var_names["lox"]:
                row["fixed"] = True
            elif var in var_names["nob"] and fix_lv_cov:
                row["fixed"] = True
            list_of_param_dicts.append(row)
        self.param_df = pd.DataFrame(list_of_param_dicts)

    def add_covariances(self, lvx_cov=True, y_cov=True):
        var_names, param_df = self.var_names, self.param_df
        list_of_param_dicts = param_df.to_dict(orient="records")
        lvx_names = var_names["nob"].difference(
            set.union(var_names["ind"], var_names["end"]))
        end_vars = self.check_missing_covs(var_names["enx"])
        lox_vars = self.check_missing_covs(var_names["lox"])
        lvx_vars = self.check_missing_covs(lvx_names)

        for x1, x2 in lox_vars:
            row = {"lhs": x1, "rel": "~~", "rhs": x2,
                   "start": 0.0, "fixed": True}
            list_of_param_dicts.append(row)
        if lvx_cov:
            for x1, x2 in lvx_vars:
                row = {"lhs": x1, "rel": "~~", "rhs": x2,
                       "start": 0.0, "fixed": False}
                list_of_param_dicts.append(row)
        if y_cov:
            for x1, x2 in end_vars:
                row = {"lhs": x1, "rel": "~~", "rhs": x2,
                       "start": 0.0, "fixed": False}
                list_of_param_dicts.append(row)
        self.param_df = pd.DataFrame(list_of_param_dicts)

    def add_means(self, fix_lv_mean=True):
        var_names, param_df = self.var_names, self.param_df
        list_of_param_dicts = param_df.to_dict(orient="records")
        vars_to_add = self.check_missing_means(var_names["all"])
        for var in vars_to_add:
            row = {"lhs": var, "rel": "~", "rhs": "1",
                   "start": 0.0, "fixed": False}
            if fix_lv_mean:
                if var in var_names["nob"]:
                    row["fixed"] = True
                    row["fixedval"] = 0.0
            if var in var_names["lox"]:
                row["fixed"] = True
            list_of_param_dicts.append(row)
        self.param_df = pd.DataFrame(list_of_param_dicts)

    def fix_first(self):
        var_names, param_df = self.var_names, self.param_df
        ind1 = (param_df["rel"] == "=~") & (
            param_df["lhs"].isin(var_names["nob"]))
        ltable = param_df.loc[ind1]
        ltable.groupby("lhs")
        for v in var_names["nob"]:
            ix = ltable["lhs"] == v
            if len(ltable.index[ix]) > 0:
                if ~np.any(ltable.loc[ix, "fixed"]):
                    param_df.loc[ltable.index[ix][0], "fixed"] = True
                    param_df.loc[ltable.index[ix][0], "fixedval"] = 1.0
        self.param_df = param_df

    @property
    def masks(self):
        param_df, var_names = self.param_df, self.var_names
        obs_names = sorted(var_names["obs"], key=_default_sort_key)
        lav_names = sorted(var_names["lav"], key=_default_sort_key)

        masks = {}
        masks['mes'] = param_df["rel"] == "=~"
        masks['reg'] = param_df["rel"] == "~"
        masks['cov'] = param_df["rel"] == "~~"
        masks['mst'] = param_df["rhs"] == "1"
        masks['rvl'] = param_df["rhs"].isin(lav_names)
        masks['rvo'] = param_df["rhs"].isin(obs_names)
        masks['lvl'] = param_df["lhs"].isin(lav_names)
        masks['lol'] = param_df["lhs"].isin(obs_names)
        masks['rvb'] = masks['rvl'] & masks['rvo']

        return masks

    def assign_matrices(self):
        param_df = self.param_df
        masks = self.masks

        ix = {}
        ix[0] = masks['mes'] & ~masks['rvl']
        ix[1] = (masks['mes'] & masks['rvb']) | (
            masks['mes'] & ~masks['rvo']) | masks['reg']
        ix[2] = (masks['cov'] & ~masks['rvl']) | (masks['cov'] & masks['lvl'])
        ix[3] = masks['cov'] & ~masks['lvl']
        ix[4] = masks['lol'] & ~masks['lvl'] & masks['reg'] & masks['mst']
        ix[5] = masks['lvl'] & masks['reg'] & masks['mst']

        param_df["mat"] = 0
        param_df["mat"] = param_df["mat"].astype(int)

        for i in range(6):
            param_df.loc[ix[i], "mat"] = i
            if i == 0:
                param_df.loc[ix[i], "r"] = param_df.loc[ix[i], "rhs"]
                param_df.loc[ix[i], "c"] = param_df.loc[ix[i], "lhs"]
            elif i < 4:
                param_df.loc[ix[i], "r"] = param_df.loc[ix[i], "lhs"]
                param_df.loc[ix[i], "c"] = param_df.loc[ix[i], "rhs"]
            else:
                param_df.loc[ix[i], "c"] = param_df.loc[ix[i], "lhs"]
                param_df.loc[ix[i], "r"] = 0

            if i == 1:
                j = (param_df["mat"] == 1) & (param_df["rel"] == "=~")
                param_df.loc[j, "r"], param_df.loc[j, "c"] \
                    = param_df.loc[j, "c"],  param_df.loc[j, "r"]
        self.param_df = param_df

    def sort_table(self):
        param_df = self.param_df
        obs_order, lav_order = self.obs_order, self.lav_order
        mats = np.unique(param_df["mat"])
        mat_dict = {}
        mat_rc = {0: (obs_order, lav_order), 1: (lav_order, lav_order),
                  2: (lav_order, lav_order), 3: (obs_order, obs_order),
                  4: ({"0": 0, 0: 0}, obs_order), 5: ({"0": 0, 0: 0}, lav_order)}
        for i in sorted(mats):
            mat = param_df.loc[param_df["mat"] == i]
            mat = BaseModel.map_rc(mat, *mat_rc[i])
            kws = {"symmetric": BaseModel.is_symmetric[i],
                   "vector": BaseModel.is_vector[i]}
            mat = BaseModel.sort_flat_representation(mat, **kws)
            mat_dict[i] = mat

        param_df = pd.concat([mat_dict[i] for i in sorted(mats)], axis=0)
        self.param_df = param_df.reset_index(drop=True)

    def fix_sample_stats(self, sample_stats):
        param_df = self.param_df
        fixed_df = param_df.loc[param_df["fixed"]
                                & ~param_df["fixedval"].isnull()]
        lox_vars = self.check_missing_covs(self.var_names["lox"], fixed_df)
        lvx_vars = self.check_missing_covs(self.var_names["lvx"], fixed_df)
        is_cov = param_df["rel"] == "~~"
        is_reg = param_df["rel"] == "~"
        for i in range(sample_stats.n_groups):
            ix_group = param_df["group"] == i
            covi = sample_stats.sample_cov_df[i]
            meani = sample_stats.sample_mean_df[i]
            for var in self.var_names["lox"]:
                ix_var = ((param_df["lhs"] == var) &
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
                ix_var = ((param_df["lhs"] == x1) & (param_df["rhs"] == x2) |
                          (param_df["rhs"] == x1) & (param_df["lhs"] == x2))
                ix = is_cov & ix_var & ix_group
                if np.any(ix):
                    param_df.loc[ix, "fixedval"] = covi.loc[x1, x2]
                    param_df.loc[ix, "fixed"] = True
            for x1, x2 in lvx_vars:
                ix_var = ((param_df["lhs"] == x1) & (param_df["rhs"] == x2) |
                          (param_df["rhs"] == x1) & (param_df["lhs"] == x2))
        self.param_df = param_df
        self.free_ix = self.param_df["free"] != 0
        self.free_df = self.param_df.loc[self.free_ix]
        label = self.free_df[["group", "lhs", "rel", "rhs"]].astype(
            str).agg(' '.join, axis=1)
        self.free_df.loc[:, "label"] = self.free_df.loc[:,
                                                        "label"].fillna(label)
        self.free_df = self.free_df.reset_index(drop=True)

    def prepare_matrices(self):
        param_df = self.param_df
        lv_order, ov_order = self.lav_order, self.obs_order
        p = len(self.var_names["obs"])
        q = len(self.var_names["lav"])
        lv_names = sorted(lv_order.keys(), key=lambda x: lv_order[x])
        ov_names = sorted(ov_order.keys(), key=lambda x: ov_order[x])
        mat_dims = {0: (p, q), 1: (q, q), 2: (q, q),
                    3: (p, p), 4: (1, p), 5: (1, q)}
        mat_rows = {0: ov_names, 1: lv_names,
                    2: lv_names, 3: ov_names, 4: ["0"], 5: ["0"]}
        mat_cols = {0: lv_names, 1: lv_names, 2: lv_names,
                    3: ov_names, 4: ov_names, 5: lv_names}
        free_mats, start_mats = {}, {}
        for j in range(self.n_groups):
            free_mats[j], start_mats[j] = {}, {}
            group_ix = param_df["group"] == j
            for i in range(6):
                mat_ix = param_df["mat"] == i
                ix = group_ix & mat_ix
                subtable = param_df.loc[ix]
                free_mat = np.zeros(mat_dims[i])
                start_mat = np.zeros(mat_dims[i])
                free = subtable.loc[~subtable["fixed"]]
                fixed = subtable.loc[subtable["fixed"]]
                free_mat[(free["r"], free["c"])] = free["free"]
                free_mats[j][i] = pd.DataFrame(
                    free_mat, index=mat_rows[i], columns=mat_cols[i])
                if i == 2:
                    start_mat = np.eye(mat_dims[i][0])
                elif i == 3:
                    start_mat = np.zeros(mat_dims[i])
                    inx = [ov_order[x] for x in self.var_names["onx"]]
                    start_mat[inx, inx] = 1.0
                else:
                    start_mat = np.zeros(mat_dims[i])
                start_mat[(fixed["r"], fixed["c"])] = fixed["fixedval"]
                start_mats[j][i] = pd.DataFrame(
                    start_mat, index=mat_rows[i], columns=mat_cols[i])
        self.free_mats = free_mats
        self.start_mats = start_mats
        self.mat_rows = mat_rows
        self.mat_cols = mat_cols
        self.mat_dims = mat_dims

    def construct_model_mats(self):
        self.prepare_matrices()
        self.apply_fixed_and_free_values()
        self.flatten_matrices()

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
                indices.append(FlattenedIndicatorIndices(
                    mat, symmetric=BaseModel.is_symmetric[i]))
            p_template = np.concatenate(p_template)
            indexer = BlockFlattenedIndicatorIndices(indices)
            self.p_templates[j] = p_template
            self.indexers[j] = indexer
            self.free_params[j] = self.p_templates[j][self.indexers[j].flat_indices]

    def reduce_parameters(self, shared=None):
        shared = [True]*6 if shared is None else shared
        self.shared = shared
        ftable = self.free_df
        ix = ftable["mod"].isnull()
        ftable["label"] = ftable["mod"].copy()
        # ftable["duplicate"] = False
        for i in range(6):
            # Identify parameters (in matrix i) without labels already
            # specified particular in the formula
            ix = ftable["mat"] == i
            not_null = ~ftable.loc[ix, "label"].isnull()
            ix1 = ix & not_null
            # Either add a label that will be unique across groups by
            # adding group id or shared across groups
            if self.shared[i]:
                label = ftable[["lhs", "rel", "rhs"]].astype(
                    str).agg(' '.join, axis=1)
            else:
                label = ftable[["group", "lhs", "rel", "rhs"]].astype(
                    str).agg(' '.join, axis=1)
                ftable.loc[ix1, "label"] = ftable.loc[ix1, [
                    "group", "label"]].astype(str).agg(' '.join, axis=1)
            ftable.loc[ix, "label"] = ftable.loc[ix, "label"].fillna(label)

        unique_values, inverse_mapping, indices = indexing_utils.unique(
            ftable["label"])
        unique_labels = pd.Series(unique_values)
        label_to_ind = pd.Series(
            unique_labels.index, index=unique_labels.values)
        self.dfree_dtheta = equality_constraint_mat(indices)
        self._unique_locs, self._first_locs = inverse_mapping, indices
        self.free_df = ftable
        self.free_df["theta_index"] = ftable["label"].map(label_to_ind)
        self.free = self.free_df["start"].fillna(0).values
        self.dfree_dgroup = {}
        self.free_to_group_free = {}
        for i in range(self.n_groups):
            ix = self.free_df.loc[self.free_df["group"] == i, "theta_index"]
            cols = ix
            nrows = len(ix)
            ncols = self.free_df.shape[0]
            rows = np.arange(nrows)
            d = np.ones(nrows)
            self.free_to_group_free[i] = ix
            self.dfree_dgroup[i] = sp.sparse.csc_array(
                (d, (rows, cols)), shape=(nrows, ncols)).T
        self.n_total_free = len(self.free_df)


docstrings = {
    'class':
    """
    BaseModel class provides a base for defining structural equation models.

    Attributes:
        matrix_names (list): The names of the matrices involved in the model.
        matrix_order (dict): The order of the matrices.
        is_symmetric (dict): Specifies if a matrix is symmetric or not.
        is_vector (dict): Specifies if a matrix is a vector or not.
        kwargs (dict): Additional keyword arguments.

    Typical Usage and Sequence of Operations:

    1. Initialize BaseModel object with specific inputs.
        - Initializes some constants and assigns input parameters to object properties.
        - Depending on the process_steps value, it performs the following operations:
            - If process_steps >= 1, it calls self.init_formula_parser(formulas).
            - If process_steps >= 2, it calls self.init_parameter_table(var_order).
            - If process_steps >= 3, it calls self.extend_param_df().
            - If sample_stats is not None and process_steps >= 4, it calls self.fix_sample_stats(sample_stats).

    Detailed Sequence of Operations:

    1. __init__():
        - Assigns input parameters to object properties.
        - Depending on the process_steps value, it calls the following methods:
            - self.init_formula_parser(formulas)
            - self.init_parameter_table(var_order)
            - self.extend_param_df()
            - self.fix_sample_stats(sample_stats)

    2. init_formula_parser(formulas):
        - Initializes the formula parser with the given formulas.
        - Assigns the formula parser, variable names, all variable names, and parameter dataframe to the object properties.

    3. init_parameter_table(var_order):
        - Calls the following methods in order:
            - self.process_parameter_table()
            - self.set_ordering(var_order)
            - self.sort_and_index_parameters()

    4. process_parameter_table():
        - Retrieves fix_lv_cov, y_cov, and fix_lv_mean from kwargs.
        - Calls the following methods in order:
            - self.add_variances(fix_lv_cov)
            - self.fix_first()
            - self.add_covariances(lvx_cov, y_cov)
            - self.add_means(fix_lv_mean)

    5. set_ordering(var_order):
        - Sets the order of latent and observed variables based on var_order.
        - Assigns the order of latent and observed variables to the object properties.

    6. sort_and_index_parameters():
        - Calls the following methods in order:
            - self.assign_matrices()
            - self.sort_table()
            - self.param_df = self.index_params(self.param_df)
            - self.param_df = self.add_bounds(self.param_df)
        - Assigns free indexes and free dataframe to the object properties.

    7. extend_param_df():
        - Makes copies of the parameter and free dataframes.
        - Concatenates and resets the original parameter and free dataframes.
        - Adds labels to the free dataframe.

    8. fix_sample_stats(sample_stats):
        - Checks missing covariances and fixes them.
        - Updates the parameter dataframe, free indexes, and free dataframe.
    """,
    'init':
        """
        Initializes the BaseModel object.

        Parameters
        ----------
        formulas : str
            Set of formulas to parse.
        sample_stats : DataFrame or None, optional
            Sample statistics, by default None. If None, no statistics are used.
        var_order : dict or None, optional
            Ordering for the variables. If None, default ordering is used, by default None.
        n_groups : int, optional
            The number of groups in the data, by default 1.
        process_steps : int, optional
            Number of steps to process for model initialization, by default 4.
        **kwargs : dict
            Arbitrary keyword arguments.
        """,
    'add_variances':
        """
        Augments the parameter DataFrame with variances for certain variables.

        This method identifies variables that are missing variances from the
        current parameter DataFrame and adds them. The variables are first
        sorted using the default_sort method. For each variable,  a new row is
        added to the parameter DataFrame with the left-hand side (lhs) and
        the right-hand side (rhs)  set to the variable itself, the relation
        (rel) set to "~~" (indicating a variance), and the starting
        value (start) set to 1.0.

        If the variable is in the set of observed exogenous variables
        (var_names["lox"]),  its variance is fixed, i.e., its "fixed" field is
        set to True. Additionally, if the variable is  a non-observed variable
        (var_names["nob"]) and the fix_lv_cov flag is True, its variance is
        also fixed.  For all other variables, the "fixed" field is set to
        False, implying the variance can be estimated. At the end, the list of
        parameter dictionaries is converted back to a DataFrame and replaces
        the current parameter DataFrame.

        Parameters
        ----------
        fix_lv_cov : bool, optional
            If True, sets the variances of the structural/latent model as fixed.
            Default is False.

        Notes
        -----
        - Uses the following class attributes: `var_names`, `param_df`
        - Modifies the class attribute: `param_df`
        - Uses the following class methods `check_missing_variances`, `default_sort`
        """,
    'fix_first':
        """
        Fixes the loading of the first observed variable onto each latent variable to 1.0.

        This method operates on the parameter DataFrame and modifies it
        in-place. The logic is as follows:
            For each non-observed variable (latent variable, var_names["nob"]),
            it checks if there is any  row in the DataFrame where the
            relationship (rel) is "=~" (indicating a loading), the left-hand
            side  (lhs) is the latent variable, and the parameter is not yet
            fixed. If such a row exists, the "fixed"  field of the first such
            row is set to True, and the "fixedval" field is set to 1.0. This
            implies that  the loading of the first observed variable onto this
            latent variable is fixed to 1.0.

        This process is repeated for all latent variables. After modifying,
        the updated DataFrame replaces the original parameter DataFrame.
        Notes
        -----
        - Uses the following class attributes: `var_names`, `param_df`
        - Modifies the class attribute: `param_df`
        """,
    'add_covariances':
        """
        Add missing covariances to the parameter DataFrame (`param_df`) for
        different variable types.

        This method checks for missing covariances among observable exogenous
        variables (`lox`), latent exogenous variables  (`lvx`), and endogenous
        variables (`enx`). For each pair of variables with missing covariance,
        it appends a new row to  the `param_df`. Covariances of exogenous
        variables (`lox`) are fixed to zero by default. Covariances of latent
        exogenous variables (`lvx`) and endogenous variables (`enx`) are added
        with initial values of zero and can be estimated from the data
        depending on the `lvx_cov` and `y_cov` parameters.

        Parameters
        ----------
        lvx_cov : bool, default True
            If True, covariance for latent exogenous variables (`lvx`) will be
            added and estimated from the data.
            If False, the covariances are assumed to be zero and will not be
            estimated from the data.
        y_cov : bool, default True
            If True, covariance for endogenous variables (`enx`) will be added
            and estimated from the data.
            If False, the covariances are assumed to be zero and will not be
            estimated from the data.

        Notes
        -----
        - Uses the following class attributes: `var_names`, `param_df`
        - Modifies the class attribute: `param_df`
        - Uses the following class methods: `check_missing_covs`
        """,
    'add_means':
        """
        Add missing means to the parameter DataFrame (`param_df`) for different
        types of variables.

        This method checks for missing means among observed exogenous (`lox`),
        endogenous (`enx`), and disturbance (`dis`)  variables. For each
        variable with a missing mean, it appends a new row to `param_df` with a
        mean of zero, which can be  estimated from the data. Means of latent
        exogenous variables (`lvx`) are fixed to zero by default and are not
        estimated from the data.

        Notes
        -----
        - Uses the following class attributes: `var_names`, `param_df`
        - Modifies the class attribute: `param_df`
        - Uses the following class methods: `check_missing_means`, `default_sort`
        """,
    'masks':
        """
        Create boolean masks for different types of model parameters.

        This method creates and assigns four different boolean masks
        (`mask_1`, `mask_2`, `mask_3`, `mask_4`) based on the  relations in
        `param_df`. These masks are used for various operations like setting
        values and checking conditions in the parameter DataFrame.

        The masks are as follows:
        - `mask_1`: True for parameters where relation is '=~' or '~~'
        - `mask_2`: True for parameters where relation is '~1' or '1~'
        - `mask_3`: True for parameters where relation is either '=~' or '~1'
                    or '1~'
        - `mask_4`: True for parameters where relation is '~~'

        Notes
        -----
        - Uses the following class attribute: `param_df`
        - Modifies the following class attributes: `mask_1`, `mask_2`,
        `mask_3`, `mask_4`
        """,
    'assign_matrices':
        """
        Assign matrix indices and related attributes to each row in the
        `param_df`.

        This method leverages masks to categorize different types of model
        parameters and assign them to specific matrices represented by the
        'mat' attribute. It also assigns row ('r')  and column ('c') indices
        for each parameter's placement within its matrix.

        Notes
        -----
        - Uses the following class attributes: `param_df`, `masks`
        - Modifies the following class attribute: `param_df`
        """,
    'sort_table':
        """
        Sort `param_df` based on a predefined order of matrices and their
        elements. This method sorts the parameters based on their assigned
        matrices ('mat') and  then their row ('r') and column ('c') indices.
        It ensures the parameters  are ordered according to their placement in
        the model.

        Notes
        -----
        - Uses the following class attributes: `param_df`, `obs_order`,
         `lav_order`
        - Modifies the following class attribute: `param_df`
        """,
    'index_params':
        """
        Assign indices to each free parameter in `param_df`.

        This method assigns a unique index to each non-fixed parameter in the
        `param_df`  under 'free' attribute. It handles parameters that are
        equal to each other by assigning them the same index. It also sets an
        'ind' attribute for each free parameter.

        Notes
        -----
        - Uses the following class attribute: `param_df`
        - Modifies the following class attribute: `param_df`
        """,
    'extend_param_df':
        """
        Extend `param_df` and `free_df` for each group in a multi-group analysis.

        This method duplicates `param_df` and `free_df` for each group in a
        multi-group analysis.  It also fills in missing 'label' entries in
        `free_df`.

        Parameters
        ----------
        values : np.ndarray
            An array of parameter values to be assigned to 'value' column of
            `param_df`.

        Notes
        -----
        - Uses the following class attributes: `param_df`, `free_df`, `n_groups`
        - Modifies the following class attributes: `param_df`, `free_df`,
        `_param_df`, `_free_df`
        """,
    'default_sort':
        """
        Returns a list of variable names sorted by default rules.

        Parameters
        ----------
        subset : set
            A set of variable names to sort.
        var_names : dict
            A dictionary of variable names classified by their roles in the model.

        Returns
        -------
        g : list
            The sorted list of variable names.
        """


}
BaseModel.__init__.__doc__ = docstrings['init']
BaseModel.add_variances.__doc__ = docstrings['add_variances']
BaseModel.fix_first.__doc__ = docstrings['fix_first']
BaseModel.add_covariances.__doc__ = docstrings['add_covariances']
BaseModel.add_means.__doc__ = docstrings['add_means']
BaseModel.masks.__doc__ = docstrings['masks']
BaseModel.assign_matrices.__doc__ = docstrings['assign_matrices']
BaseModel.sort_table.__doc__ = docstrings['sort_table']
BaseModel.index_params.__doc__ = docstrings['index_params']
BaseModel.extend_param_df.__doc__ = docstrings['extend_param_df']
BaseModel.default_sort.__doc__ = docstrings['default_sort']
BaseModel.__doc__ = docstrings['class']
