import numpy as np
import pandas as pd
import scipy as sp

from ..utilities import indexing_utils
from ..utilities.indexing_utils import tril_indices
from ..utilities.linalg_operations import _vech, _vec
from .indexers import FlattenedArray, BlockFlattenedArrays, equality_constraint_mat
from .param_table import ParameterTable

pd.set_option("mode.chained_assignment", None)


class FixedValueManager:

    @staticmethod
    def _get_fixed_mask(param_df, exclude_null=True):
        mask = param_df["fixed"]
        if exclude_null:
            mask = mask & ~param_df["fixedval"].isnull()
        return mask

    @staticmethod
    def _get_cov_mask(param_df, var1, var2):
        var1_var2_mask = (param_df["lhs"] == var1) & (param_df["rhs"] == var2)
        var2_var1_mask = (param_df["lhs"] == var2) & (param_df["rhs"] == var1)
        var1_var2_mask = (var1_var2_mask | var2_var1_mask) & (param_df["rel"] == "~~")
        return var1_var2_mask

    @staticmethod
    def _get_mean_mask(param_df, var):
        return (param_df["lhs"] == var) & (param_df["rhs"] == "1") & (param_df["rel"] == "~")

    @staticmethod
    def _update_value(param_df, mask, value):
        if np.any(mask):
            param_df.loc[mask, "fixedval"] = value
            param_df.loc[mask, "fixed"] = True
        return param_df

    @staticmethod
    def _update_sample_stats(param_df, vars, sample_cov, sample_mean, group_mask=None):
        n_vars = len(vars)
        for j, k in list(zip(*tril_indices(n_vars))):
            varj, vark = vars[j], vars[k]
            varj_vark_mask = FixedValueManager._get_cov_mask(param_df, varj, vark)
            varj_vark_mask = varj_vark_mask & group_mask if group_mask is not None else varj_vark_mask
            param_df = FixedValueManager._update_value(param_df, varj_vark_mask, sample_cov.loc[varj, vark])
            if j == k:
                varj_mask = FixedValueManager._get_mean_mask(param_df, varj)
                varj_mask = varj_mask & group_mask if group_mask is not None else varj_mask
                param_df = FixedValueManager._update_value(param_df, varj_mask, sample_mean.loc[varj])
        return param_df

    def update_sample_stats(self, sample_stats=None):
        sample_stats = self.sample_stats if sample_stats is None else sample_stats
        param_df = self.get_table()
        lav_order = self.get_lav_order()
        var_list = sorted(list(self.var_names["lox"]), key=lambda x: lav_order[x])
        for i in range(self.n_groups):
            param_df = self._update_sample_stats(param_df,
                                                 var_list,
                                                 sample_stats.sample_cov_df[i],
                                                 sample_stats.sample_mean_df[i],
                                                 group_mask=param_df["group"] == i)
        self.set_table(param_df)


class ParameterMapping:

    @staticmethod
    def _make_indexer(param_df, p, q, symmetries, mat_dims):
        indexers = []
        for i in range(6):
            sub_table = param_df.loc[(param_df["mat"] == i) & (param_df["free"] != 0)]
            free_mat = np.zeros(mat_dims[i])
            if len(sub_table) > 0:
                free_mat[sub_table["r"], sub_table["c"]] = sub_table["free"]
            indexers.append(FlattenedArray(free_mat, symmetries[i]))
        indexer = BlockFlattenedArrays(indexers)
        return indexer

    def make_indexer(self):
        param_df = self.get_table()
        group_param_df = param_df.loc[param_df["group"] == 0]
        self.indexer = self._make_indexer(group_param_df, self.n_obs_vars, self.n_lav_vars,
                                          self.is_symmetric, self.mat_dims)

    @staticmethod
    def _make_parameter_templates(param_df, p, q, symmetries, mat_dims, var_names, ov_order, lv_order):
        template = []
        for i in range(6):
            sub_table = param_df.loc[param_df["mat"] == i]
            free_table = sub_table.loc[sub_table["free"] != 0]
            fixed_table = sub_table.loc[sub_table["free"] == 0]
            mat_template = np.zeros(mat_dims[i])
            if i == 0:
                rix = [ov_order[x] for x in var_names["lvo"]]
                cix = [lv_order[x] for x in var_names["lvo"]]
                mat_template[rix, cix] = 1.0
            elif i == 2:
                mat_template = np.eye(mat_dims[i][0])
            elif i == 3:
                inx = [ov_order[x] for x in var_names["onx"]]
                mat_template[inx, inx] = 1.0
            else:
                mat_template = np.zeros(mat_dims[i])

            if len(free_table) > 0:
                mat_template[free_table["r"], free_table["c"]] = free_table["start"]

            if len(fixed_table) > 0:
                mat_template[fixed_table["r"], fixed_table["c"]] = fixed_table["fixedval"]

            if symmetries[i]:
                mat_template = _vech(mat_template)
            else:
                mat_template = _vec(mat_template)
            template.append(mat_template)
        template = np.concatenate(template)
        return template

    def make_parameter_templates(self):
        self.parameter_templates = {}
        param_df = self.get_table()
        for i in range(self.n_groups):
            group_param_df = param_df.loc[param_df["group"] == i]
            self.parameter_templates[i] = self._make_parameter_templates(group_param_df,
                                                                         self.n_obs_vars,
                                                                         self.n_lav_vars,
                                                                         self.is_symmetric,
                                                                         self.mat_dims,
                                                                         self.var_names,
                                                                         self.get_obs_order(),
                                                                         self.get_lav_order())


class ModelBase(ParameterTable, FixedValueManager, ParameterMapping):
    def __init__(self, formula, **kwargs):
        super().__init__(formula, **kwargs)
        self._param_df = self.get_table().copy()
        self._free_df = self.get_free_table().copy()

    def duplicate_parameter_table(self, n_groups=None):
        param_df = self._param_df.copy()
        n_groups = self.n_groups if n_groups is None else n_groups
        keys = range(n_groups)
        param_df = pd.concat([param_df] * n_groups, keys=keys, names=['group'])
        param_df.reset_index(inplace=True)
        label = param_df[["group", "lhs", "rel", "rhs"]].astype(str).agg(' '.join, axis=1)
        param_df.loc[:, "label"] = param_df.loc[:, "label"].fillna(label)
        self.set_table(param_df)

    def reduce_parameters(self, shared=None):
        shared = [True] * 6 if shared is None else shared
        self.shared = shared
        ftable = self.get_free_table()
        ix = ftable["mod"].isnull()
        ftable["label"] = ftable["mod"].copy()
        for i in range(6):
            ix = ftable["mat"] == i
            not_null = ~ftable.loc[ix, "label"].isnull()
            ix1 = ix & not_null
            if self.shared[i]:
                label = ftable[["lhs", "rel", "rhs"]].astype(str).agg(' '.join, axis=1)
            else:
                label = ftable[["group", "lhs", "rel", "rhs"]].astype(str).agg(' '.join, axis=1)
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
