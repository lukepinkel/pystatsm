import numpy as np
import pandas as pd

from .sort_key import _default_sort_key
from ..utilities.indexing_utils import tril_indices


class ModelBuilder:
    """
    Class that adds default parameters to a parameter table.
    """

    def add_default_params(self, **kwargs):
        """
        Adds variances, covariances, and means to parameter table and fixes first latent indicator
        """
        self.add_variances(fix_lv_cov=kwargs.get('fix_lv_var', False))
        self.fix_first()
        self.add_covariances(lvx_cov=~kwargs.get('fix_lv_cov', False), y_cov=~kwargs.get('y_cov', True))
        self.add_means(fix_lv_mean=kwargs.get('fix_lv_mean', True))
        self.add_latent_dummies()

    def check_missing_variances(self, vars_to_check):
        """
        Checks for missing variances in the parameter table.

        Parameters
        ----------
        vars_to_check : set
            set of variables to check for variances

        Returns
        -------
        vars_to_add : set
            set of variables to add to parameter table
        """
        param_df = self.param_df
        if type(vars_to_check) is not set:
            vars_to_check = set(vars_to_check)
        cov_ix = (param_df["rel"] == "~~")
        sym_ix = (param_df["lhs"] == param_df["rhs"])
        existing_vars = param_df.loc[cov_ix & sym_ix, ["lhs", "rhs"]]
        existing_var_set = set(existing_vars.values.flatten().tolist())
        vars_to_add = vars_to_check - existing_var_set
        return vars_to_add

    def add_variances(self, fix_lv_cov=False):
        """
        Adds variances to parameter table.
        """
        var_names, param_df = self.var_names, self.param_df
        vars_to_add = self.check_missing_variances(var_names["all"])
        list_of_param_dicts = param_df.to_dict(orient="records")
        for var in vars_to_add:
            row = {"lhs": var, "rel": "~~", "rhs": var, "start": 1.0, "dummy": False}
            if var in var_names["lox"]:
                row["fixed"] = True
            elif var in var_names["nob"] and fix_lv_cov:
                row["fixed"] = True
            else:
                row["fixed"] = False
            list_of_param_dicts.append(row)
        self.param_df = pd.DataFrame(list_of_param_dicts)

    def check_missing_covs(self, vars_to_check, param_df_to_check=None):
        """
        Checks for missing covariances in the parameter table.
        Parameters
        ----------
        vars_to_check
        param_df_to_check

        Returns
        -------
        pairs_to_add : list
        """
        param_df = self.param_df if param_df_to_check is None else param_df_to_check
        vars_to_check = np.asarray(sorted(vars_to_check, key=_default_sort_key))
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

    def add_covariances(self, lvx_cov=True, y_cov=True):
        """
        Parameters
        ----------
        lvx_cov : bool
            whether to add covariances between latent exogenous variables
        y_cov : bool
            whether to add covariances between observed exogenous variables

        Returns
        -------
        None
        """
        var_names, param_df = self.var_names, self.param_df
        list_of_param_dicts = param_df.to_dict(orient="records")
        end_vars = self.check_missing_covs(var_names["enx"])
        lox_vars = self.check_missing_covs(var_names["lox"])
        lvx_vars = self.check_missing_covs(var_names["lvx"])
        for x1, x2 in lox_vars:
            row = {"lhs": x1, "rel": "~~", "rhs": x2,
                   "start": 0.0, "fixed": True, "dummy": False}
            list_of_param_dicts.append(row)
        if lvx_cov:
            for x1, x2 in lvx_vars:
                row = {"lhs": x1, "rel": "~~", "rhs": x2,
                       "start": 0.0, "fixed": False, "dummy": False}
                list_of_param_dicts.append(row)
        if y_cov:
            for x1, x2 in end_vars:
                row = {"lhs": x1, "rel": "~~", "rhs": x2,
                       "start": 0.0, "fixed": False, "dummy": False}
                list_of_param_dicts.append(row)
        self.param_df = pd.DataFrame(list_of_param_dicts)

    def check_missing_means(self, vars_to_check, param_df_to_check=None):
        """
        Checks for missing means in the parameter table.

        Parameters
        ----------
        vars_to_check : set
            set of variables to check for means
        param_df_to_check : pd.DataFrame
            parameter table to check for means

        Returns
        -------
        vars_to_add : set
            set of variables to add to parameter table
        """
        param_df = self.param_df if param_df_to_check is None else param_df_to_check
        if type(vars_to_check) is not set:
            vars_to_check = set(vars_to_check)
        ix = (param_df["rel"] == "~") & (param_df["rhs"] == "1")
        existing_vars = param_df.loc[ix, "lhs"]
        existing_var_set = set(existing_vars.values.flatten().tolist())
        vars_to_add = vars_to_check - existing_var_set
        return vars_to_add

    def add_means(self, fix_lv_mean=True):
        """
        Adds means to parameter table.

        Parameters
        ----------
        fix_lv_mean : bool
            whether to fix means of latent variables to zero

        Returns
        -------
        None
        """
        var_names, param_df = self.var_names, self.param_df
        list_of_param_dicts = param_df.to_dict(orient="records")
        vars_to_add = self.check_missing_means(var_names["all"])
        for var in vars_to_add:
            row = {"lhs": var, "rel": "~", "rhs": "1",
                   "start": 0.0, "fixed": False, "dummy": False}
            if fix_lv_mean:
                if var in var_names["nob"]:
                    row["fixed"] = True
                    row["fixedval"] = 0.0
            if var in var_names["lox"]:
                row["fixed"] = True
            list_of_param_dicts.append(row)
        self.param_df = pd.DataFrame(list_of_param_dicts)

    def check_latent_dummies(self, vars_to_check, param_df_to_check=None):
        param_df = self.param_df if param_df_to_check is None else param_df_to_check
        lav_names, obs_names = self.var_names["lav"], self.var_names["obs"]
        mes = param_df["rel"] == "=~"
        rvo = param_df["rhs"].isin(obs_names)
        rvb = param_df["rhs"].isin(lav_names) & param_df["rhs"].isin(obs_names)
        pars_to_exclude = (mes & rvb) | (mes & ~rvo) | param_df["rel"] == "~"
        vars_to_add = set(vars_to_check).copy()
        for var in vars_to_check:
            mask = ((param_df.loc[:, "rhs"] == var) &
                    (param_df.loc[:, "lhs"] == var) &
                    (param_df.loc[:, "rel"] == "=~") &
                    ~pars_to_exclude &
                    (param_df.loc[:, "fixedval"] == True)
                    )
            if np.any(mask):
                vars_to_add.remove(var)
        return vars_to_add

    def add_latent_dummies(self):
        param_df = self.param_df
        vars_to_check = self.var_names["lvo"]
        vars_to_add = self.check_latent_dummies(vars_to_check)
        if len(vars_to_add) > 0:
            list_of_param_dicts = param_df.to_dict(orient="records")
            for var in vars_to_add:
                row = {"lhs": var, "rel": "=~", "rhs": var, "fixed": True, "fixedval": 1, "dummy": True}
                list_of_param_dicts.append(row)
            self.param_df = pd.DataFrame(list_of_param_dicts)

    def fix_first(self):
        """
        Fixes the first loading of each latent variable to 1.0.

        Returns
        -------
        None
        """
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
