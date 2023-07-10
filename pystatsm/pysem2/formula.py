import re

import numpy as np
import pandas as pd


class FormulaParser:
    """
    A class to parse a formula and extract variables and parameters.

    Parameters
    ----------
    formulas : str
        A string containing the formulas.

    Attributes
    ----------
    param_df : pandas.DataFrame
        A DataFrame containing the parameters.
    var_names : dict
        A dictionary containing the names of the variables
    from the formulas.
    """

    def __init__(self, formulas):
        self._parse_formula(formulas)
        self.var_names = self.classify_variables(self.param_df)

    def _parse_formula(self, formulas):
        """
        Parses a formula and extracts the parameters.
        Parameters
        ----------
        formulas : str

        Returns
        -------
        None
        """
        formulas = re.sub(r'\s*#.*', '', formulas)
        self._param_list = []
        equations = formulas.strip().split('\n')
        for equation in equations:
            if equation.strip():
                self.unpack_equation(equation)
        self.param_df = pd.DataFrame(self._param_list)

    @staticmethod
    def _get_var_pair(left_side, right_side, rel):
        """
        Gets a variable pair from a left-hand side and a right-hand side.
        Parameters
        ----------
        left_side : str
        right_side : str
        rel : str

        Returns
        -------
        None
        """
        comps = right_side.strip().split('*')
        left_side = left_side.strip()
        if len(comps) > 1:
            mod, name = comps[0].strip(), comps[1].strip()
            try:
                fixedval, fixed, label = float(mod), True, None
            except ValueError:
                fixedval, fixed, label = np.inf, False, mod
        else:
            mod, name = None, comps[0].strip()
            fixedval, fixed, label = np.inf, False, None
        row = {"lhs": left_side, "rel": rel, "rhs": name, "mod": mod, "label": label,
               "fixedval": fixedval, "fixed": fixed, "dummy": False}

        return row

    def unpack_equation(self, equation):
        """
        Unpacks an equation into a list of parameters.

        Parameters
        ----------
        equation : str
            A string containing the equation.

        Returns
        -------
        None
        """
        if "=~" in equation:
            rel = "=~"
        elif "~~" in equation:
            rel = "~~"
        else:
            rel = "~"
        lhss, rhss = equation.split(rel)
        for left_side in lhss.split('+'):
            for right_side in rhss.split('+'):
                self._param_list.append(self._get_var_pair(left_side, right_side, rel))

    @staticmethod
    def classify_variables(param_df):
        """
        Classifies the variables in the parameter DataFrame.

        Parameters
        ----------
        param_df : pandas.DataFrame
            A DataFrame containing the parameters.

        Returns
        -------
        names : dict
            A dictionary containing the names of the variables.
        """
        one_set = {"1"}
        measurement_mask = param_df["rel"] == "=~"
        regressions_mask = (param_df["rel"] == "~") & (param_df["rhs"] != "1")
        all_var_names = set(param_df[["lhs", "rhs"]].values.flatten()) - one_set
        nob_var_names = set(param_df.loc[measurement_mask, "lhs"])
        obs_var_names = all_var_names - nob_var_names
        ind_var_names = set(param_df.loc[measurement_mask, "rhs"])
        end_var_names = set(param_df.loc[regressions_mask, "lhs"])
        exo_var_names = set(param_df.loc[regressions_mask, "rhs"]) - one_set  # exog variables
        reg_var_names = set.union(end_var_names, exo_var_names)
        lvo_var_names = reg_var_names - nob_var_names
        lav_var_names = lvo_var_names | nob_var_names
        lox_var_names = lav_var_names - (nob_var_names | end_var_names | ind_var_names)
        loy_var_names = lav_var_names - (nob_var_names | exo_var_names | ind_var_names)
        lvx_var_names = nob_var_names - (ind_var_names | end_var_names)  # true latent exog vars
        enx_var_names = end_var_names - exo_var_names  # endo but not exo var
        onx_var_names = obs_var_names - lox_var_names
        names = {"all": all_var_names, "nob": nob_var_names,
                 "obs": obs_var_names, "ind": ind_var_names,
                 "end": end_var_names, "exo": exo_var_names,
                 "reg": reg_var_names, "lvo": lvo_var_names,
                 "lav": lav_var_names, "lox": lox_var_names,
                 "loy": loy_var_names, "enx": enx_var_names,
                 "lvx": lvx_var_names, "onx": onx_var_names}
        return names

    @property
    def n_lav_vars(self):
        """The number of latent variables both dummy and true model latent variables"""
        return len(self.var_names["lav"])

    @property
    def n_obs_vars(self):
        """The number of observed variables"""
        return len(self.var_names["obs"])

    @property
    def mat_dims(self):
        """The dimensions of the matrices in the model"""
        n_obs_vars, n_lav_vars = self.n_obs_vars, self.n_lav_vars
        return {0: (n_obs_vars, n_lav_vars), 1: (n_lav_vars, n_lav_vars),
                2: (n_lav_vars, n_lav_vars), 3: (n_obs_vars, n_obs_vars),
                4: (1, n_obs_vars), 5: (1, n_lav_vars)}
