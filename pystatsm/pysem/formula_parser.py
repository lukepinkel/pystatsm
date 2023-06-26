#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 16:18:10 2023

@author: lukepinkel
"""


import re
import pandas as pd


class FormulaParser:
    """
    A class to parse a set of formulas and extract variables and parameters
    from the formulas.
    """

    def __init__(self, formulas):
        """
        Initialize the FormulaParser with a set of formulas.

        Parameters
        ----------
        formulas : str
            A string of formulas to be parsed, separated by newlines.
        """
        self._formulas = formulas
        self.parse_formula(formulas)
        self._param_df = pd.DataFrame(self._param_list)
        self._var_names = self.classify_variables(self._param_df)

    def parse_formula(self, formulas):
        """
        Parse a set of formulas (in a single string separated by newlines),
        extracting all variables and parameters.

        Parameters
        ----------
        formulas : str
            A string of formulas to be parsed, separated by newlines.

        Returns
        ----------
        parameters : list of dict
            A list of dictionaries, each representing a parameter extracted
            from the formulas.
        """
        formulas = re.sub(r'\s*#.*', '', formulas)
        self._param_list = []
        equations = formulas.strip().split('\n')
        for eq in equations:
            if eq.strip():
                self.unpack_equation(eq)

    @staticmethod
    def _get_var_pair(ls, rs, rel):
        """
        Extracts a variable pair and relationship from a formula.

        Parameters
        ----------
        ls : str
            The left-hand side of the equation.

        rs : str
            The right-hand side of the equation.

        rel : str
            The relationship operator in the equation
            (one of "=~", "~~", or "~").

        Returns
        ----------
        row : dict
            A dictionary representing the extracted variable pair
            and relationship.
        """
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
            label = None  # f"{ls}{rel}{name}"
            fixedval = None
        row = {"lhs": ls, "rel": rel, "rhs": name, "mod": mod,
               "label": label, "fixedval": fixedval,
               "fixed": fixed}
        return row

    def unpack_equation(self, eq):
        """
        Unpacks an equation into component variable pairs and relationships.

        Parameters
        ----------
        eq : str
            The equation to be unpacked.

        parameters : list of dict
            A list of dictionaries, each representing a parameter.

        Returns
        ----------
        parameters : list of dict
            The updated list of dictionaries representing parameters,
            after unpacking the equation.
        """
        if "=~" in eq:
            rel = "=~"
        elif "~~" in eq:
            rel = "~~"
        else:
            rel = "~"
        lhss, rhss = eq.split(rel)
        for ls in lhss.split('+'):
            for rs in rhss.split('+'):
                row = self._get_var_pair(ls, rs, rel)
                self._param_list.append(row)

    @staticmethod
    def classify_variables(param_df):
        """
        Classifies variables from the formulas into different categories
        based on their roles.

        The variables are classified into the following categories:
        - 'all': All variables in the model.
        - 'nob': Non-observed (latent) variables.
        - 'obs': Observed variables, i.e., variables not in 'nob'.
        - 'ind': Indicator variables observed or unobserved.
        - 'end': Endogenous variables, variables determined within the
                 system of the model.
        - 'exo': Exogenous variables, variables not determined within the
                 system of the model.
        - 'reg': Any variables involved in regression equations
                 (i.e., 'end' union 'exo').
        - 'lvo': Observed variables that are part of the structural model
                 (intersection of 'reg' and 'obs').
        - 'lav': All variables treated as part of the structural model
                 (union of 'lvo' and 'nob').
        - 'lox': Observed exogenous variables in the structural model
                 (difference of 'lav' and union of 'nob', 'end', 'ind').
        - 'loy': Observed endogenous variables in the structural model
                 (difference of 'lav' and union of 'nob', 'exo', 'ind').
        - 'lvx': True latent exogenous variables
                 (difference of 'nob' and union of 'ind', 'end').
        - 'enx': Endogenous variables not considered exogenous
                 (difference of 'end' and 'exo').

        These categories are not mutually exclusive and some variables
        may belong to multiple categories.

        Parameters
        ----------
        param_df : pandas.DataFrame
            DataFrame where each row represents a parameter in the formula.

        Returns
        ----------
        names : dict
            A dictionary where keys are categories and values are sets of
            variable names in each category.
        """
        one_set = set(["1"])
        measurement_mask = param_df["rel"] == "=~"
        regressions_mask = ((param_df["rel"] == "~")
                            & (param_df["rhs"] != "1"))
        all_var_names = set(
            param_df[["lhs", "rhs"]].values.flatten()) - one_set
        # unobserved variables i.e. lv
        nob_var_names = set(param_df.loc[measurement_mask, "lhs"])
        # observed variables i.e. ov  - rows of Lambda
        obs_var_names = all_var_names - nob_var_names
        # indicator variables observed or unobserved v_inds
        ind_var_names = set(param_df.loc[measurement_mask, "rhs"])
        # endog variables
        end_var_names = set(param_df.loc[regressions_mask, "lhs"])
        exo_var_names = set(
            param_df.loc[regressions_mask, "rhs"]) - one_set  # exog variables
        # any variables involved in regression
        reg_var_names = set.union(end_var_names, exo_var_names)
        # regression variables that are observed i.e. observed vars to add
        # to structural model - the rows and cols of Lambda with identity
        lvo_var_names = reg_var_names - nob_var_names
        # all effectively latent variables i.e. treated as part of structural
        # model - cols of lambda and row/col of beta and phi
        lav_var_names = lvo_var_names | nob_var_names
        # observed exog variables in structural part of model i.e.latent
        # observed - fix Phi and beta at sample stats
        lox_var_names = lav_var_names - \
            (nob_var_names | end_var_names | ind_var_names)
        # observed endog variables in structural part of model i.e.
        # latent observed
        loy_var_names = lav_var_names - \
            (nob_var_names | exo_var_names | ind_var_names)
        lvx_var_names = nob_var_names - \
            (ind_var_names | end_var_names)  # true latent exog vars
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
