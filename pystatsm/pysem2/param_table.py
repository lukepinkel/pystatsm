import numpy as np
import pandas as pd

from .formula import FormulaParser
from .model_builder import ModelBuilder
from .model_matrices import ModelMatrixMapper
from .variable_ordering import VariableOrder

pd.set_option("mode.chained_assignment", None)


class ParameterTable(FormulaParser, ModelBuilder, ModelMatrixMapper, VariableOrder):
    """
    Class that creates a parameter table from a formula.
    """
    matrix_names = ["L", "B", "F", "P", "a", "b"]
    matrix_order = {"L": 0, "B": 1, "F": 2, "P": 3, "a": 4, "b": 5}
    is_symmetric = {0: False, 1: False, 2: True, 3: True, 4: False, 5: False}
    is_vector = {0: False, 1: False, 2: False, 3: False, 4: True, 5: True}

    def __init__(self, formula, **kwargs):
        """

        Parameters
        ----------
        formula
        kwargs
        """
        super().__init__(formula)
        self.set_lav_order(kwargs.get("lav_order", None))  # from VariableOrder
        self.set_obs_order(kwargs.get("obs_order", None))  # from VariableOrder
        if kwargs.get("add_default_params", True):
            self.add_default_params(**kwargs)  # from ModelBuilder
        if kwargs.get("assign_matrices", True):
            self.assign_matrices()  # from ModelMatrixMapper
        if kwargs.get("sort_table", True):
            self.sort_table()  # from VariableOrder
        if kwargs.get("index_parameters", True):
            self.index_params()  # from this class
        if kwargs.get("add_bounds", True):
            self.add_bounds()  # from this class

    def set_table(self, param_df):
        self.param_df = param_df

    def get_table(self):
        return self.param_df.copy()

    def get_free_table(self):
        return self.param_df.loc[self.param_df["free"] != 0].copy().reset_index(drop=True)

    @staticmethod
    def _add_bounds(param_dfs):
        ix = (param_dfs["lhs"] == param_dfs["rhs"]) & (
                param_dfs["rel"] == "~~")
        param_dfs["lb"] = None
        param_dfs.loc[ix, "lb"] = 0
        param_dfs.loc[~ix, "lb"] = None
        param_dfs["ub"] = None
        return param_dfs

    def add_bounds(self):
        """
        Adds lower and upper bounds to the parameter table.

        Parameters
        ----------
        param_df : pandas.DataFrame
            parameter table

        Returns
        -------
        None
        """
        self.set_table(self._add_bounds(self.get_table()))

    @staticmethod
    def _index_params(param_df):
        """
        This function manages parameter indexing in a lavaan-style formula's dataframe. 
        It tracks the fixed or free status of parameters and those that are set to be equal.
        Each parameter that is free and not tied to others gets a unique index, while 
        parameters that are tied together share the same index. For example
                | fixed | label |
                |-------|-------|
                | False | NaN   |
                | False | 'A'   |
                | False | 'A'   |
                | True  | NaN   |
                | False | NaN   |
                | False | 'B'   |
                | False | 'B'   |
            
        becomes
        
                | fixed | label | free | ind |
                |-------|-------|------|-----|
                | False | NaN   | 1    | 0   |
                | False | 'A'   | 2    | 1   |
                | False | 'A'   | 2    | 1   |
                | True  | NaN   | 0    | -   |
                | False | NaN   | 3    | 2   |
                | False | 'B'   | 4    | 3   |
                | False | 'B'   | 4    | 3   |
        """
        # Set initial all parameters to be 'free' (indicated by 0)
        param_df["free"] = 0

        # Identify free parameters (i.e., parameters not set to be fixed in the model)
        free_parameter_mask = ~param_df["fixed"]

        # Identify parameters that are tied to others (i.e., parameters with a label)
        labeled_parameter_mask = ~param_df["label"].isnull()

        # Prepare a dictionary to store parameter labels and their corresponding indices
        parameter_links = {}

        # Get unique labels for parameters (ignoring nulls)
        unique_labels = np.unique(param_df.loc[labeled_parameter_mask, "label"])

        # For each unique label, find the smallest index of a parameter with that label.
        # These parameters are considered 'representatives' of their group
        for label in unique_labels:
            parameters_with_same_label_mask = param_df["label"] == label
            same_label_indices = param_df.loc[parameters_with_same_label_mask, "label"].index.values
            representative_index = np.min(same_label_indices)

            parameter_links[label] = representative_index, same_label_indices
            labeled_parameter_mask[representative_index] = False  # Unset the label for the representative parameter

        # Find parameters that are free and not tied to others
        independent_parameter_mask = free_parameter_mask & ~labeled_parameter_mask

        # Assign unique indices to independent parameters
        param_df.loc[independent_parameter_mask, "free"] = np.arange(1, 1 + len(param_df[independent_parameter_mask]))

        # Assign the same index to parameters tied together (based on the index of their group representative)
        for representative_index, same_label_indices in parameter_links.values():
            param_df.loc[same_label_indices, "free"] = param_df.loc[representative_index, "free"]

        # Prepare a 'ind' column
        param_df["ind"] = 0

        # Assign a unique, zero-based index to parameters that are considered 'free' after the above process
        free_after_process_mask = param_df["free"] != 0
        param_df.loc[free_after_process_mask, "ind"] = np.arange(np.sum(free_after_process_mask))

        return param_df

    def index_params(self):
        self.set_table(self._index_params(self.get_table()))
