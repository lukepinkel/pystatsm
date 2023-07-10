
import re
import numpy as np
import pandas as pd


def _default_sort_key(item):
    match = re.match(r"([a-zA-Z]+)(\d+)", item)
    if match:
        alphabetic_part = match.group(1)
        numeric_part = int(match.group(2))
    else:
        alphabetic_part = item
        numeric_part = 0
    return (alphabetic_part, numeric_part)


class VariableOrder:

    @staticmethod
    def default_sort(subset, var_names):
        """
        Sorts a set of variable names into a standard order.

        Parameters
        ----------
        subset : set
            set of variable names to sort
        var_names : dict
            dictionary of variable names

        Returns
        -------
        g : list
            sorted list of variable names
        """
        g = []
        g.extend(sorted(subset & var_names["obs"] & var_names["ind"], key=_default_sort_key))
        g.extend(sorted(subset & var_names["nob"] & var_names["ind"], key=_default_sort_key))
        g.extend(sorted(subset & var_names["obs"] & var_names["end"], key=_default_sort_key))
        g.extend(sorted(subset & var_names["nob"] - (var_names["nob"] & var_names["ind"]), key=_default_sort_key))
        u = set(g)
        g.extend(sorted(subset - u, key=_default_sort_key))
        return g

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

    def sort_table(self):
        param_df = self.get_table()
        obs_order, lav_order = self.get_obs_order(), self.get_lav_order()
        self.set_table(self._sort_table(param_df, obs_order, lav_order))

    @staticmethod
    def _sort_table(param_df, obs_order, lav_order):
        if "mat" not in param_df.columns:
            ValueError("Parameters must be assigned to matrices first.")
        mats = np.unique(param_df["mat"])
        mat_dict = {}
        mat_rc = {0: (obs_order, lav_order), 1: (lav_order, lav_order),
                  2: (lav_order, lav_order), 3: (obs_order, obs_order),
                  4: ({"0": 0, 0: 0}, obs_order), 5: ({"0": 0, 0: 0}, lav_order)}
        is_symmetric = {0: False, 1: False, 2: True, 3: True, 4: False, 5: False}
        is_vector = {0: False, 1: False, 2: False, 3: False, 4: True, 5: True}
        for i in sorted(mats):
            mat = param_df.loc[param_df["mat"] == i]
            rmap, cmap = mat_rc[i]
            mat["r"], mat["c"] = mat["r"].map(rmap), mat["c"].map(cmap)
            mat = VariableOrder.sort_flat_representation(mat, is_symmetric[i], is_vector[i])
            mat_dict[i] = mat
        param_df = pd.concat([mat_dict[i] for i in sorted(mats)], axis=0)
        return param_df

    def reorder_table(self, obs_order=None, lav_order=None):
        self.set_obs_order(obs_order)
        self.set_lav_order(lav_order)
        self.assign_matrices()
        self.sort_table()
        self.index_params()

    @staticmethod
    def _ord_dict(iterable):
        return dict(zip(iterable, np.arange(len(iterable))))

    def set_obs_order(self, obs_order):
        if obs_order is None:
            obs_order = sorted(self.var_names["obs"], key=_default_sort_key)
        if type(obs_order) is not dict:
            obs_order = self._ord_dict(obs_order)
        self.obs_order = obs_order

    def set_lav_order(self, lav_order):
        if lav_order is None:
            lav_order = self.default_sort(self.var_names["lav"], self.var_names)
        if type(lav_order) is not dict:
            lav_order = self._ord_dict(lav_order)
        self.lav_order = lav_order

    def get_lav_order(self):
        return self.lav_order

    def get_obs_order(self):
        return self.obs_order

    @property
    def sorted_lav_names(self):
        lv_order = self.get_lav_order()
        lv_names = sorted(lv_order.keys(), key=lambda x: lv_order[x])
        return lv_names

    @property
    def sorted_obs_names(self):
        ov_order = self.obs_order
        ov_names = sorted(ov_order.keys(), key=lambda x: ov_order[x])
        return ov_names

    @property
    def mat_labels(self):
        ov_names, lv_names = self.sorted_obs_names, self.sorted_lav_names
        mat_rows = {0: ov_names, 1: lv_names, 2: lv_names, 3: ov_names, 4: ["0"], 5: ["0"]}
        mat_cols = {0: lv_names, 1: lv_names, 2: lv_names, 3: ov_names, 4: ov_names, 5: lv_names}
        return mat_rows, mat_cols
