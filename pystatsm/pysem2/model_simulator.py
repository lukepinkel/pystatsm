
import numpy as np
import pandas as pd
import scipy as sp
from ..utilities.random import r_lkj, exact_rmvnorm
from .model_specification import ModelSpecification
from .formula import FormulaParser

rng=np.random.default_rng()

class SimulatedSEM:
    def __init__(self, formula, n_groups, n_obs_min=1000, n_obs_max=5000, rng=None):
        self.formula = formula
        self.n_groups = n_groups
        self.n_obs_min = n_obs_min
        self.n_obs_max = n_obs_max
        self.rng = np.random.default_rng() if rng is None else rng

    def simulate_group_sizes(self):
        return self.rng.integers(low=self.n_obs_min, high=self.n_obs_max, size=self.n_groups)

    def generate_data(self, n_obs):
        formula_parser = FormulaParser(self.formula)
        exog_var_list = list(formula_parser.var_names["lox"])
        nonx_var_list = sorted(formula_parser.var_names["obs"] - set(exog_var_list))
        obs_var_list = exog_var_list + nonx_var_list
        if len(exog_var_list)>0:
            exog_cov = r_lkj(1, 1, len(exog_var_list)).squeeze()
            nonx_cov = np.eye(len(nonx_var_list))
            obs_cov = sp.linalg.block_diag(exog_cov, nonx_cov)
        else:
            obs_cov = np.eye(len(nonx_var_list))
        means = np.linspace(-3, 3, self.n_groups)
        dfs = []
        for i in range(self.n_groups):
            arr = exact_rmvnorm(obs_cov, n_obs[i], np.repeat(means[i], len(obs_var_list)))
            arr = pd.DataFrame(arr, columns=obs_var_list)
            arr["group"] = i
            dfs.append(arr)
        df = pd.concat(dfs, axis=0, ignore_index=True)
        return df

    def set_parameter(self, mat, lhs_eq_rhs, low=0.5, high=1.0):
        mask = (self.param_df["mat"] == mat) & (~self.param_df["fixed"])
        if lhs_eq_rhs is not None:
            mask &= (self.param_df["lhs"] == self.param_df["rhs"])
        self.param_df.loc[mask, "start"] = self.rng.uniform(low, high, size=mask.sum())

    def set_reasonable_params(self):
        self.set_parameter(mat=0, lhs_eq_rhs=None, low=0.8, high=0.9)
        self.set_parameter(mat=1, lhs_eq_rhs=None, low=0.5, high=0.9)
        self.set_parameter(mat=2, lhs_eq_rhs=True, low=2.0, high=3.0)
        self.set_parameter(mat=3, lhs_eq_rhs=True,  low=0.5, high=0.9)
        self.model_spec.set_table(self.param_df)

    def simulate(self):
        n_obs = self.simulate_group_sizes()
        df = self.generate_data(n_obs)
        self.model_spec = ModelSpecification(self.formula, data=df, group_col="group", shared=[True, False, True, False, False, False])
        self.param_df = self.model_spec.get_table()
        self.set_reasonable_params()
        self.model_spec.make_parameter_templates()
        self.model_spec.reduce_parameters(self.model_spec.shared)
        L, B, F, P, a, b = {}, {}, {}, {}, {}, {}
        Sigma = {}
        mu = {}
        free = self.model_spec.free
        for i in range(self.model_spec.n_groups):
            group_free = self.model_spec.transform_free_to_group_free(free, i)
            par = self.model_spec.group_free_to_par(group_free, i)
            L[i], B[i], F[i], P[i], a[i], b[i] = self.model_spec.par_to_model_mats(par)
            B[i] = np.linalg.inv(np.eye(B[i].shape[0]) - B[i])
            L[i] = pd.DataFrame(L[i], index=self.model_spec.mat_labels[0][0], columns=self.model_spec.mat_labels[1][0])
            B[i] = pd.DataFrame(B[i], index=self.model_spec.mat_labels[0][1], columns=self.model_spec.mat_labels[1][1])
            F[i] = pd.DataFrame(F[i], index=self.model_spec.mat_labels[0][2], columns=self.model_spec.mat_labels[1][2])
            P[i] = pd.DataFrame(P[i], index=self.model_spec.mat_labels[0][3], columns=self.model_spec.mat_labels[1][3])
            a[i] = pd.DataFrame(a[i], index=self.model_spec.mat_labels[0][4], columns=self.model_spec.mat_labels[1][4])
            b[i] = pd.DataFrame(b[i], index=self.model_spec.mat_labels[0][5], columns=self.model_spec.mat_labels[1][5])
            Sigma[i] = L[i].dot(B[i]).dot(F[i].dot(B[i].T)).dot(L[i].T)+P[i]
            mu[i] = L[i].dot(B[i]).dot(b[i].T)+a[i].T
        dfs = []
        for i in range(self.n_groups):
            arr = rng.multivariate_normal(mean= mu[i].values.flatten(), cov=Sigma[i].values, size=n_obs[i])
            arr = pd.DataFrame(arr, columns=Sigma[i].columns)
            arr["group"] = i
            dfs.append(arr)
        df = pd.concat(dfs, axis=0, ignore_index=True)
        return df
