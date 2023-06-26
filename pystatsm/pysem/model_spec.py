import numpy as np
import scipy as sp
import pandas as pd
from ..utilities.linalg_operations import _invec, _invech
from .param_table import BaseModel
from .model_data import ModelData


def _sparse_post_mult(A, S):
    prod = S.T.dot(A.T)
    prod = prod.T
    return prod


pd.set_option("mode.chained_assignment", None)


def equality_constraint_mat(unique_locs):
    n = unique_locs.max()+1
    m = len(unique_locs)
    row = np.arange(m)
    col = unique_locs
    data = np.ones(m)
    arr = sp.sparse.csc_matrix((data, (row, col)), shape=(m, n))
    return arr


class ModelSpecification(BaseModel):

    def __init__(self, formula, data, group_col, shared):
        model_data = ModelData.from_dataframe(data, group_col)
        super().__init__(formula, model_data, var_order=model_data.var_order,
                         n_groups=model_data.n_groups)
        self.construct_model_mats()
        self.reduce_parameters(shared)
        self.model_data = model_data
        self.p = len(self.var_names["obs"])
        self.q = len(self.var_names["lav"])
        self._check_complex = False

    def transform_free_to_theta(self, free):
        theta = free[self._first_locs]
        return theta

    def transform_free_to_group_free(self, free, i):
        group_free = free[self.free_to_group_free[i]]
        return group_free

    def transform_theta_to_free(self, theta):
        free = theta[self._unique_locs]
        return free

    def jac_group_free_to_free(self, arr_free, i, axes=(0,)):
        if 0 in axes:
            arr_free = self.dfree_dgroup[i].dot(arr_free)
        if 1 in axes:
            arr_free = _sparse_post_mult(arr_free, self.dfree_dgroup[i].T)
        return arr_free

    def jac_free_to_theta(self, arr_free, axes=(0,)):
        if 0 in axes:
            arr_free = self.dfree_dtheta.dot(arr_free)
        if 1 in axes:
            arr_free = _sparse_post_mult(arr_free, self.dfree_dtheta.T)
        return arr_free

    def group_free_to_par(self, free, i):
        par = self.p_templates[i].copy()
        if self._check_complex:
            if np.iscomplexobj(free):
                par = par.astype(complex)
        par[self.indexers[i].flat_indices] = free
        return par

    def par_to_model_mats(self, par, i):
        slices = self.indexers[i].slices
        shapes = self.indexers[i].shapes
        L = _invec(par[slices[0]], *shapes[0])
        B = _invec(par[slices[1]], *shapes[1])
        F = _invech(par[slices[2]])
        P = _invech(par[slices[3]])
        a = _invec(par[slices[4]], *shapes[4])
        b = _invec(par[slices[5]], *shapes[5])
        return L, B, F, P, a, b

    def group_free_to_model_mats(self, free, i):
        par = self.group_free_to_par(free, i)
        mats = self.par_to_model_mats(par, i)
        return mats