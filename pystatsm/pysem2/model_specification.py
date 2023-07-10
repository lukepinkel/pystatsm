
import numpy as np

from ..utilities.func_utils import triangular_number
from ..utilities.linalg_operations import _invec, _invech
from .model_base import ModelBase
from .model_data import ModelData

def _sparse_post_mult(A, S):
    prod = S.T.dot(A.T)
    prod = prod.T
    return prod


class ModelSpecification(ModelBase):
    def __init__(self, formula, data=None, group_col=None, shared=None, model_data=None, **kwargs):
        if model_data is None:
            self.model_data = ModelData.from_dataframe(data, group_col)
        else:
            self.model_data = model_data
        self.n_groups = self.model_data.n_groups
        super().__init__(formula, **kwargs)
        self.duplicate_parameter_table(self.n_groups)
        self.update_sample_stats(sample_stats=self.model_data)
        self.make_indexer()
        self.make_parameter_templates()
        self.reduce_parameters(shared)
        self._check_complex = False
        self.get_constants()

    def get_constants(self):
        self.model_data.subset_and_order(self.obs_order)
        self.p, self.q = len(self.var_names["obs"]), len(self.var_names["lav"])
        self.bounds = [tuple(x) for x in self.free_df[["lb", "ub"]].values[self._first_locs]]
        self.p, self.q = len(self.var_names["obs"]), len(self.var_names["lav"])
        self.p2, self.q2 = triangular_number(self.p), triangular_number(self.q)
        self.nf, self.nf2 = len(self.indexer.flat_indices), triangular_number(len(self.indexer.flat_indices))
        self.nt = len(self.indexer.first_locs)
        self.n_par = len(self.parameter_templates[0])
        self.ll_const = 1.8378770664093453 * self.p
        self.gsizes = np.array(list(self.model_data.n_obs.values()))
        self.gweights = self.gsizes / np.sum(self.gsizes)
        self.n = np.sum(self.gsizes)
        self.make_derivative_matrices()
        self.theta = self.transform_free_to_theta(self.free)

    def make_derivative_matrices(self):
        self.indexer.create_derivative_arrays(
            [(0, 0), (1, 0), (2, 0), (1, 1), (2, 1), (5, 0), (5, 1)])
        self.dA = self.indexer.dA
        self.n_group_theta = len(self._first_locs)
        self.dSm = np.zeros((self.n_groups, self.p2 + self.p, self.nf))
        self.d2Sm = np.zeros((self.n_groups, self.p2 + self.p, self.nf, self.nf))
        self.m_size = self.indexer.block_sizes
        self.m_kind = self.indexer.block_indices
        self.d2_kind = self.indexer.block_pair_types
        self.d2_inds = self.indexer.colex_descending_inds
        s, r = np.triu_indices(self.p, k=0)
        self._vech_inds = r + s * self.p
        self.unique_locs = self.indexer.unique_locs
        self.free_names = self.free_df["label"]
        self.theta_names = self.free_df.iloc[self._first_locs]["label"]
        self._grad_kws = dict(dA=self.dA, m_size=self.m_size, m_type=self.m_kind,
                              n=self.nf)
        self._hess_kws = dict(dA=self.dA, m_size=self.m_size, d2_inds=self.d2_inds,
                              first_deriv_type=self.m_kind, second_deriv_type=self.d2_kind,
                              n=self.nf)
        self._grad_kws1 = dict(dL=np.zeros((self.p, self.q)),
                               dB=np.zeros((self.q, self.q)),
                               dF=np.zeros((self.q, self.q)),
                               dP=np.zeros((self.p, self.p)),
                               da=np.zeros((self.p)),
                               db=np.zeros((self.q)),
                               r=self.indexer.row_indices,
                               c=self.indexer.col_indices,
                               m_type=self.m_kind,
                               n=self.nf)

    def transform_free_to_theta(self, free):
        theta = free[self._first_locs]
        return theta

    def transform_free_to_group_free(self, free, i):
        group_free = free[self.free_to_group_free[i]]
        return group_free

    def transform_group_free_to_par(self, group_free, i):
        par = self.parameter_templates[i].copy()
        if self._check_complex:
            if np.iscomplexobj(group_free):
                par = par.astype(complex)
        par[self.indexers[i].flat_indices] = group_free
        return par

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
        par = self.parameter_templates[i].copy()
        if self._check_complex:
            if np.iscomplexobj(free):
                par = par.astype(complex)
        par[self.indexer.flat_indices] = free
        return par

    def par_to_model_mats(self, par):
        slices = self.indexer.slices
        shapes = self.indexer.shapes
        L = _invec(par[slices[0]], *shapes[0])
        B = _invec(par[slices[1]], *shapes[1])
        F = _invech(par[slices[2]])
        P = _invech(par[slices[3]])
        a = _invec(par[slices[4]], *shapes[4])
        b = _invec(par[slices[5]], *shapes[5])
        return L, B, F, P, a, b

    def group_free_to_model_mats(self, free, i):
        par = self.group_free_to_par(free, i)
        mats = self.par_to_model_mats(par)
        return mats
