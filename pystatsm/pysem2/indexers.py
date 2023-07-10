import numpy as np
import scipy as sp

from ..utilities.func_utils import triangular_number
from ..utilities.indexing_utils import (nonzero, unique,
                                                        tril_indices)
from ..utilities.linalg_operations import (_vec, _vech)


class FlattenedArray:
    def __init__(self, array, symmetric=False, block_order=0, offset=0,
                 offset_nonzero=0, offset_unique=0):
        shape = array.shape
        if symmetric:
            v, _size = _vech(array), triangular_number(shape[-1])
        else:
            v, _size = _vec(array), np.product(shape[-2:])
        self._shape = shape
        self._size = _size
        self._symmetric = symmetric
        self._offset = offset
        self._start = 0
        self._stop = self._size
        (self._unique_values, self._unique_locs,
         self._first_locs) = self.unique_nonzero(v)
        self._flat_indices = nonzero(v).reshape(-1, order='F')
        self._row_indices, self._col_indices = nonzero(array)
        self._rc_indices = self._row_indices, self._col_indices
        self._n_nnz = len(self._flat_indices)
        self._offset_nonzero = offset_nonzero
        self._start_nnz = 0
        self._stop_nnz = self._n_nnz
        self._n_unz = len(self._unique_values)
        self._offset_unique = offset_unique
        self._start_unq = 0
        self._stop_unq = self._n_unz
        self._block_order = block_order
        self._block_indices = np.repeat(block_order, self._n_nnz)

    @staticmethod
    def unique_nonzero(v):
        u_vals, u_locs, f_locs = unique(v)
        mask = u_vals != 0
        zloc = np.where(u_vals == 0)[0]
        u_vals = u_vals[mask]
        if len(zloc) > 0:
            zmask = u_locs != zloc
        else:
            zmask = np.ones(len(u_locs), dtype=bool)
        u_locs = u_locs[zmask]
        if len(zloc) > 0:
            u_locs[u_locs > zloc] = u_locs[u_locs > zloc] - 1
        f_locs = f_locs[mask]
        return u_vals, u_locs, f_locs

    def update_offsets(self, **kwargs):
        for key, val in kwargs.items():
            if key in ["offset_nonzero", "offset_unique", "offset"]:
                setattr(self, key, val)

    def add_offsets(self, flat_index_obj):
        for name in ["_nnz", "_unq", ""]:
            val = getattr(flat_index_obj, "stop" + name)
            setattr(self, "_offset" + name, val)

    @property
    def stop(self):
        return self._stop + self._offset

    @property
    def start(self):
        return self._start + self._offset

    @property
    def stop_unq(self):
        return self._stop_unq + self._offset_unique

    @property
    def start_unq(self):
        return self._start_unq + self._offset_unique

    @property
    def stop_nnz(self):
        return self._stop_nnz + self._offset_nonzero

    @property
    def start_nnz(self):
        return self._start_nnz + self._offset_nonzero

    @property
    def flat_indices(self):
        return self._flat_indices + self._offset

    @property
    def first_locs(self):
        return self._first_locs + self._offset

    @property
    def unique_locs(self):
        return self._unique_locs + self._offset_unique

    def __str__(self):
        return (f"{self.start}:{self.stop}"
                f"{self.start_nnz}:{self.stop_nnz}"
                f"{self.start_unq}:{self.stop_unq}")


class BlockFlattenedArrays:
    def __init__(self, flat_arrays, shared_values=None):
        self._n_objs = len(flat_arrays)
        if shared_values is None:
            shared_values = np.arange(self._n_objs)
        self.flat_arrays = flat_arrays
        for i in range(1, self._n_objs):
            self.flat_arrays[i].add_offsets(self.flat_arrays[i - 1])
            self.flat_arrays[i]._block_indices = i + \
                                                 self.flat_arrays[i]._block_indices
        self._tril_inds = tril_indices(self.n_nonzero)

    @property
    def unique_locs(self):
        return np.concatenate([obj.unique_locs for obj in self.flat_arrays])

    @property
    def first_locs(self):
        return np.concatenate([obj.first_locs for obj in self.flat_arrays])

    @property
    def flat_indices(self):
        return np.concatenate([obj.flat_indices for obj in self.flat_arrays])

    @property
    def block_indices(self):
        return np.concatenate([obj._block_indices for obj in self.flat_arrays])

    @property
    def n_nonzero(self):
        return sum([obj._n_nnz for obj in self.flat_arrays])

    @property
    def col_indices(self):
        return np.concatenate([obj._col_indices for obj in self.flat_arrays])

    @property
    def row_indices(self):
        return np.concatenate([obj._row_indices for obj in self.flat_arrays])

    @property
    def slices(self):
        return [slice(obj.start, obj.stop) for obj in self.flat_arrays]

    @property
    def slices_nonzero(self):
        return [slice(obj.start_nnz, obj.stop_nnz) for obj in self.flat_arrays]

    @property
    def slices_unique(self):
        return [slice(obj.start_unq, obj.stop_unq) for obj in self.flat_arrays]

    @property
    def shapes(self):
        return [obj._shape for obj in self.flat_arrays]

    @property
    def is_symmetric(self):
        return [obj._symmetric for obj in self.flat_arrays]

    @property
    def unique_indices(self):
        return unique(self.unique_locs)[2]

    def create_derivative_arrays(self, nnz_cross_derivs=None):
        if nnz_cross_derivs is None:
            nnz_cross_derivs = list(zip(*tril_indices(self._n_objs)))

        deriv_shape = tuple(np.max(np.array(self.shapes), axis=0))
        dA = np.zeros((self.n_nonzero,) + deriv_shape)
        r, c = self.row_indices, self.col_indices
        block_indices = self.block_indices
        block_sizes = np.zeros((self.n_nonzero, 2), dtype=int)
        shapes = self.shapes
        is_symmetric = self.is_symmetric
        for i in range(self.n_nonzero):
            dA[i, r[i], c[i]] = 1.0
            kind = block_indices[i]
            block_sizes[i] = shapes[kind]
            if is_symmetric[kind]:
                dA[i, c[i], r[i]] = 1.0

        block_i = block_indices[self._tril_inds[0]]
        block_j = block_indices[self._tril_inds[1]]

        block_pairs = np.vstack([block_i, block_j]).T
        self.nf2 = triangular_number(self.n_nonzero)
        block_pair_types = np.zeros(self.nf2, dtype=int)

        for ii, (i, j) in enumerate(nnz_cross_derivs, 1):
            mask_i, mask_j = (block_i == i), (block_j == j)
            block_pair_types[mask_i & mask_j] = ii

        self.dA = dA
        self.block_sizes = block_sizes
        self.block_types_paired = block_pairs
        self.block_pair_types = block_pair_types
        self.colex_descending_inds = np.vstack(self._tril_inds).T


def equality_constraint_mat(unique_locs):
    n = unique_locs.max() + 1
    m = len(unique_locs)
    row = np.arange(m)
    col = unique_locs
    data = np.ones(m)
    arr = sp.sparse.csc_matrix((data, (row, col)), shape=(m, n))
    return arr
