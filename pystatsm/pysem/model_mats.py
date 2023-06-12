#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 01:22:49 2023

@author: lukepinkel
"""


import numpy as np
from ..utilities.indexing_utils import nonzero, unique, tril_indices
from ..utilities.linalg_operations import _vec, _vech, _invech, _invec
from ..utilities.func_utils import triangular_number


class FlattenedIndicatorIndices(object):
    """
    This class represents an array in a flattened form with multiple offset-based indices for efficient storage 
    and indexing of the array's unique non-zero elements and all non-zero elements.

    Parameters
    ----------
    array : array-like
        Input array that needs to be flattened and indexed.
    symmetric : bool, optional
        If True, array is considered symmetric. Default is False.
    block_order : int, optional
        Order of the array block. Default is 0.
    offset : int, optional
        Offset for indexing in the complete flattened array. Default is 0.
    offset_nonzero : int, optional
        Offset for indexing non-zero elements in the flattened array. Default is 0.
    offset_unique : int, optional
        Offset for indexing unique non-zero elements in the flattened array. Default is 0.
    
    Attributes
    ----------
    _shape : tuple
        Shape of the original array.
    _size : int
        Total number of elements in the original array.
    _symmetric : bool
        True if the original array is symmetric, False otherwise.
    _offset : int
        Offset for indexing in the complete flattened array.
    _unique_values : numpy.ndarray
        Unique non-zero values in the flattened array.
    _unique_locs : numpy.ndarray
        Locations of unique non-zero values in the flattened array.
    _first_locs : numpy.ndarray
        First occurrence of each unique non-zero value in the flattened array.
    _flat_indices : numpy.ndarray
        Indices of non-zero values in the flattened array.
    _row_indices : numpy.ndarray
        Row indices of non-zero elements in the original array.
    _col_indices : numpy.ndarray
        Column indices of non-zero elements in the original array.
    _rc_indices : tuple of numpy.ndarray
        Tuple containing row and column indices of non-zero elements in the original array.
    _n_nnz : int
        Number of non-zero elements in the original array.
    _n_unz : int
        Number of unique non-zero elements in the original array.
    _block_order : int
        Order of the array block.
    _block_indices : numpy.ndarray
        Indices of the array block.

    """
    def __init__(self, array, symmetric=False, block_order=0, offset=0 ,
                 offset_nonzero=0, offset_unique=0):
        shape = array.shape
        if symmetric:
            v = _vech(array)
            _size = triangular_number(shape[-1])
        else:
            v = _vec(array)
            _size = np.product(shape[-2:])
        self._shape = shape
        self._size = _size
        self._symmetric = symmetric
        self._offset = offset
        self._start = 0
        self._stop = self._size
        self._unique_values, self._unique_locs, self._first_locs = self.unique_nonzero(v)
        self._flat_indices = nonzero(v).reshape(-1, order='F')
        self._row_indices, self._col_indices = nonzero(array)
        self._rc_indices = self._row_indices, self._col_indices
        self._n_nnz = len(self._flat_indices)
        
        self._offset_nonzero = offset_nonzero
        self._start_nonzero = 0
        self._stop_nonzero = self._n_nnz
        
        self._n_unz = len(self._unique_values)
        self._offset_unique = offset_unique
        self._start_unique = 0
        self._stop_unique = self._n_unz
        
        self._block_order = block_order
        self._block_indices = np.repeat(block_order, self._n_nnz)
        
    @staticmethod
    def unique_nonzero(v):
        """
        Returns the unique non-zero elements of the input array along with their locations and first occurrences.
    
        Parameters
        ----------
        v : array-like
            Input array.
    
        Returns
        -------
        u_vals : numpy.ndarray
            Unique non-zero values in the input array.
        u_locs : numpy.ndarray
            Locations of unique non-zero values in the input array.
        f_locs : numpy.ndarray
            First occurrence of each unique non-zero value in the input array.
        """
        u_vals, u_locs, f_locs = unique(v)                  # get unique values, their locations and first occurrences
        mask = u_vals!=0                                    # create a boolean mask to get non-zero unique values
        zloc = np.where(u_vals==0)[0]                       # find the location of zero in unique values
        u_vals = u_vals[mask]                               # apply the mask to get non-zero unique values
        if len(zloc)>0:  
            zmask = u_locs!=zloc                            # create a mask where locations of unique values are not equal to location of zero
        else:
            zmask = np.ones(len(u_locs), dtype=bool)        # if there is no zero in unique values, all elements are valid
        u_locs = u_locs[zmask]                              # apply the mask to get locations of non-zero unique values
        if len(zloc)>0:
            u_locs[u_locs>zloc] =  u_locs[u_locs>zloc] - 1      # decrease indices that are greater than the index of zero
        f_locs = f_locs[mask]                               # apply the mask to get first occurrences of non-zero unique values
        return u_vals, u_locs, f_locs
    
    def update_offsets(self, **kwargs):
        """
        Updates the offset attributes of the object.
    
        Parameters
        ----------
        kwargs : dict
            Keyword arguments containing offset attributes to be updated.
        """
        for key, val in kwargs.items():
            if key in ["offset_nonzero", "offset_unique", "offset"]:
                setattr(self, key, val)
        
    def add_offsets(self, flat_index_obj):
        """
        Adds the offsets of another `FlattenedIndicatorIndices` object to the current object.
    
        Parameters
        ----------
        flat_index_obj : FlattenedIndicatorIndices object
            Another `FlattenedIndicatorIndices` object whose offsets are to be added.
        """
        for name in ["_nonzero", "_unique", ""]:
            val = getattr(flat_index_obj, "stop"+name)
            setattr(self, "_offset"+name, val)
      
    def to_vec(self):
        """
        Returns the vectorized form of the unique non-zero elements in the flattened array.
    
        Returns
        -------
        v : numpy.ndarray
            Vector containing the unique non-zero elements in the flattened array.
        """
        v_u=self._unique_values[self._unique_locs]
        v = np.zeros(self._size)
        v[self._flat_indices] = v_u
        return v
    
    def to_array(self):
        """
        Returns the original array form of the unique non-zero elements in the flattened array.
    
        Returns
        -------
        arr : numpy.ndarray
            Array containing the unique non-zero elements in the flattened array.
        """
        v = self.to_vec()
        if self._symmetric:
            arr = _invech(v)
        else:
            arr = _invec(v, *self._shape)
        return arr
        
    def set_equal(self, lhs, rhs):
        """
        Replaces all occurrences of a value in the unique non-zero elements with another value.
    
        Parameters
        ----------
        lhs : float
            The value to be replaced.
        rhs : float
            The value to replace with.
        """
        v_u=self._unique_values[self._unique_locs]
        v_u[v_u==lhs]=rhs
        v = np.zeros(self._size)
        v[self._flat_indices] = v_u
        self._unique_values, self._unique_locs, self._first_locs = self.unique_nonzero(v)
        self._flat_indices = nonzero(v).squeeze()
        self._n_nnz = len(self._flat_indices)
        self._stop_nonzero = self._n_nnz
        self._n_unz = len(self._unique_values)
        self._stop_unique = self._n_unz
        
                
    @property
    def stop(self):
        return self._stop + self._offset
        
    @property
    def start(self):
        return self._start + self._offset
    
    @property
    def stop_unique(self):
        return self._stop_unique + self._offset_unique
        
    @property
    def start_unique(self):
        return self._start_unique + self._offset_unique
    
    @property
    def stop_nonzero(self):
        return self._stop_nonzero  + self._offset_nonzero
        
    @property
    def start_nonzero(self):
        return self._start_nonzero + self._offset_nonzero
    
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
        return f"{self.start}:{self.stop} {self.start_nonzero}:{self.stop_nonzero} {self.start_unique}:{self.stop_unique}"

    
class BlockFlattenedIndicatorIndices(object):
    """
    This class represents a list of `FlattenedIndicatorIndices` objects in a unified form, 
    providing collective indexing for efficient storage and operations.

    Parameters
    ----------
    flat_objects : list of FlattenedIndicatorIndices objects
        List of `FlattenedIndicatorIndices` objects to be unified.
    shared_values : array-like, optional
        Array containing shared values for the unified `FlattenedIndicatorIndices` objects. 
        Default is an array of increasing numbers from 0 to the length of `flat_objects`.

    Attributes
    ----------
    _n_objs : int
        Number of `FlattenedIndicatorIndices` objects.
    _flat_objects : list of FlattenedIndicatorIndices objects
        List of `FlattenedIndicatorIndices` objects to be unified.
    _tril_inds : tuple of numpy.ndarray
        Tuple containing row and column indices of the lower triangle of the unified array.

    """
    def __init__(self, flat_objects, shared_values=None):
        self._n_objs = len(flat_objects)
        if shared_values is None:
            shared_values = np.arange(len(flat_objects))
        self._flat_objects = flat_objects
        for i in range(1, self._n_objs):
            self._flat_objects[i].add_offsets(self._flat_objects[i-1])
            self._flat_objects[i]._block_indices = i+ self._flat_objects[i]._block_indices
        self._tril_inds = tril_indices(self.n_nonzero)
    
    @property
    def unique_locs(self):
        return np.concatenate([obj.unique_locs for obj in self._flat_objects])
    
    @property
    def first_locs(self):
        return np.concatenate([obj.first_locs for obj in self._flat_objects])
    
    @property
    def flat_indices(self):
        return np.concatenate([obj.flat_indices for obj in self._flat_objects])
                                              
    @property
    def block_indices(self):
        return np.concatenate([obj._block_indices for obj in self._flat_objects])
    
    @property
    def n_nonzero(self):
        return sum([obj._n_nnz for obj in self._flat_objects])   
    
    @property
    def col_indices(self):
        return np.concatenate([obj._col_indices for obj in self._flat_objects])
                   
    @property
    def row_indices(self):
        return np.concatenate([obj._row_indices for obj in self._flat_objects])
    
    @property
    def slices(self):
        return [slice(obj.start, obj.stop) for obj in self._flat_objects]
    
    @property
    def slices_nonzero(self):
        return [slice(obj.start_nonzero, obj.stop_nonzero) for obj in self._flat_objects]                  
    
    @property
    def slices_unique(self):
        return [slice(obj.start_unique, obj.stop_unique) for obj in self._flat_objects]

    @property
    def shapes(self):
        return [obj._shape for obj in self._flat_objects]

    @property
    def is_symmetric(self):
        return [obj._symmetric for obj in self._flat_objects]
    
    @property
    def unique_indices(self):
        return unique(self.unique_locs)[2]
    
    def create_derivative_arrays(self, nonzero_cross_derivs=None):
        if nonzero_cross_derivs is None:
            nonzero_cross_derivs =list(zip(*tril_indices(self._n_objs)))
        
        deriv_shape = tuple(np.max(np.array(self.shapes), axis=0))
        dA = np.zeros((self.n_nonzero,)+deriv_shape)
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
        
        for ii, (i, j) in enumerate(nonzero_cross_derivs, 1):
            mask_i, mask_j = (block_i == i),  (block_j == j)
            block_pair_types[mask_i & mask_j] = ii
            
        self.dA = dA
        self.block_sizes = block_sizes
        self.block_types_paired = block_pairs
        self.block_pair_types = block_pair_types
        self.colex_descending_inds = np.vstack(self._tril_inds).T


                                  
