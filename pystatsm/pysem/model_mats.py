#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 01:22:49 2023

@author: lukepinkel
"""


import numpy as np
from ..utilities.indexing_utils import nonzero, unique
from ..utilities.linalg_operations import _vec, _vech, _invech, _invec
from ..utilities.func_utils import triangular_number


class FlattenedIndicatorIndices(object):
    
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
        self._flat_indices = nonzero(v).squeeze()
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
        u_vals, u_locs, f_locs = unique(v)
        mask = u_vals!=0
        zloc = np.where(u_vals==0)[0]
        u_vals = u_vals[mask]
        if len(zloc)>0:
            zmask = u_locs!=zloc
        else:
            zmask = np.ones(len(u_locs), dtype=bool)
        u_locs = u_locs[zmask]
        u_locs[u_locs>zloc] =  u_locs[u_locs>zloc] - 1
        f_locs = f_locs[mask]
        return u_vals, u_locs, f_locs
    
    def update_offsets(self, **kwargs):
        for key, val in kwargs.items():
            if key in ["offset_nonzero", "offset_unique", "offset"]:
                setattr(self, key, val)
        
    def add_offsets(self, flat_index_obj):
        for name in ["_nonzero", "_unique", ""]:
            val = getattr(flat_index_obj, "stop"+name)
            setattr(self, "_offset"+name, val)
      
    def to_vec(self):
        v_u=self._unique_values[self._unique_locs]
        v = np.zeros(self._size)
        v[self._flat_indices] = v_u
        return v
    
    def to_array(self):
        v = self.to_vec()
        if self._symmetric:
            arr = _invech(v)
        else:
            arr = _invec(v, *self._shape)
        return arr
        
    def set_equal(self, lhs, rhs):
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
    
    def __init__(self, flat_objects, shared_values=None):
        self._n_objs = len(flat_objects)
        if shared_values is None:
            shared_values = np.arange(len(flat_objects))
        self._flat_objects = flat_objects
        for i in range(1, self._n_objs):
            self._flat_objects[i].add_offsets(self._flat_objects[i-1])
    
    @property
    def unique_locs(self):
        return np.concatenate([obj.unique_locs for obj in self._flat_objects])
    
    @property
    def first_locs(self):
        return np.concatenate([obj.first_locs for obj in self._flat_objects])
                           
    @property
    def flat_indices(self):
        return np.concatenate([obj.flat_indices for obj in self._flat_objects])
                           
      
