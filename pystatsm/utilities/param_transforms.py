# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 23:32:16 2023

@author: lukepinkel
"""
import numpy as np
import scipy as sp
from . import indexing_utils
from .linalg_operations import gb_diag

from abc import ABC, abstractmethod


class ParameterTransformBase(ABC):
    
    @staticmethod
    @abstractmethod
    def _fwd(self, params):
        """Perform forward transform"""
        pass
    
    @staticmethod
    @abstractmethod
    def _rvs(self, transformed_params):
        """Perform reverse transform"""
        pass
    
    @staticmethod
    @abstractmethod
    def _jac_fwd(self, params):
        """Compute forward transform first derivative"""
        pass
    
    @staticmethod
    @abstractmethod
    def _jac_rvs(self, transformed_params):
        """Compute reverse transform first derivative"""
        pass
    
    @staticmethod
    @abstractmethod
    def _hess_fwd(self, params):
        """Compute forward transform second derivative"""
        pass
    
    @staticmethod
    @abstractmethod
    def _hess_rvs(self, transformed_params):
        """Compute reverse transform second derivative"""
        pass

    def _apply_method_unreadable(self, x, method):
        input_shape = x.shape[:-1] if x.ndim > 1 else ()
        x_iter = np.nditer(x, flags=['multi_index', 'refs_ok']) if x.ndim > 1 else iter([x])
        output_shape = input_shape + (getattr(self, method)(x[0]).shape[0],) if x.ndim > 1 else (getattr(self, method)(x).shape[0],)
        res = np.empty(output_shape, dtype=x.dtype)
        for x_val in x_iter:
            idx = x_iter.multi_index if x.ndim > 1 else ()
            res[idx] = getattr(self, method)(x_val)
        return res
    
    def _apply_method(self, x, method):
        if np.isscalar(x):
            return getattr(self, method)(x)
        ndim = x.ndim
        func = getattr(self, method)
        if ndim==1:
            out = func(x)
        else:
            out = func(x[(0,)*(ndim-1)])
        output_shape = out.shape if x.ndim==1 else  x.shape[:-1] + out.shape 

        res = np.empty(output_shape, dtype=x.dtype)

        if x.ndim == 1:
            res[...] = out
        else:
            for index in np.ndindex(x.shape[:-1]):
                res[index] = func(x[index])
        
        return res

    
    def fwd(self, x):
        return self._apply_method(x, '_fwd')

    def rvs(self, y):
        return self._apply_method(y, '_rvs')
    
    def jac_fwd(self, x):
        return self._apply_method(x, '_jac_fwd')

    def jac_rvs(self, y):
        return self._apply_method(y, '_jac_rvs')
    
    def hess_fwd(self, x):
        return self._apply_method(x, '_hess_fwd')

    def hess_rvs(self, y):
        return self._apply_method(y, '_hess_rvs')


class CombinedTransform(ParameterTransformBase):
    def __init__(self, transforms, index_objects):
        self.transforms = transforms
        self.index_ranges = self._process_index_objects(index_objects)

    def _process_index_objects(self, index_objects):
        index_ranges = []
        start = 0
        for index_obj in index_objects:
            if isinstance(index_obj, (int, np.integer)):
                stop = start + index_obj
            elif isinstance(index_obj, (slice, tuple, list, np.ndarray)):
                stop = index_obj[-1] + 1
            else:
                raise ValueError("Invalid index object type")
            index_ranges.append((start, stop))
            start = stop
        return index_ranges

    def _apply_transform(self, params, method_name):
        transformed_params_list = []
        for (start, stop), transform in zip(self.index_ranges, self.transforms):
            method = getattr(transform, method_name)
            transformed_params_list.append(method(params[..., start:stop]))

        if method_name in ('_fwd', '_rvs'):
            return np.concatenate(transformed_params_list, axis=-1)

        if method_name in ('_jac_fwd', '_jac_rvs'):
            return gb_diag(*transformed_params_list)

        if method_name in ('_hess_fwd', '_hess_rvs'):
            return  gb_diag(*transformed_params_list)

    def _fwd(self, params):
        return self._apply_transform(params, '_fwd')

    def _rvs(self, transformed_params):
        return self._apply_transform(transformed_params, '_rvs')

    def _jac_fwd(self, params):
        return self._apply_transform(params, '_jac_fwd')

    def _jac_rvs(self, transformed_params):
        return self._apply_transform(transformed_params, '_jac_rvs')

    def _hess_fwd(self, params):
        return self._apply_transform(params, '_hess_fwd')

    def _hess_rvs(self, transformed_params):
        return self._apply_transform(transformed_params, '_hess_rvs')
    

class OrderedTransform(ParameterTransformBase):
    """
    A class that performs a transformation on ordered sequences of real numbers 
    in R^q.
    
    The forward transformation converts an ordered sequence x into a new 
    sequence y using the differences between consecutive elements and the
    logarithm. The reverse transformation converts the sequence y back to the
    original ordered sequence x using the exponential function and
    cumulative sums.

    The domain of the forward transformation is the set of x in R^q such that
    
        x_{i} < x_{i+1} for i=0, 1, ..., q-2. 
    
    The range of the forward transformation is the set of sequences in R^q such
    that y_0 is unrestricted and y_i is greater than or equal to negative
    infinity for i=1, 2, ..., q-1.

    The domain of the reverse transformation is the range of the forward 
    transform and its range is the domain of the forward transform

    The class also calculates the Jacobian (`_jac_fwd` and `_jac_rvs`) 
    and Hessian (`_hess_fwd` and `_hess_rvs`) of the forward and
    reverse transformations.

    Attributes
    ----------
    None
    
    Methods
    -------
    _rvs(y):
        Perform the reverse transformation from y to x.
    _fwd(x):
        Perform the forward transformation from x to y.
    _jac_rvs(y):
        Calculate the Jacobian of the reverse transformation.
    _hess_rvs(y):
        Calculate the Hessian of the reverse transformation.
    _jac_fwd(x):
        Calculate the Jacobian of the forward transformation.
    _hess_fwd(x):
        Calculate the Hessian of the forward transformation.
    """
    
    @staticmethod
    def _rvs(y):
        """
        Perform the reverse transformation from y to x.
        
        Parameters
        ----------
        y : numpy.ndarray
            A sequence of real numbers in R^q.
        Returns
        -------
        x : numpy.ndarray
            The transformed ordered sequence in R^q.
        """
        q = len(y)
        x = y.copy()
        x[1:q] = np.cumsum(np.exp(y[1:q]))+x[0]
        return x

    @staticmethod
    def _fwd(x):
        """
        Perform the forward transformation from x to y.
        
        Parameters
        ----------
        x : numpy.ndarray
            An ordered sequence of real numbers in R^q.
        
        Returns
        -------
        y : numpy.ndarray
            The transformed sequence in R^q.
        """
        q = len(x)
        y = x.copy()
        y[1:q] = np.log(np.diff(x[:q]))
        return y
    
    
    @staticmethod
    def _jac_rvs(y):
        """
        Calculate the Jacobian of the reverse transformation.
        Parameters
        ----------
        y : numpy.ndarray
            A sequence of real numbers in R^q.

        -------
        dx_dy : numpy.ndarray
            The Jacobian matrix of the reverse transformation (shape: (q, q)).
        """
        q = len(y)
        dx_dy = np.zeros((q, q))
        inds = indexing_utils.tril_indices(q)
        vals = np.repeat(np.r_[1, np.exp(y[1:q])], np.arange(q, 0, -1))
        dx_dy[inds] = vals
        # dx_dy[:, 0] = 1.0
        # for i in range(1, q):
        #     dx_dy[i:, i] = np.exp(y[i])
        return dx_dy

    @staticmethod
    def _hess_rvs(y):
        """
        Calculate the Hessian of the reverse transformation.
        
        Parameters
        ----------
        y : numpy.ndarray
            A sequence of real numbers in R^q.

        Returns
        -------
        d2x_dy2 : numpy.ndarray
            The Hessian tensor of the reverse transformation (shape: (q, q, q)).
        """
        q = len(y)
        d2x_dy2 = np.zeros((q, q, q))
        z = np.exp(y)
        for i in range(1, q):
            for j in range(1, i+1):
                d2x_dy2[i, j, j] = z[j]
        return d2x_dy2
    

    @staticmethod
    def _jac_fwd(x):
        """
        Calculate the Jacobian of the forward transformation.
        
        Parameters
        ----------
        x : numpy.ndarray
            An ordered sequence of real numbers in R^q.
        q : int
            The length of the sequence x.
        
        Returns
        -------
        dy_dx : numpy.ndarray
            The Jacobian matrix of the forward transformation (shape: (q, q)).
        """
        q = len(x)
        dy_dx = np.zeros((q, q))
        dy_dx[0, 0] = 1
        ii = np.arange(1, q)
        dy_dx[ii, ii] = 1 / (x[ii] - x[ii-1])
        dy_dx[ii, ii-1] =  -1 / (x[ii] - x[ii - 1])
        # for i in range(1, q):
        #     dy_dx[i, i] = 1 / (x[i] - x[i - 1])
        #     dy_dx[i, i - 1] = -1 / (x[i] - x[i - 1])
        return dy_dx
    
    @staticmethod
    def _hess_fwd(x):
        """
        Calculate the Hessian of the forward transformation.
        
        Parameters
        ----------
        x : numpy.ndarray
            An ordered sequence of real numbers in R^q.
        
        Returns
        -------
        d2y_dx2 : numpy.ndarray
            The Hessian tensor of the forward transformation (shape: (q, q, q)).
        """
        q = len(x)
        d2y_dx2 = np.zeros((q, q, q))
        for i in range(1, q):
            d2y_dx2[i, i, i] = -1 / (x[i] - x[i - 1])**2
            d2y_dx2[i, i - 1, i - 1] = -1 / (x[i] - x[i - 1])**2
            d2y_dx2[i, i, i - 1] = 1 / (x[i] - x[i - 1])**2
            d2y_dx2[i, i - 1, i] = d2y_dx2[i, i, i - 1]  # Use symmetry
        return d2y_dx2


class LogTransform(ParameterTransformBase):
    
    @staticmethod
    def _fwd(params):
        return np.log(params)

    @staticmethod
    def _rvs(transformed_params):
        return np.exp(transformed_params)

    @staticmethod
    def _jac_fwd(params):
        if np.ndim(params) == 0:
            return 1 / params
        else:
            return np.diag(1 / params)


    @staticmethod
    def _jac_rvs(transformed_params):
        if np.ndim(transformed_params) == 0:
            return np.exp(transformed_params)
        else:
            return np.diag(np.exp(transformed_params))

    @staticmethod
    def _hess_fwd(params):
        if np.ndim(params) == 0:
            return -1 / (params**2)
        else:
            n = len(params)
            hess = np.zeros((n, n, n))
            hess[np.arange(n), np.arange(n), np.arange(n)] = -1 / (params**2)
            return hess
    @staticmethod
    def _hess_rvs(transformed_params):
        if np.ndim(transformed_params) == 0:
            return np.exp(transformed_params)
        else:
            n = len(transformed_params)
            hess = np.zeros((n, n, n))
            hess[np.arange(n), np.arange(n), np.arange(n)] = np.exp(transformed_params)
            return hess

class TanhTransform(ParameterTransformBase):
    @staticmethod
    def _fwd(params):
        return np.arctanh(params)

    @staticmethod
    def _rvs(transformed_params):
        return np.tanh(transformed_params)

    @staticmethod
    def _jac_rvs(transformed_params):
        if np.ndim(transformed_params) == 0:
            return 1 - np.tanh(transformed_params) ** 2
        else:
            return np.diag(1 - np.tanh(transformed_params) ** 2)

    @staticmethod
    def _jac_fwd(params):
        if np.ndim(params) == 0:
            return 1 / (1 - params ** 2)
        else:
            return np.diag(1 / (1 - params ** 2))

    @staticmethod
    def _hess_rvs(transformed_params):
        if np.ndim(transformed_params) == 0:
            return -2 * np.tanh(transformed_params) * (1 - np.tanh(transformed_params) ** 2)
        else:
            n = len(transformed_params)
            hess = np.zeros((n, n, n))
            hess[np.arange(n), np.arange(n), np.arange(n)] = -2 * np.tanh(transformed_params) * (1 - np.tanh(transformed_params) ** 2)
            return hess

    @staticmethod
    def _hess_fwd(params):
        if np.ndim(params) == 0:
            return 2 * params / (1 - params ** 2) ** 2
        else:
            n = len(params)
            hess = np.zeros((n, n, n))
            hess[np.arange(n), np.arange(n), np.arange(n)] = 2 * params / (1 - params ** 2) ** 2
            return hess

