# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 23:32:16 2023

@author: lukepinkel
"""
import numpy as np
import scipy as sp
from . import indexing_utils
from .linalg_operations import (gb_diag, mat_size_to_lhv_size, mat_size_to_hv_size, 
                                lhv_indices, hv_indices, lhv_ind_parts, lower_half_vec,
                                inv_lower_half_vec, _invecl)
from .special_mats import lmat, nmat
from .dchol import dchol, unit_matrices
from . import unconstrained_chol 
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
            elif isinstance(index_obj, (tuple, list, np.ndarray)):
                start, stop = index_obj  # Unpack the tuple into start and stop
            elif isinstance(index_obj, slice):
                start, stop = index_obj.start, index_obj.stop
            else:
                raise ValueError("Invalid index object type")
            index_ranges.append((start, stop))
            start = stop
        return index_ranges

    def _apply_transform(self, params, method_name):
        input_len = params.shape[-1]
        if method_name in ('_fwd', '_rvs'):
            result = np.zeros_like(params)
        elif method_name in ('_jac_fwd', '_jac_rvs'):
            result = np.eye(input_len)
        elif method_name in ('_hess_fwd', '_hess_rvs'):
            result = np.zeros((input_len, input_len, input_len))

        prev_stop = 0
        for (start, stop), transform in zip(self.index_ranges, self.transforms):
            # Apply the identity transform for the range between the previous stop and the current start
            if prev_stop < start:
                if method_name in ('_fwd', '_rvs'):
                    result[..., prev_stop:start] = params[..., prev_stop:start]

            method = getattr(transform, method_name)
            sliced_params = params[..., start:stop]
            transformed_slice = method(sliced_params)
            
            if method_name in ('_fwd', '_rvs'):
                result[..., start:stop] = transformed_slice
            elif method_name in ('_jac_fwd', '_jac_rvs'):
                result[..., start:stop, start:stop] = transformed_slice
            elif method_name in ('_hess_fwd', '_hess_rvs'):
                result[..., start:stop, start:stop, start:stop] = transformed_slice

            prev_stop = stop

        # Apply the identity transform for the remaining range after the last stop
        if prev_stop < input_len:
            if method_name in ('_fwd', '_rvs'):
                result[..., prev_stop:] = params[..., prev_stop:]

        return result

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
    
def _hess_chain_rule(d2z_dy2, dy_dx, dz_dy, d2y_dx2):
    H1 = np.einsum("ijk,kl->ijl", d2z_dy2, dy_dx, optimize=True)
    H1 = np.einsum("ijk,jl->ilk", H1, dy_dx, optimize=True)
    H2 = np.einsum("ij,jkl->ikl", dz_dy, d2y_dx2, optimize=True)
    d2z_dx2 = H1 + H2
    return d2z_dx2

class ComposedTransform(ParameterTransformBase):
    def __init__(self, transforms):
        self.transforms = transforms

    def _fwd(self, params):
        for transform in self.transforms:
            params = transform.fwd(params)
        return params

    def _rvs(self, transformed_params):
        for transform in reversed(self.transforms):
            transformed_params = transform.rvs(transformed_params)
        return transformed_params

    def _jac_fwd(self, params):
        jacobian = np.eye(params.shape[0])
        for transform in self.transforms:
            jac = transform.jac_fwd(params)
            params = transform.fwd(params)
            jacobian = np.dot(jac,  jacobian)
        return jacobian

    def _jac_rvs(self, transformed_params):
        jacobian = np.eye(transformed_params.shape[0])
        for transform in reversed(self.transforms):
            jac = transform.jac_rvs(transformed_params)
            transformed_params = transform.rvs(transformed_params)
            jacobian = np.dot(jac,  jacobian)
        return jacobian

    def _hess_fwd(self, x0):
                
        input_vars = [x0]
        for transform in self.transforms:
            input_vars.append(transform._fwd(input_vars[-1]))
        
        jacobians = []
        for i, transform in enumerate(self.transforms):
            jacobians.append(transform._jac_fwd(input_vars[i]))
        
        jacs = [jacobians[0]]
        for i in range(1, len(self.transforms)-1):
            jacs.append(jacobians[i].dot(jacs[i-1]))
        
        hessians = []
        for i, transform in enumerate(self.transforms):
            hessians.append(transform._hess_fwd(input_vars[i]))
        
        
        hess = [hessians[0]]
        for i in range(1, len(self.transforms)):
            hess.append(_hess_chain_rule(hessians[i], jacs[i-1], jacobians[i], hess[i-1]))
        return hess[-1]

    def _hess_rvs(self, u):
        output_vars = [u]
        for transform in reversed(self.transforms):
            output_vars.append(transform._rvs(output_vars[-1]))
        
        jacobians = []
        for i, transform in enumerate(reversed(self.transforms)):
            jacobians.append(transform._jac_rvs(output_vars[i]))
        
        jacs = [jacobians[0]]
        for i in range(1, len(self.transforms) - 1):
            jacs.append(jacobians[i].dot(jacs[i - 1]))
        
        hessians = []
        for i, transform in enumerate(reversed(self.transforms)):
            hessians.append(transform._hess_rvs(output_vars[i]))
        
        hess = [hessians[0]]
        for i in range(1, len(self.transforms)):
            hess.append(_hess_chain_rule(hessians[i], jacs[i - 1], jacobians[i], hess[i - 1]))
        return hess[-1]

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


class IdentityTransform(ParameterTransformBase):
    @staticmethod
    def _fwd(params):
        return params

    @staticmethod
    def _rvs(transformed_params):
        return transformed_params

    @staticmethod
    def _jac_rvs(transformed_params):
        if np.ndim(transformed_params) == 0:
            return 1
        else:
            n = len(transformed_params)
            J = np.eye(n)
            return J

    @staticmethod
    def _jac_fwd(params):
        if np.ndim(params) == 0:
            return 1
        else:
            n = len(params)
            J = np.eye(n)
            return J

    @staticmethod
    def _hess_rvs(transformed_params):
        if np.ndim(transformed_params) == 0:
            return 0
        else:
            n = len(transformed_params)
            hess = np.zeros((n, n, n))
            return hess

    @staticmethod
    def _hess_fwd(params):
        if np.ndim(params) == 0:
            return 0
        else:
            n = len(params)
            hess = np.zeros((n, n, n))
            return hess


class UnconstrainedCholeskyCorr(ParameterTransformBase):
    
    
    def __init__(self, mat_size):
        self.mat_size = mat_size
        
    @staticmethod
    def _fwd(x):
        return unconstrained_chol.fwd(x)
        
    @staticmethod
    def _rvs(y):
        return unconstrained_chol.rvs(y)
    
    @staticmethod
    def _jac_fwd(x):
        return unconstrained_chol.jac_fwd(x)
        
    @staticmethod
    def _jac_rvs(y):
        return unconstrained_chol.jac_rvs(y)
    
    @staticmethod
    def _hess_fwd(x):
        return unconstrained_chol.hess_fwd(x)
    
    @staticmethod
    def _hess_rvs(y):
        return unconstrained_chol.hess_rvs(y)
    
    
class CovCorr(ParameterTransformBase):
    
    def __init__(self, mat_size):
        self.mat_size = mat_size
        self.hv_size = mat_size_to_hv_size(mat_size)
        self.row_inds, self.col_inds = hv_indices((mat_size, mat_size))
        self.diag_inds, = np.where(self.row_inds==self.col_inds)
        self.tril_inds, = np.where(self.row_inds!=self.col_inds)
        self.row_diag_inds = self.diag_inds[self.row_inds]
        self.col_diag_inds = self.diag_inds[self.col_inds]
        ii = self.row_diag_inds!=self.col_diag_inds
        self.dr_perm = np.vstack((self.col_diag_inds[ii], self.row_diag_inds[ii])).T.flatten()
        self.dc_perm = np.repeat(self.tril_inds, 2)
        self.ii = ii

        
    def _fwd(self, x):
        sj = np.sqrt(x[self.row_diag_inds])
        sk = np.sqrt(x[self.col_diag_inds])
        y = x / (sj * sk)
        y[self.diag_inds] = x[self.diag_inds] #np.log(x[self.diag_inds])
        return y
    
    def _rvs(self, y):
        sj = np.sqrt(y[self.row_diag_inds])
        sk = np.sqrt(y[self.col_diag_inds])
        x = y * sj * sk
        x[self.diag_inds] = y[self.diag_inds]
        return x
    
    def _jac_fwd(self, x):
        dy_dx = np.zeros((x.shape[0],)*2)
        sj = np.sqrt(x[self.row_diag_inds])
        sk = np.sqrt(x[self.col_diag_inds])
        t1 = 1 / (sj*sk)
        t2 = -1.0 / 2.0 * x / (sj**3 * sk)
        t3 = -1.0 / 2.0 * x / (sk**3 * sj)
        t = np.vstack((t3, t2)).T[self.ii].flatten()
        dy_dx[self.diag_inds, self.diag_inds] = 1.0# / x[self.diag_inds]
        dy_dx[self.tril_inds, self.tril_inds] = t1[self.tril_inds]
        dy_dx[self.dc_perm, self.dr_perm] = t
        return dy_dx
    
    def _jac_rvs(self, y):
        dx_dy = np.zeros((y.shape[0],)*2)
        sj = np.sqrt(y[self.row_diag_inds])
        sk = np.sqrt(y[self.col_diag_inds])
        t1 = (sj*sk)
        t2 = 1.0 / 2.0 * y * sk / sj
        t3 = 1.0 / 2.0 * y * sj / sk
        t = np.vstack((t3, t2)).T[self.ii].flatten()
        dx_dy[self.diag_inds, self.diag_inds] = 1.0#np.exp(y[self.diag_inds])
        dx_dy[self.tril_inds, self.tril_inds] = t1[self.tril_inds]
        dx_dy[self.dc_perm, self.dr_perm] = t
        return dx_dy
    
    def _hess_fwd(self, x):
        rix, cix = self.row_diag_inds, self.col_diag_inds
        d2y_dx2 = np.zeros((self.hv_size,)*3)
        for i in range(self.hv_size):
            j, k = rix[i], cix[i]
            if j!=k:
                xi, xj, xk = x[i], x[j], x[k] #w, y, z = x[i], x[j], x[k]
                sjk =  np.sqrt(xj * xk)       #syz = np.sqrt(y * z)
                d2yi_dxidxj = -xk / (2.0 * sjk**3) #d2y_dwdy = -z / (2.0 * syz**3)
                d2yi_dxidxk = -xj / (2.0 * sjk**3) #d2y_dwdz = -y / (2.0 * syz**3)
                d2yi_dxjdxj = (3.0 * xi * xk**2)   / (4.0 * sjk**5)  #d2y_dydy = (3.0 * w * z**2)  / (4.0 * syz**5)
                d2yi_dxjdxk = (3.0 * xi * xj * xk) / (4.0 * sjk**5) - xi / (2.0 * sjk**3) #d2y_dydz = (3.0 * w * y * z) / (4.0 * syz**5) - (w) / (2.0 * syz**3)
                d2yi_dxkdxk = (3.0 * xi * xj**2)   / (4.0 * sjk**5) #d2y_dzdz = (3.0 * w * y**2)  / (4.0 * syz**5)
                
                d2y_dx2[i, i, j] = d2y_dx2[i, j, i] = d2yi_dxidxj
                d2y_dx2[i, i, k] = d2y_dx2[i, k, i] = d2yi_dxidxk
                d2y_dx2[i, j, j] = d2yi_dxjdxj
                d2y_dx2[i, j, k] = d2y_dx2[i, k, j] = d2yi_dxjdxk
                d2y_dx2[i, k, k] = d2yi_dxkdxk
        return d2y_dx2
    
    def _hess_rvs(self, y):
        rix, cix = self.row_diag_inds, self.col_diag_inds
        d2x_dy2 = np.zeros((self.hv_size,)*3)
        for i in range(self.hv_size):
            j, k = rix[i], cix[i]
            if j!=k:
               yi, yj, yk = y[i], y[j], y[k]  
               d2xi_dyidyj = np.sqrt(yk) / (2.0 * np.sqrt(yj))
               d2xi_dyidyk = np.sqrt(yj) / (2.0 * np.sqrt(yk))
               d2xi_dyjdyj = (-yi * np.sqrt(yk)) / (4.0 * np.sqrt(yj)**3)
               d2xi_dyjdyk = yi / (4.0 * np.sqrt(yj * yk))
               d2xi_dykdyk = (-yi * np.sqrt(yj)) / (4.0 * np.sqrt(yk)**3)
               
               d2x_dy2[i, i, j] = d2x_dy2[i, j, i] = d2xi_dyidyj
               d2x_dy2[i, i, k] = d2x_dy2[i, k, i] = d2xi_dyidyk
               d2x_dy2[i, j, j] = d2xi_dyjdyj
               d2x_dy2[i, j, k] = d2x_dy2[i, k, j] = d2xi_dyjdyk
               d2x_dy2[i, k, k] = d2xi_dykdyk
        return d2x_dy2
        
   
class LogScale(ParameterTransformBase):
    
    def __init__(self, mat_size):
        self.mat_size = mat_size
        self.hv_size = mat_size_to_hv_size(mat_size)
        self.row_inds, self.col_inds = hv_indices((mat_size, mat_size))
        self.diag_inds, = np.where(self.row_inds==self.col_inds)
        self.tril_inds, = np.where(self.row_inds!=self.col_inds)
    
    def _fwd(self, x):
        ix = self.diag_inds
        y = x.copy()
        y[ix] = np.log(x[ix])
        return y
    
    def _rvs(self, y):
        ix = self.diag_inds
        x = y.copy()
        x[ix] = np.exp(y[ix])
        return x
        
    def _jac_fwd(self, x):
        ix = self.diag_inds
        dy_dx = np.eye(x.shape[-1])
        dy_dx[ix, ix] = 1 / x[ix]
        return dy_dx
    
    def _jac_rvs(self, y):
        ix = self.diag_inds
        dx_dy = np.eye(y.shape[-1])
        dx_dy[ix, ix] = np.exp(y[ix])
        return dx_dy
    
    def _hess_fwd(self, x):
        ix = self.diag_inds
        d2y_dx2 = np.zeros((x.shape[-1],)*3)
        d2y_dx2[ix, ix, ix] = -1 / x[ix]**2
        return d2y_dx2
        
    def _hess_rvs(self, y):
        ix = self.diag_inds
        d2x_dy2 = np.zeros((y.shape[-1],)*3)
        d2x_dy2[ix, ix, ix] = np.exp(y[ix])
        return d2x_dy2
    
    
class CorrCholesky(ParameterTransformBase):
    
    def __init__(self, mat_size):
        self.n = self.mat_size = mat_size
        self.m = self.vec_size = int((mat_size + 1) * mat_size / 2)
        self.I = np.eye(self.n)
        self.N = nmat(self.n)
        self.E = lmat(self.n)
        self.diag_inds = np.diag_indices(self.mat_size)
        self.lhv_size = mat_size_to_lhv_size(mat_size)
        self.row_inds, self.col_inds = lhv_indices((mat_size, mat_size))
        self.row_sort = np.argsort(self.row_inds)
        self.ind_parts = lhv_ind_parts(mat_size)
        self.row_norm_inds = [self.row_sort[a:b] for a, b in self.ind_parts]
        j, i = np.triu_indices(self.n)
        self.non_diag, = np.where(j!=i)
        self.dM, self.d2M = unit_matrices(mat_size)
        row_inds, col_inds = hv_indices((self.mat_size, self.mat_size))
        self.tril_inds, = np.where(row_inds!=col_inds)
        
    def _fwd(self, x):
        R = _invecl(x)
        L = np.linalg.cholesky(R)
        y = lower_half_vec(L)
        return y
    
    def _rvs(self, y):
        L = inv_lower_half_vec(y)
        L[self.diag_inds] = np.sqrt(1-np.linalg.norm(L, axis=-1)**2)
        R = np.dot(L, L.T)
        x = lower_half_vec(R)
        return x
    
    def _jac_fwd(self, x):
        M = _invecl(x)
        _, dL, _ = dchol(M, self.dM, self.d2M, order=1)
        dL = dL.reshape(np.prod(dL.shape[:2]), dL.shape[2], order='F')
        dy_dx = self.E.dot(dL)[np.ix_(self.tril_inds, self.tril_inds)]
        return dy_dx
    
    def _jac_rvs(self, y):
        x = self._rvs(y)
        dx_dy = np.linalg.inv(self._jac_fwd(x))
        return dx_dy
    
    def _hess_fwd(self, x):
        M = _invecl(x)
        _, _, d2L = dchol(M, self.dM, self.d2M, order=2)
        k = (np.prod(d2L.shape[:2]),)
        d2L = d2L.reshape(k+d2L.shape[2:], order='F')
        d2L = d2L[self.E.tocoo().col]
        d2y_dx2 = d2L[np.ix_(self.tril_inds, self.tril_inds, self.tril_inds)]
        return d2y_dx2
    
    def _hess_rvs(self, y):
        x = self._rvs(y)
        d2y_dx2 = self._hess_fwd(x)
        dx_dy = self._jac_rvs(y)
        d2x_dy2 = np.einsum("rst,ir,sj,tk->ijk", -d2y_dx2, dx_dy, dx_dy, dx_dy, optimize=True)
        return d2x_dy2
        

class OffDiagMask(ParameterTransformBase):
    
    def __init__(self, transform):
        self.mat_size = transform.mat_size
        self.hv_size = mat_size_to_hv_size(self.mat_size)
        self.row_inds, self.col_inds = hv_indices((self.mat_size, self.mat_size))
        self.diag_inds, = np.where(self.row_inds==self.col_inds)
        self.tril_inds, = np.where(self.row_inds!=self.col_inds)
        self.transform = transform
        
    def _fwd(self, x):
        y = x.copy()
        y[self.tril_inds] = self.transform._fwd(y[self.tril_inds])
        return y
    
    def _rvs(self, y):
        x = y.copy()
        x[self.tril_inds] = self.transform._rvs(x[self.tril_inds])
        return x
    
    def _jac_fwd(self, x):
        dy_dx = np.zeros((self.hv_size, self.hv_size))
        ii, ij = self.diag_inds, self.tril_inds
        dy_dx[np.ix_(ij, ij)] = self.transform._jac_fwd(x[ij].copy())
        dy_dx[np.ix_(ii, ii)] = np.eye(len(ii))
        return dy_dx
    
    def _jac_rvs(self, y):
        dx_dy = np.zeros((self.hv_size, self.hv_size))
        ii, ij = self.diag_inds, self.tril_inds
        dx_dy[np.ix_(ij, ij)] = self.transform._jac_rvs(y[ij].copy())
        dx_dy[np.ix_(ii, ii)] = np.eye(len(ii))
        return dx_dy
    
    def _hess_fwd(self, x):
        d2y_dx2 = np.zeros((self.hv_size, self.hv_size, self.hv_size))
        ij = self.tril_inds
        d2y_dx2[np.ix_(ij, ij, ij)] = self.transform._hess_fwd(x[ij].copy())
        return d2y_dx2
    
    def _hess_rvs(self, y):
        d2x_dy2 = np.zeros((self.hv_size, self.hv_size, self.hv_size))
        ij = self.tril_inds
        d2x_dy2[np.ix_(ij, ij, ij)] = self.transform._hess_rvs(y[ij].copy())
        return d2x_dy2

  
    
class CholeskyCov(ComposedTransform):
    
    def __init__(self, mat_size):
        self.covn_to_corr = CovCorr(mat_size)
        self.corr_to_chol = OffDiagMask(CorrCholesky(mat_size))
        self.chol_to_real = OffDiagMask(UnconstrainedCholeskyCorr(mat_size))
        self.vars_to_logs = LogScale(mat_size)
        tanh = TanhTransform()
        tanh.mat_size = mat_size
        self.tanh = OffDiagMask(tanh)
        
        super().__init__(
            [self.covn_to_corr,
            self.corr_to_chol,
            self.chol_to_real,
            self.tanh,
            self.vars_to_logs,
            ]
        )


