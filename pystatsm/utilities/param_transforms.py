# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 23:32:16 2023

@author: lukepinkel
"""
import numpy as np
import scipy as sp
from . import indexing_utils

class OrderedTransform(object):
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
    _rvs(y, q):
        Perform the reverse transformation from y to x.
    _fwd(x, q):
        Perform the forward transformation from x to y.
    _jac_rvs(y, q):
        Calculate the Jacobian of the reverse transformation.
    _hess_rvs(y, q):
        Calculate the Hessian of the reverse transformation.
    _jac_fwd(x, q):
        Calculate the Jacobian of the forward transformation.
    _hess_fwd(x, q):
        Calculate the Hessian of the forward transformation.
    """
    @staticmethod
    def _rvs(y, q):
        """
        Perform the reverse transformation from y to x.
        
        Parameters
        ----------
        y : numpy.ndarray
            A sequence of real numbers in R^q.
        q : int
            The length of the sequence y.
        
        Returns
        -------
        x : numpy.ndarray
            The transformed ordered sequence in R^q.
        """
        x = y.copy()
        x[1:q] = np.cumsum(np.exp(y[1:q]))+x[0]
        return x

    @staticmethod
    def _fwd(x, q):
        """
        Perform the forward transformation from x to y.
        
        Parameters
        ----------
        x : numpy.ndarray
            An ordered sequence of real numbers in R^q.
        q : int
            The length of the sequence x.
        
        Returns
        -------
        y : numpy.ndarray
            The transformed sequence in R^q.
        """
        y = x.copy()
        y[1:q] = np.log(np.diff(x[:q]))
        return y
    
    
    @staticmethod
    def _jac_rvs(y, q):
        """
        Calculate the Jacobian of the reverse transformation.
        Parameters
        ----------
        y : numpy.ndarray
            A sequence of real numbers in R^q.
        q : int
            The length of the sequence y.
        
        Returns
        -------
        dx_dy : numpy.ndarray
            The Jacobian matrix of the reverse transformation (shape: (q, q)).
        """
        dx_dy = np.zeros((q, q))
        inds = indexing_utils.tril_indices(q)
        vals = np.repeat(np.r_[1, np.exp(y[1:q])], np.arange(q, 0, -1))
        dx_dy[inds] = vals
        # dx_dy[:, 0] = 1.0
        # for i in range(1, q):
        #     dx_dy[i:, i] = np.exp(y[i])
        return dx_dy

    @staticmethod
    def _hess_rvs(y, q):
        """
        Calculate the Hessian of the reverse transformation.
        
        Parameters
        ----------
        y : numpy.ndarray
            A sequence of real numbers in R^q.
        q : int
            The length of the sequence y.
        
        Returns
        -------
        d2x_dy2 : numpy.ndarray
            The Hessian tensor of the reverse transformation (shape: (q, q, q)).
        """
        d2x_dy2 = np.zeros((q, q, q))
        z = np.exp(y)
        for i in range(1, q):
            for j in range(1, i+1):
                d2x_dy2[i, j, j] = z[j]
        return d2x_dy2
    

    @staticmethod
    def _jac_fwd(x, q):
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
    def _hess_fwd(x, q):
        """
        Calculate the Hessian of the forward transformation.
        
        Parameters
        ----------
        x : numpy.ndarray
            An ordered sequence of real numbers in R^q.
        q : int
            The length of the sequence x.
        
        Returns
        -------
        d2y_dx2 : numpy.ndarray
            The Hessian tensor of the forward transformation (shape: (q, q, q)).
        """
        d2y_dx2 = np.zeros((q, q, q))
        for i in range(1, q):
            d2y_dx2[i, i, i] = -1 / (x[i] - x[i - 1])**2
            d2y_dx2[i, i - 1, i - 1] = -1 / (x[i] - x[i - 1])**2
            d2y_dx2[i, i, i - 1] = 1 / (x[i] - x[i - 1])**2
            d2y_dx2[i, i - 1, i] = d2y_dx2[i, i, i - 1]  # Use symmetry
        return d2y_dx2



