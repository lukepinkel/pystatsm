#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 01:18:16 2023

@author: lukepinkel
"""

import patsy

import numpy as np
import scipy as sp
import scipy.stats
import pandas as pd
from ..utilities.formula import design_matrices                 
from ..utilities import optimizer_utils, func_utils


class ModelData(object):
    """
    A class for storing and managing model data, including input variables,
    output variables, and weights.
    
    Attributes
    ----------
    data : list
        A list of arrays or dataframes, each representing an input or 
        output variable.
    design_info : list
        A list of design_info objects corresponding to each variable in `data`.
    indexes : list
        A list of indexes for each input or output variable.
    columns : list
        A list of column names for each input or output variable.
    names : list
        A list of names for each input or output variable.
    weights : array_like, optional
        An array of weights for the data, default is None.
    """
    def __init__(self, *args, weights=None):
        """
        Initialize the ModelData object.
        
        Parameters
        ----------
        *args : variable number of input/output variables
            Input and output variables as pandas DataFrames, Series, or numpy arrays.
        weights : array_like, optional
            An array of weights for the data, default is None.
        """
        self.data = []
        self.design_info = []
        self.indexes = []
        self.columns = []
        self.names = []
        self.weights = weights
        for arg in args:
            self.store(arg)
            
    @property
    def has_weights(self):
        """
        Check if the ModelData object has weights.
    
        Returns
        -------
        bool
            True if the object has weights, False otherwise.
        """
        return False if self.weights is None else True
    
    def store(self, data, varname="x"):
        """
        Store the input or output variable as a numpy array and update attributes.
        
        Parameters
        ----------
        data : pd.DataFrame, pd.Series, or np.ndarray
            Input or output variable.
        varname : str, optional
            Variable name prefix for numpy arrays, default is "x".
        """
        if isinstance(data, (pd.DataFrame, pd.Series)):
            self.indexes.append(data.index)
            if isinstance(data, pd.DataFrame):
                self.columns.append(data.columns)
                self.data.append(data.to_numpy())
                if hasattr(data, "design_info"):
                    design_info = data.design_info
                else:
                    design_info = None
                self.design_info.append(design_info)
            else:  # pd.Series
                self.names.append(data.name)
                self.data.append(data.to_numpy().reshape(-1, 1))
        else:  # numpy array
            self.design_info.append(None)
            self.indexes.append(np.arange(data.shape[0]))
            self.columns.append([f"{varname}{i}" for i in range(data.shape[1])] 
                                if data.ndim==2 else [f"{varname}0"])
            self.data.append(data)
    
    def __getitem__(self, index):
        """
        Get a tuple of the indexed data.
        
        Parameters
        ----------
        index : int or slice
            Index or slice for selecting the data.
        
        Returns
        -------
        tuple
            A tuple of the indexed data.
        """
        indexed_data = tuple(x[index] for x in self.data)
        if self.weights is not None:
            indexed_weights = self.weights[index]
            return (*indexed_data, indexed_weights)
        return (*indexed_data,)
    
    def __iter__(self):
        """
        Return an iterator for the data.
        
        Returns
        -------
        iterator
            Iterator for the data.
        """

        if self.weights is not None:
            return iter((*self.data, self.weights))
        return iter(self.data)
    
    def __len__(self):
        """
        Get the length of the data.
    
        Returns
        -------
        int
            Length of the data.
        """
        l = len(self.data) if self.weights is None else len(self.data)+1
        return l

    def __repr__(self):
        """
        Return a string representation of the ModelData object.
        
        Returns
        -------
        str
            String representation of the ModelData object.
        """
        s = [str(d.shape) for d in self.data if d is not None]
        if self.weights is not None:
            s.append(str(self.weights.shape))
        return f"ModelData({', '.join(s)})"
    
    @classmethod
    def from_formulas(cls, data, main_formula, *formula_args, **formula_kwargs):
        """
        Create a ModelData object from Patsy formulas.
    
        Parameters
        ----------
        data : pd.DataFrame
            DataFrame containing the data to be used in the formulas.
        main_formula : str
            A Patsy formula for the main model.
        *formula_args : str
            Additional Patsy formulas as positional arguments.
        **formula_kwargs : str
            Additional Patsy formulas as keyword arguments.
    
        Returns
        -------
        ModelData
            A ModelData object with the data from the formulas.
        """
        formulas = list(formula_args) + list(formula_kwargs.values())
        y, X_main = patsy.dmatrices(main_formula, data=data, return_type='dataframe')
        
        model_matrices = [X_main]
        for formula in formulas:
            lhs, X = design_matrices(formula, data)
            model_matrices.append(X)
        model_matrices.append(y)
        return cls(*model_matrices)
    
    def add_weights(self, weights=None):
        """
        Add weights to the ModelData object.

        Parameters
        ----------
        weights : array_like, optional
            An array of weights for the data, default is None.
        """
        self.weights = np.ones(len(self.data[-1])) if weights is None else weights
        
    def flatten_y(self):
        """
        Flatten the output variable (y) in the data.
        """
        self.data[-1] = self.data[-1].flatten()
        
    def drop_missing(self):
        """
        Drop rows containing nan values from the data arrays.
        """
        # Find rows with nan values
        nan_rows = np.any([np.isnan(arr).any(axis=1) for arr in self.data], axis=0)
    
        # Remove rows with nan values from data arrays
        for i, arr in enumerate(self.data):
            self.data[i] = arr[~nan_rows]
    
        # Update indexes
        for i, idx in enumerate(self.indexes):
            self.indexes[i] = idx[~nan_rows]
    
        # Update weights if they are present
        if self.weights is not None:
            self.weights = self.weights[~nan_rows]

        

class OrdinalModelData(ModelData):
    def __init__(self, *args, weights=None):
        super().__init__(*args, weights=weights)
        self._regression_data = self.data
        self.compute_ordinal_matrices()
        
    @classmethod
    def from_formulas(cls, data, main_formula, *formula_args, **formula_kwargs):
        main_formula = main_formula.replace("~", "~ 1 +")  # Add intercept to the main formula
        return super().from_formulas(data, main_formula, *formula_args, **formula_kwargs)

    def drop_intercept(self):
        for i, d_info in enumerate(self.design_info):
            if d_info is not None and "Intercept" in d_info.column_names:
                intercept_idx = d_info.column_names.index("Intercept")
                self.data[i] = np.delete(self.data[i], intercept_idx, axis=1)
                self.columns[i] = self.columns[i].delete(intercept_idx)
                d_info.column_names.remove("Intercept")
                d_info.column_name_indexes.pop("Intercept")

    def compute_ordinal_matrices(self):
        self.drop_intercept()
        self._regression_data[1] = self._regression_data[1].flatten()
        if len(self._regression_data)==2:
            X, y, = self._regression_data
        else:
            X, y, _ = self._regression_data
        unique, indices, inverse, counts = np.unique(y, return_index=True, return_inverse=True, return_counts=True, equal_nan=True)
        n_unique = len(unique)
        n_obs = len(y)
        row_ind, col_ind = np.arange(n_obs), inverse
        Y = sp.sparse.csc_matrix((np.ones_like(y), (row_ind, col_ind)), shape=(n_obs, n_unique))
        A1, A2 = Y[:, :-1], Y[:, 1:]
        o1, o2 = Y[:, -1].A.flatten() * 30e1, Y[:, 0].A.flatten() * -10e5
        o1ix = o1 != 0
        B1, B2 = np.block([A1.A, -X]), np.block([A2.A, -X])
        self.data = [B1, B2, o1, o2, o1ix]
        self.counts = counts
        self.n_cats = self.q = n_unique - 1
        self.tau_init = self._get_initial_thresholds(counts)
        self.unique = unique
        self.A1, self.A2 = A1, A2
        self.o1, self.o2 = o1, o2
        self.o1ix = o1ix
    
    @staticmethod
    def _get_initial_thresholds(counts):
        tau_init = sp.special.ndtri(counts.cumsum()[:-1]/np.sum(counts))
        return tau_init

    def __getitem__(self, index):
        indexed_data = tuple(x[index] for x in self.data)
        if self.weights is not None:
            indexed_weights = self.weights[index]
            return (*indexed_data, indexed_weights)
        return (*indexed_data,)

    def __iter__(self):
        if self.weights is not None:
            return iter((*self.data, self.weights))
        return iter(self.data)

    def __len__(self):
        l = len(self.data) if self.weights is None else len(self.data) + 1
        return l
    
    def __repr__(self):
        s = [str(d.shape) for d in self.data if d is not None]
        if self.weights is not None:
            s.append(str(self.weights.shape))
        return f"OrdinalModelData({', '.join(s)})"
    


class RegressionMixin(object):
    """
    A mixin class for regression models.
    
    Attributes
    ----------
    model_data : ModelData
        Contains the processed data for the model.
    """

    def __init__(self, formula=None, data=None, X=None, y=None, weights=None,
                 data_class=None, *args, **kwargs):
        """
        Initialize the RegressionMixin instance.
        
        Parameters
        ----------
        formula : str, optional
            A patsy formula specifying the model.
        data : DataFrame, optional
            A DataFrame containing the data.
        X : array_like, optional
            An array of independent variables.
        y : array_like, optional
            An array of dependent variables.
        weights : array_like, optional
            An array of weights to be applied in the model.
        *args : tuple
            Additional positional arguments.
        **kwargs : dict
            Additional keyword arguments.
        """
        self.model_data = self._process_data(formula, data, X, y, data_class=data_class,
                                             *args, **kwargs)
        self.model_data.add_weights(weights)
        
    @staticmethod
    def _process_data(formula=None, data=None, X=None, y=None, data_class=None,
                      default_varname='x', *args, **kwargs):
        """
        Process input data for the regression model.
    
        Parameters
        ----------
        formula : str, optional
            A patsy formula specifying the model.
        data : DataFrame, optional
            A DataFrame containing the data.
        X : array_like, optional
            An array of independent variables.
        y : array_like, optional
            An array of dependent variables.
        default_varname : str, optional
            The default variable name if none is provided.
        *args : tuple
            Additional positional arguments.
        **kwargs : dict
            Additional keyword arguments.
    
        Returns
        -------
        ModelData
            The processed data for the model.
        """
        data_class = ModelData if data_class is None else data_class
        if formula is not None and data is not None:
            model_data = data_class.from_formulas(data, formula, *args, **kwargs)
            y, X = patsy.dmatrices(formula, data=data, return_type='dataframe')
        elif X is not None and y is not None:
            model_data = data_class(*((X,)+args+(y,)))
        if np.ndim(model_data.data[-1]) == 2:
            if model_data.data[-1].shape[1] == 1:
                model_data.flatten_y()
        return model_data

                           
    @staticmethod
    def sandwich_cov(grad_weight, X, leverage=None, kind="HC0"):
        """
        Calculate sandwich covariance matrix for robust standard errors.
        
        Parameters
        ----------
        grad_weight : array_like
            An array of gradient weights.
        X : array_like
            An array of independent variables.
        leverage : array_like, optional
            An array of leverage values.
        kind : str, optional
            The type of robust covariance matrix to compute (default is "HC0").
        
        Returns
        -------
        array_like
            The sandwich covariance matrix.
        """
        w, h = grad_weight, leverage
        n, p = X.shape
        w = w ** 2
        if kind == "HC0":
            omega = w
        elif kind == "HC1":
            omega = w * n / (n - p)
        elif kind == "HC2":
            omega = w / (1.0 - h)
        elif kind == "HC3":
            omega = w / (1.0 - h)**2
        elif kind == "HC4":
            omega = w / (1.0 - h)**np.minimum(4.0, h / np.mean(h))
        B = np.dot((X * omega.reshape(-1, 1)).T, X)
        return B

    @staticmethod
    def _compute_leverage_cholesky(WX=None, Linv=None):
        """
        Compute the leverage values using the Cholesky decomposition method.
    
        Parameters
        ----------
        WX : array_like, optional
            Weighted independent variables.
        Linv : array_like, optional
            Inverse of the lower triangular Cholesky factor L.
    
        Returns
        -------
        array_like
            The leverage values.
        """
        if Linv is None:
            G = np.dot(WX.T, WX)
            L = np.linalg.cholesky(G)
            Linv, _ = scipy.linalg.lapack.dtrtri(L, lower=1)
        Q = Linv.dot(WX.T)
        h = np.sum(Q**2, axis=0)
        return h

    @staticmethod
    def _compute_leverage_qr(WX=None, Q=None):
        """
        Compute the leverage of the data points using QR decomposition.
        
        Parameters
        ----------
        WX : array_like, optional
            The weighted design matrix.
        Q : array_like, optional
            The Q matrix resulting from the QR decomposition.
        
        Returns
        -------
        h : array_like
            Leverage values for each data point.
        """
        if Q is None:
            Q, R = np.linalg.qr(WX)
        h = np.sum(Q**2, axis=1)
        return h

    @staticmethod
    def _rsquared(y, yhat):
        """
        Compute the coefficient of determination (R-squared) for the regression model.
    
        Parameters
        ----------
        y : array_like
            The observed response variable.
        yhat : array_like
            The predicted response variable.
    
        Returns
        -------
        r2 : float
            The R-squared value.
        """
        rh = y - yhat
        rb = y - np.mean(y)
        r2 = 1.0 - np.sum(rh**2) / np.sum(rb**2)
        return r2
    
    def make_termwise_constraint_matrices(self, scale=True):
        ii = np.arange(self.n_params)
        constraint_matrices = {}
        names = [x for x in self.x_design_info.term_names if x!="Intercept"]
        for name in names:
            term_slice = self.x_design_info.term_name_slices[name]
            C = np.zeros((len(ii[term_slice]), self.n_params))
            C[np.arange(len(ii[term_slice])), ii[term_slice]] = 1.0
            constraint_matrices[name] = C.copy()
        self.constraint_matrices = constraint_matrices
        
    def ll_tests(self, scale=True):
        names = [x for x in self.x_design_info.term_names if x!="Intercept"]
        wald_tests = {}
        score_tests = {}
        for name in names:
            term_slice = self.x_design_info.term_name_slices[name]
            params_unconstrained = self.params
            zero_indices = np.arange(self.n_params)
            zero_indices = zero_indices[:-1] if scale else zero_indices
            zero_indices = zero_indices[term_slice]
            constraint = optimizer_utils.ZeroConstraint(self.n_params, zero_indices)
            
            constraint = sp.optimize.NonlinearConstraint(constraint.func, 
                                                jac=constraint.grad, 
                                                lb=np.zeros(len(zero_indices)), 
                                                ub=np.zeros(len(zero_indices)))
            
            constraints = [constraint]
            
            default_minimize_kws = dict(fun=self.full_loglike, jac=self.gradient, 
                                        hess=self.hessian,
                                        method="trust-constr",
                                        constraints=constraints)
            minimize_kws = func_utils.handle_default_kws(None, default_minimize_kws)
            x0 = params_unconstrained.copy()
            x0[zero_indices] = 0
            opt = sp.optimize.minimize(x0=x0, **minimize_kws)
            params_constrained = opt.x
            constraint_derivative = constraint.jac(params_constrained)
            hess_unconstrained_inv = np.linalg.inv(self.hessian(params_unconstrained))
            hess_constrained_inv = np.linalg.inv(self.hessian(params_constrained))
            grad_i_unconstrained = self.gradient_i(params_unconstrained)
            grad_i_constrained = self.gradient_i(params_constrained)
            grad_constrained = np.sum(grad_i_constrained, axis=0)
            grad_unconstrained_cov = np.dot(grad_i_unconstrained.T, grad_i_unconstrained)
            grad_constrained_cov =  np.dot(grad_i_constrained.T, grad_i_constrained)
            
            wald_tests[name] = self._wald_test(params_unconstrained, 
                                               params_constrained,
                                               constraint_derivative,
                                               hess_unconstrained_inv,
                                               grad_unconstrained_cov,
                                               return_dataframe=True)
            
            score_tests[name] =  self._score_test(grad_constrained, 
                                            constraint_derivative,
                                            hess_constrained_inv, 
                                            grad_constrained_cov,
                                            return_dataframe=True)
        self.wald_tests = pd.concat(wald_tests)
        self.score_tests = pd.concat(score_tests)
        
        
    
    
    
    
    
        
