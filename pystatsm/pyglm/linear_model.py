#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 19:14:22 2023

@author: lukepinkel
"""
import tqdm                                                                    # analysis:ignore
import patsy
import numba
import numpy as np
import scipy as sp
import scipy.stats
import pandas as pd
from scipy.special import loggamma, digamma, polygamma                         # analysis:ignore
from functools import cached_property                                          # analysis:ignore
from patsy import PatsyError
from abc import ABCMeta, abstractmethod
from ..utilities import output
from ..utilities.linalg_operations import wdiag_outer_prod, wls_qr, nwls       # analysis:ignore
from ..utilities.func_utils import symmetric_conf_int, handle_default_kws
from ..utilities.data_utils import _check_shape, _check_type                   # analysis:ignore
from ..utilities.formula import design_matrices                  # analysis:ignore
from .links import (LogitLink, ProbitLink, Link, LogLink, ReciprocalLink,      # analysis:ignore
                    PowerLink)                                                 # analysis:ignore
from .families import (Binomial, ExponentialFamily, Gamma, Gaussian,           # analysis:ignore
                       IdentityLink, InverseGaussian, NegativeBinomial,        # analysis:ignore
                       Poisson)                                                # analysis:ignore

LN2PI = np.log(2 * np.pi)

class LikelihoodModel(metaclass=ABCMeta):
    """
    Abstract base class for likelihood-based models.

    This class serves as a blueprint for building likelihood-based models. 
    It provides an interface  for implementing log-likelihood, gradient, and
    Hessian functions, as well as methods for fitting, information criteria, 
    pseudo-R-squared, parameter inference, and hypothesis testing.

    Subclasses must implement the following methods:
        _loglike
        _gradient
        _hessian
        _fit

    Additional methods that can be used or overridden in subclasses are also provided.
    """

    @staticmethod
    @abstractmethod
    def _loglike(params, data):
        """
        Compute the log-likelihood of the model.

        Parameters
        ----------
        params : array_like
            Model parameters.
        data : any
            Data used to compute the log-likelihood.

        Returns
        -------
        float
            Log-likelihood value.
        """
        pass

    @staticmethod
    @abstractmethod
    def _gradient(params, data):
        """
        Compute the gradient of the log-likelihood with respect to the parameters.

        Parameters
        ----------
        params : array_like
            Model parameters.
        data : any
            Data used to compute the gradient.

        Returns
        -------
        ndarray
            Gradient of the log-likelihood.
        """
        pass

    @staticmethod
    @abstractmethod
    def _hessian(params, data):
        """
        Compute the Hessian of the log-likelihood with respect to the parameters.

        Parameters
        ----------
        params : array_like
            Model parameters.
        data : any
            Data used to compute the Hessian.

        Returns
        -------
        ndarray
            Hessian of the log-likelihood.
        """
        pass

    @staticmethod
    @abstractmethod
    def _fit(params, data):
        """
        Fit the model to the given data.

        Parameters
        ----------
        params : array_like
            Initial model parameters.
        data : any
            Data used to fit the model.

        Returns
        -------
        dict
            Fitted model parameters and additional information.
        """
        pass

    @staticmethod
    def _get_information(ll, n_params, n_obs):
        """
        Computes various information criteria for model comparison and selection.

        This method calculates the Akaike Information Criterion (AIC), the
        corrected Akaike Information Criterion (AICc), the Bayesian Information 
        Criterion (BIC), and the Consistent Akaike Information Criterion (CAIC) 
        based on the provided log-likelihood,  number of parameters, and number
        of observations.

        Parameters
        ----------
        ll : float
            The log-likelihood of the model.
        n_params : int
            The number of parameters in the model.
        n_obs : int
            The number of observations in the dataset.

        Returns
        -------
        tuple
            A tuple containing the computed values of AIC, AICc, BIC, and CAIC,
            respectively.
        """
        logn = np.log(n_obs)
        tll = 2 * ll
        aic = tll + 2 * n_params
        aicc = tll + 2 * n_params * n_obs / (n_obs - n_params - 1)
        bic = tll + n_params * logn
        caic = tll + n_params * (logn + 1)
        return aic, aicc, bic, caic

    @staticmethod
    def _get_pseudo_rsquared(ll_model, ll_null, n_params, n_obs):
        """
        Compute pseudo R-squared values for the model.
        
        This method computes the Cox-Snell, Nagelkerke, McFadden, and 
        McFadden-Bartlett pseudo R-squared values, as well as the likelihood 
        ratio test statistic.
    
        Parameters
        ----------
        ll_model : float
            The log-likelihood of the fitted model.
        ll_null : float
            The log-likelihood of the null model.
        n_params : int
            The number of parameters in the model.
        n_obs : int
            The number of observations in the dataset.
    
        Returns
        -------
        tuple
            A tuple containing the computed values of Cox-Snell, Nagelkerke, 
            McFadden, McFadden-Bartlett pseudo R-squared, and likelihood ratio 
            test statistic, respectively.
        """
        r2_cs = 1-np.exp(2.0/n_obs * (ll_model - ll_null))
        r2_nk = r2_cs / (1-np.exp(2.0 / n_obs * -ll_null))
        r2_mc = 1.0 - ll_model / ll_null
        r2_mb = 1.0 - (ll_model - n_params) / ll_null
        llr = 2.0 * (ll_null - ll_model)
        return r2_cs, r2_nk, r2_mc, r2_mb, llr

    @staticmethod
    def _parameter_inference(params, params_se, degfree, param_labels):
        """
        Create a parameter inference table.
    
        Parameters
        ----------
        params : array_like
            Model parameters.
        params_se : array_like
            Standard errors of the parameters.
        degfree : int
            Degrees of freedom for the parameter inference.
        param_labels : array_like
            Labels of the parameters.
    
        Returns
        -------
        DataFrame
            A pandas DataFrame containing the parameter inference table.
        """
        res = output.get_param_table(params, params_se, degfree, param_labels)
        return res

    @staticmethod
    def _make_constraints(fixed_indices, fixed_values):
        """
        Create constraints for optimization.
        
        Parameters
        ----------
        fixed_indices : array_like
            Indices of the fixed parameters.
        fixed_values : array_like
            Values of the fixed parameters.
        
        Returns
        -------
        list
            A list of constraints for optimization.
        """
        constraints = []
        if np.asarray(fixed_indices).dtype==bool:
            fixed_indices = np.where(fixed_indices)
        for i, xi in list(zip(fixed_indices, fixed_values)):
            constraints.append({"type":"eq", "fun":lambda x: x[i] - xi})
        return constraints

    @staticmethod
    def _profile_loglike_constrained(par, ind, params, loglike, grad, hess,
                                     minimize_kws=None, return_info=False):
        """
        Calculate the profile log-likelihood for a model with constrained
        parameters by directly  incorporating constraints during the 
        optimization process.
        
        Parameters
        ----------
        par : array_like
            Parameter values to be constrained.
        ind : array_like
            Indices of the constrained parameters.
        params : array_like
            Initial parameter estimates for the optimization.
        loglike : callable
            Function to compute the log-likelihood.
        grad : callable
            Function to compute the gradient of the log-likelihood.
        hess : callable
            Function to compute the Hessian of the log-likelihood.
        minimize_kws : dict, optional
            Additional keyword arguments to be passed to the
            scipy.optimize.minimize function.
        return_info : bool, optional
            If True, return the optimization result object instead of the 
            log-likelihood value. Default is False.
        
        Returns
        -------
        float or OptimizeResult
            If return_info is False, return the profile log-likelihood for 
            the constrained model.  Otherwise, return the optimization result
            object.
        """
        par, ind = np.atleast_1d(par), np.atleast_1d(ind)
        constraints = []
        for i, xi in list(zip(ind, par)):
            constraints.append({"type":"eq", "fun":lambda x: x[i] - xi})
        default_minimize_kws = dict(fun=loglike, jac=grad, hess=hess,
                                    method="trust-constr",
                                    constraints=constraints)
        minimize_kws = handle_default_kws(minimize_kws, default_minimize_kws)
        x0 = params.copy()
        x0[ind] = par
        opt = sp.optimize.minimize(x0=x0, **minimize_kws)
        if return_info:
            res = opt
        else:
            res = loglike(opt.x)
        return res

    @staticmethod
    def _profile_loglike_restricted(par, ind, params, loglike, grad, hess,
                                    minimize_kws=None, return_info=False):
        """
        Calculate the profile log-likelihood for a model with restricted 
        parameters by optimizing  over the free (unconstrained) parameters 
        and substituting the fixed (restricted) parameter 
        values.
        
        Parameters
        ----------
        par : array_like
            Parameter values to be restricted.
        ind : array_like
            Indices of the restricted parameters.
        params : array_like
            Initial parameter estimates for the optimization.
        loglike : callable
            Function to compute the log-likelihood.
        grad : callable
            Function to compute the gradient of the log-likelihood.
        hess : callable
            Function to compute the Hessian of the log-likelihood.
        minimize_kws : dict, optional
            Additional keyword arguments to be passed to the 
            scipy.optimize.minimize function.
        return_info : bool, optional
            If True, return the optimization result object instead of the 
            log-likelihood value.  Default is False.
        
        Returns
        -------
        float or OptimizeResult
            If return_info is False, return the profile log-likelihood for 
            the restricted model.  Otherwise, return the optimization result
            object.
        """
        par, ind = np.atleast_1d(par), np.atleast_1d(ind)
        free_ind = np.setdiff1d(np.arange(len(params)), ind)
        full_params = params.copy()
        full_params[ind] = par

        def restricted_func(free_params):
            full_params[free_ind] = free_params
            full_params[ind] = par
            fval = loglike(full_params)
            return fval

        def restricted_grad(free_params):
            full_params[free_ind] = free_params
            full_params[ind] = par
            g = grad(full_params)[free_ind]
            return g

        def restricted_hess(free_params):
            full_params[free_ind] = free_params
            full_params[ind] = par
            H = hess(full_params)[free_ind][:, free_ind]
            return H


        default_minimize_kws = dict(fun=restricted_func,
                                    jac=restricted_grad,
                                    hess=restricted_hess,
                                    method="trust-constr")
        minimize_kws = handle_default_kws(minimize_kws, default_minimize_kws)
        x0 = params.copy()[free_ind]
        opt = sp.optimize.minimize(x0=x0, **minimize_kws)
        if return_info:
            res = opt
        else:
            res = restricted_func(opt.x)
        return res


    @staticmethod
    def _solve_interval(profile_loglike, par, par_se, lli, method="root",
                       return_info=False):
        """
        Compute the confidence interval for a parameter using the profile 
        log-likelihood.
        
        Parameters
        ----------
        profile_loglike : callable
            Function to compute the profile log-likelihood.
        par : float
            Point estimate of the parameter.
        par_se : float
            Standard error of the parameter.
        lli : float
            Log-likelihood value used to compute the interval.
        method : str, optional
            Method to solve the interval, either 'root' or 'lstq'. Default 
            is 'root'.
        return_info : bool, optional
            If True, return additional optimization information. 
            Default is False.
        
        Returns
        -------
        tuple
            The lower and upper bounds of the confidence interval.
        """
        if method == "lstq":
            left=sp.optimize.minimize(lambda x:(profile_loglike(x)-lli)**2,
                                        par-2 * par_se)
            right=sp.optimize.minimize(lambda x:(profile_loglike(x)-lli)**2,
                                         par+2 * par_se)
            if return_info:
                left, right = left, right
            else:
                left, right = left.x, right.x
        else:
            for kl in range(1, 20):
                if profile_loglike(par - kl * par_se) > lli:
                    break
            for kr in range(1, 20):
                if profile_loglike(par + kr * par_se) > lli:
                    break
            left_bracket = (par - kl * par_se, par)
            right_bracket = (par, par + kr * par_se)
            left = sp.optimize.root_scalar(lambda x:profile_loglike(x)-lli,
                                           bracket=left_bracket)
            right = sp.optimize.root_scalar(lambda x:profile_loglike(x)-lli,
                                            bracket=right_bracket)
            if return_info:
                left, right = left, right
            else:
                left, right = left.root, right.root
        return left, right
    
    @staticmethod
    def _lr_test(ll_unconstrained, ll_constrained, constraint_dim, 
                 return_dataframe=True):
        """
        Perform the likelihood ratio test for nested models.
    
        Parameters
        ----------
        ll_unconstrained : float
            Log-likelihood value of the unconstrained model.
        ll_constrained : float
            Log-likelihood value of the constrained model.
        constraint_dim : int
            Dimension of the constraint, representing the difference in the
            number of parameters between 
            the unconstrained and constrained models.
        return_dataframe : bool, optional
            If True, the result is returned as a pandas DataFrame. 
            Default is True.
    
        Returns
        -------
        DataFrame or dict
            A pandas DataFrame or dictionary containing the likelihood ratio 
            test statistic, degrees of freedom, and p-value.
        """
        ll_full = ll_unconstrained
        ll_null = ll_constrained
        df = constraint_dim
        lr_stat = 2.0 * (ll_full - ll_null)
        p_value = sp.stats.chi2(df=df).sf(lr_stat)
        res = {"Stat":lr_stat, "df":df ,"P Value":p_value}
        if return_dataframe:  
            res = pd.DataFrame(res, index=["LR Test"])
        return res
    
    @staticmethod
    def _wald_test(params_unconstrained, 
                   params_constrained,
                   constraint_derivative,
                   hess_unconstrained_inv,
                   grad_unconstrained_cov,
                   return_dataframe=True):
        """
        Perform the Wald test for nested models.
    
        Parameters
        ----------
        params_unconstrained : array_like
            Unconstrained model parameters.
        params_constrained : array_like
            Constrained model parameters.
        constraint_derivative : array_like
            Derivative of the constraint with respect to the parameters.
        hess_unconstrained_inv : array_like
            Inverse of the Hessian matrix for the unconstrained model.
        grad_unconstrained_cov : array_like
            Covariance matrix of the gradient for the unconstrained model.
        return_dataframe : bool, optional
            If True, the result is returned as a pandas DataFrame. 
            Default is True.
    
        Returns
        -------
        DataFrame or dict
            A pandas DataFrame or dictionary containing the Wald test statistic,
            degrees of freedom, 
            and p-value.
        """
        theta, theta0 = params_unconstrained, params_constrained
        A, B = hess_unconstrained_inv, grad_unconstrained_cov
        C = constraint_derivative
        V = np.dot(A, np.dot(B, A))
        M = np.linalg.inv(C.dot(V).dot(C.T))
        a = np.dot(C, theta - theta0)
        wald_stat = np.dot(a, M.dot(a))
        df = constraint_derivative.shape[0]
        p_value = sp.stats.chi2(df=df).sf(wald_stat)
        res = {"Stat":wald_stat, "df":df ,"P Value":p_value}
        if return_dataframe:  
            res = pd.DataFrame(res, index=["Wald Test"])
        return res
    
    @staticmethod
    def _score_test(grad_constrained, 
                    constraint_derivative,
                    hess_constrained_inv, 
                    grad_constrained_cov,
                    return_dataframe=True):
        """
         Perform the score test for nested models.
        
         Parameters
         ----------
         grad_constrained : array_like
             Gradient of the constrained model.
         constraint_derivative : array_like
             Derivative of the constraint with respect to the parameters.
         hess_constrained_inv : array_like
             Inverse of the Hessian matrix for the constrained model.
         grad_constrained_cov : array_like
             Covariance matrix of the gradient for the constrained model.
         return_dataframe : bool, optional
             If True, the result is returned as a pandas DataFrame. 
             Default is True.
        
         Returns
         -------
         DataFrame or dict
             A pandas DataFrame or dictionary containing the score test 
             statistic, degrees of freedom,  and p-value.
         """
        A, B = hess_constrained_inv, grad_constrained_cov
        C, D = constraint_derivative, grad_constrained
        V = np.dot(A, np.dot(B, A))
        M = np.linalg.inv(C.dot(V).dot(C.T))
        a = np.dot(C, np.dot(A, D))
        score_stat = np.dot(a, M.dot(a))
        df = constraint_derivative.shape[0]
        p_value = sp.stats.chi2(df=df).sf(score_stat)
        res = {"Stat":score_stat, "df":df ,"P Value":p_value}
        if return_dataframe:  
            res = pd.DataFrame(res, index=["Score Test"])
        return res

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
        

class RegressionMixin(object):
    """
    A mixin class for regression models.
    
    Attributes
    ----------
    model_data : ModelData
        Contains the processed data for the model.
    """

    def __init__(self, formula=None, data=None, X=None, y=None, weights=None,
                 *args, **kwargs):
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
        self.model_data = self._process_data(formula, data, X, y, *args, **kwargs)
        self.model_data.add_weights(weights)
        
    @staticmethod
    def _process_data(formula=None, data=None, X=None, y=None,
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
        if formula is not None and data is not None:
            model_data = ModelData.from_formulas(data, formula, *args, **kwargs)
            y, X = patsy.dmatrices(formula, data=data, return_type='dataframe')
        elif X is not None and y is not None:
            model_data = ModelData(*((X,)+args+(y,)))
        
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

        
class LinearModel(RegressionMixin, LikelihoodModel):
    """
    A linear regression model class that inherits from RegressionMixin and LikelihoodModel.

    Parameters
    ----------
    formula : str, optional
        A formula representing the relationship between the response and predictor variables.
    data : DataFrame, optional
        A pandas DataFrame containing the data to be used for the linear regression model.
    X : array-like, optional
        The predictor variables in the linear regression model.
    y : array-like, optional
        The response variable in the linear regression model.
    weights : array-like, optional
        Weights to be applied to the data during the regression fitting process.
    *args : tuple
        Additional positional arguments.
    **kwargs : dict
        Additional keyword arguments.
    """
    def __init__(self, formula=None, data=None, X=None, y=None, weights=None,
                 *args, **kwargs):
        super().__init__(formula=formula, data=data, X=X, y=y, weights=weights, 
                         *args, **kwargs)
        self.xinds, self.yinds = self.model_data.indexes
        self.xcols, self.ycols = self.model_data.columns
        self.X, self.y, self.weights = self.model_data
        self.n = self.n_obs = self.X.shape[0]
        self.p = self.n_var = self.X.shape[1]
        self.x_design_info, self.y_design_info = self.model_data.design_info
        self.formula = formula
        self.data = data

    @staticmethod
    def _loglike(params, data, reml=True):
        """
        Compute the log-likelihood of the linear model.
    
        Parameters
        ----------
        params : array_like
            Model parameters.
        data : tuple
            Tuple containing the predictor variables (X) and response variable (y).
        reml : bool, optional
            Whether to use Restricted Maximum Likelihood (REML) estimation. Default is True.
    
        Returns
        -------
        float
            Log-likelihood value.
        """
        beta, logvar = params[:-1], params[-1]
        tau = np.exp(-logvar)
        X, y = data
        n, p = X.shape
        n = n - p if reml else n
        const = (-n / 2) * (LN2PI + logvar)
        yhat = X.dot(beta)
        resid = y - yhat
        sse = np.dot(resid.T, resid)
        ll = const - (tau * sse) / 2
        return -ll

    def loglike(self, params, reml=True):
        """
        Compute the log-likelihood of the linear model.
        
        Parameters
        ----------
        params : array_like
            Model parameters.
        reml : bool, optional
            Whether to use Restricted Maximum Likelihood (REML) estimation. Default is True.
        
        Returns
        -------
        float
            Log-likelihood value.
        """
        return self._loglike(params, (self.X, self.y), reml=reml)

    @staticmethod
    def _gradient(params, data, reml=True):
        """
        Compute the gradient of the log-likelihood function with respect to the parameters.
    
        Parameters
        ----------
        params : array_like
            Model parameters.
        data : tuple
            Tuple containing the predictor variables (X) and response variable (y).
        reml : bool, optional
            Whether to use Restricted Maximum Likelihood (REML) estimation. Default is True.
    
        Returns
        -------
        ndarray
            Gradient of the log-likelihood function.
        """
        beta, logvar = params[:-1], params[-1]
        tau = np.exp(-logvar)
        X, y = data
        n, p = X.shape
        n = n - p if reml else n
        grad = np.zeros_like(params)
        yhat = X.dot(beta)
        resid = y - yhat
        grad[:-1] = tau * np.dot(resid.T, X)
        grad[-1] = (-n / 2) + (tau * np.dot(resid.T, resid)) / 2.0
        return -grad

    def gradient(self, params, reml=True):
        """
        Compute the gradient of the log-likelihood function with respect to the parameters.
    
        Parameters
        ----------
        params : array_like
            Model parameters.
        reml : bool, optional
            Whether to use Restricted Maximum Likelihood (REML) estimation. Default is True.
    
        Returns
        -------
        ndarray
            Gradient of the log-likelihood function.
        """
        return self._gradient(params, (self.X, self.y), reml=reml)

    @staticmethod
    def _hessian(params, data, reml=True):
        """
        Compute the Hessian of the log-likelihood function with respect to the parameters.
    
        Parameters
        ----------
        params : array_like
            Model parameters.
        data : tuple
            Tuple containing the predictor variables (X) and response variable (y).
        reml : bool, optional
            Whether to use Restricted Maximum Likelihood (REML) estimation. Default is True.
    
        Returns
        -------
        ndarray
            Hessian of the log-likelihood function.
        """
        beta, logvar = params[:-1], params[-1]
        tau = np.exp(-logvar)
        X, y = data
        hess = np.zeros((params.shape[0],)*2)
        yhat = X.dot(beta)
        resid = y - yhat
        hess[:-1, :-1] = -tau * np.dot(X.T, X)
        hess[:-1, -1] = -tau * np.dot(X.T, resid)
        hess[-1, :-1] = -tau * np.dot(X.T, resid)
        hess[-1, -1] = -(tau * np.dot(resid.T, resid)) / 2
        return -hess

    def hessian(self, params, reml=True):
        """
        Compute the Hessian of the log-likelihood function with respect to the parameters.
        
        Parameters
        ----------
        params : array_like
            Model parameters.
        reml : bool, optional
            Whether to use Restricted Maximum Likelihood (REML) estimation. Default is True.
        
        Returns
        -------
        ndarray
            Hessian of the log-likelihood function.
        """
        return self._hessian(params, (self.X, self.y), reml=reml)

    @staticmethod
    def _fit1(params, data, reml=True):
        """
        Fit the linear model using Cholesky decomposition (Method 1).
        
        Parameters
        ----------
        params : array_like
            Model parameters.
        data : tuple
            Tuple containing the predictor variables (X) and response variable (y).
        reml : bool, optional
            Whether to use Restricted Maximum Likelihood (REML) estimation. Default is True.
        
        Returns
        -------
        tuple
            Tuple containing the matrices and parameters needed for the model.
        """
        X, y = data
        n, p = X.shape
        G = X.T.dot(X)
        c = X.T.dot(y)
        L = np.linalg.cholesky(G)
        w = sp.linalg.solve_triangular(L, c, lower=True)
        sse = y.T.dot(y) - w.T.dot(w)
        beta = sp.linalg.solve_triangular(L.T, w, lower=False)
        Linv = np.linalg.inv(L)
        Ginv = np.dot(Linv.T, Linv)
        n = n - p if reml else n
        params = np.zeros(len(beta)+1)
        params[:-1], params[-1] = beta, np.log(sse / n)
        return G, Ginv, L, Linv, sse, beta, params

    @staticmethod
    def _fit2(params, data, reml=True):
        """
        Fit the linear model using Cholesky decomposition (Method 2).
        
        Parameters
        ----------
        params : array_like
            Model parameters.
        data : tuple
            Tuple containing the predictor variables (X) and response variable (y).
        reml : bool, optional
            Whether to use Restricted Maximum Likelihood (REML) estimation. Default is True.
        
        Returns
        -------
        tuple
            Tuple containing the matrices and parameters needed for the model.
        """
        X, y = data
        n, p = X.shape
        G = np.dot(X.T, X)
        c = X.T.dot(y)
        L = np.linalg.cholesky(G)
        Linv, _ = scipy.linalg.lapack.dtrtri(L, lower=1)
        w = Linv.dot(c)
        sse = y.T.dot(y) - w.T.dot(w)
        beta = np.dot(Linv.T, w)
        Ginv = np.dot(Linv.T, Linv)
        n = n - p if reml else n
        params = np.zeros(len(beta)+1)
        params[:-1], params[-1] = beta, np.log(sse / n)
        return G, Ginv, L, Linv, sse, beta, params

    def _fit(self, reml=True):
        """
        Fit the linear model using the best Cholesky decomposition method available.
        
        Parameters
        ----------
        reml : bool, optional
            Whether to use Restricted Maximum Likelihood (REML) estimation. Default is True.
        
        Returns
        -------
        None
            The fitted model parameters are stored as attributes of the LinearModel object.
        """

        G, Ginv, L, Linv, sse, beta, params = self._fit2(
            None, (self.X, self.y), reml=reml)
        self.G, self.Ginv, self.L, self.Linv = G, Ginv, L, Linv
        self.sse = sse
        self.scale = np.exp(params[-1]/2)
        self.scale_ml = np.sqrt(sse / self.n)
        self.scale_unbiased = np.sqrt(sse / (self.n - self.p))
        self.beta = beta
        self.params = params
        self.params_hess = self.hessian(self.params)
        self.params_cov = np.linalg.inv(self.params_hess)
        self.params_se = np.diag(np.linalg.inv(self.hessian(self.params)))**0.5
        self.res = output.get_param_table(self.params, self.params_se,
                                          self.n-self.p,
                                          list(self.xcols)+["log_scale"],
                                          )
        self.params_labels = list(self.xcols)+["log_scale"]

    @staticmethod
    def _get_coef_constrained(params, sse, data, C, d=None, L=None, Linv=None):
        """
        Get the constrained coefficients of the linear model.
        
        Parameters
        ----------
        params : array_like
            Model parameters.
        sse : float
            Sum of squared errors.
        data : tuple
            Tuple containing the predictor variables (X) and response variable (y).
        C : ndarray
            Constraint matrix.
        d : array_like, optional
            Vector of constants for the constraints. Default is None (assumed to be a zero vector).
        L : ndarray, optional
            Lower-triangular Cholesky factor of the Gram matrix. Default is None.
        Linv : ndarray, optional
            Inverse of the lower-triangular Cholesky factor. Default is None.
        
        Returns
        -------
        tuple
            Tuple containing the constrained coefficients and the sum of squared errors for the constrained model.
        """
        if Linv is None:
            if L is None:
                X, y = data
                L = np.linalg.cholesky(X.T.dot(X))
            Linv, _ = scipy.linalg.lapack.dtrtri(L, lower=1)
        d = np.zeros(len(C)) if d is None else d
        G = Linv.dot(C.T)
        Q, R = np.linalg.qr(G)
        Cb = C.dot(params[:-1])
        w = sp.linalg.solve_triangular(R.T, Cb-d, lower=True)
        sse_constrained = sse + np.dot(w.T, w)
        beta_constrained = params[:-1] - Linv.T.dot(Q.dot(w))
        return beta_constrained, sse_constrained
        
    
    def _bootstrap(self, n_boot, verbose=True):
        """
        Perform a non-parametric bootstrap to estimate model coefficients and error variance.
        
        Parameters
        ----------
        n_boot : int
            Number of bootstrap samples to generate.
        verbose : bool, optional
            Whether to display a progress bar during bootstrapping. Default is True.
        
        Returns
        -------
        ndarray
            An array containing the bootstrapped model parameters.
        """
        pbar = tqdm.tqdm(total=n_boot, smoothing=0.001) if verbose else None
        params = np.zeros((n_boot, self.p+1))
        i = 0
        while i < n_boot:
            try:
                ix = np.random.choice(self.n, self.n)
                Xb, yb = self.X[ix],  self.y[ix]
                G, c = Xb.T.dot(Xb), Xb.T.dot(yb)
                L = np.linalg.cholesky(G)
                w = sp.linalg.solve_triangular(L, c, lower=True)
                params[i, :-1] = sp.linalg.solve_triangular(L.T, w, lower=False)
                params[i, -1] =  (yb.T.dot(yb) - w.T.dot(w)) 
                i += 1
                if verbose:
                    pbar.update(1)
            except np.linalg.LinAlgError:
                pass
        if verbose:
            pbar.close()
        return params
    
    @staticmethod
    @numba.jit(nopython=True,parallel=True)
    def bootstrap_chol(X, y, params, n_boot):
        """
        Perform a non-parametric bootstrap using Cholesky decomposition (JIT-compiled with Numba).
        
        Parameters
        ----------
        X : ndarray
            Predictor variables.
        y : ndarray
            Response variable.
        params : ndarray
            An array to store the bootstrapped model parameters.
        n_boot : int
            Number of bootstrap samples to generate.
        
        Returns
        -------
        ndarray
            An array containing the bootstrapped model parameters.
        """
        n = X.shape[0]
        for i in numba.prange(n_boot):
            ii = np.random.choice(n, n)
            Xboot = X[ii]
            yboot = y[ii]
            G = np.dot(Xboot.T, Xboot)
            c = np.dot(Xboot.T, yboot)
            L = np.linalg.cholesky(G)
            w = np.linalg.solve(L, c)#numba_dtrtrs(L, c, "L")
            params[i, :-1] = np.linalg.solve(L.T, w).T#numba_dtrtrs(L.T, w).T
            params[i, -1] = (np.dot(yboot.T, yboot) - np.dot(w.T, w))[0, 0]
        return params

    def _bootstrap_jitted(self, n_boot):
        """
        Perform a non-parametric bootstrap using the JIT-compiled Cholesky decomposition method.
    
        Parameters
        ----------
        n_boot : int
            Number of bootstrap samples to generate.
    
        Returns
        -------
        ndarray
            An array containing the bootstrapped model parameters.
        """
        params = np.zeros((n_boot, self.p+1))
        params = self.bootstrap_chol(self.X, self.y.reshape(-1, 1), params, n_boot)
        return params
    
    def _permutation_test(self, vars_of_interest, n_perms=5000,
                       verbose=True, rng=None):
        """
        Perform a permutation test on the variables of interest.
    
        Parameters
        ----------
        vars_of_interest : list
            List of indices of the predictor variables of interest.
        n_perms : int, optional
            Number of permutations to perform. Default is 5000.
        verbose : bool, optional
            Whether to show a progress bar. Default is True.
        rng : np.random.Generator, optional
            A random number generator instance.
    
        Returns
        -------
        tuple
            Tuple containing p-values adjusted for family-wise error rate (FWER) and 
            unadjusted p-values.
        """
        rng = np.random.default_rng() if rng is None else rng
        pbar = tqdm.tqdm(total=n_perms, smoothing=0.001) if verbose else None
        p_values = np.zeros(len(vars_of_interest))
        p_values_fwer = np.zeros(len(vars_of_interest))
        abst = np.abs(self.tvalues[vars_of_interest])
        
        n, p = self.n, self.p
        ixc = np.setdiff1d(np.arange(p), vars_of_interest)
        X, y = self.X, self.y
        G = np.dot(X.T, X)
        L = np.linalg.cholesky(G)
        Linv = np.linalg.inv(L)
        Ginv = np.diag(np.dot(Linv.T, Linv))
        Xc = X[:, ixc]
        
        Gc = G[ixc][:, ixc]
        c = np.dot(X[:, ixc].T, y)
        Lc = np.linalg.cholesky(Gc)
        w = np.linalg.solve(Lc, c)
        g = np.linalg.solve(Lc.T, w)
        u = Xc.dot(g)
        r = y - u
        for i in range(n_perms):
            z = u + r[rng.permutation(n)]
            c = X.T.dot(z)
            w = sp.linalg.solve_triangular(L, c, lower=True)
            s2 =  (y.T.dot(y) - w.T.dot(w)) / (n - p)
            beta = sp.linalg.solve_triangular(L.T, w, lower=False)
            beta_se = np.sqrt(s2 * Ginv)           
            abstp = np.abs(beta / beta_se)[vars_of_interest]
            p_values_fwer += (abstp.max()>abst) / n_perms
            p_values +=  (abstp>abst) / n_perms
            if verbose:
                pbar.update(1)
        if verbose:
            pbar.close()
        return p_values_fwer, p_values
    
    

class GLM(RegressionMixin, LikelihoodModel):
    """
    Generalized Linear Model (GLM) class for fitting regression models.

    Parameters
    ----------
    formula : str, optional
        A patsy formula specifying the model to be fitted.
    data : pandas.DataFrame, optional
        A DataFrame containing the data to be used for model fitting.
    X : ndarray, optional
        The predictor variables matrix.
    y : ndarray, optional
        The response variable.
    family : ExponentialFamily, optional
        The distribution family to use. Default is Gaussian.
    scale_estimator : str, optional
        The scale estimator to use. Default is "M".
    weights : ndarray, optional
        Optional array of weights to be used in the model fitting.
    *args
        Additional positional arguments.
    **kwargs
        Additional keyword arguments.

    Attributes
    ----------
    xinds, yinds : array-like
        Indexes for predictor and response variables.
    xcols, ycols : array-like
        Columns for predictor and response variables.
    X, y, weights : ndarray
        Predictor variables matrix, response variable, and weights.
    n, n_obs : int
        Number of observations.
    p, n_var : int
        Number of variables.
    x_design_info, y_design_info : DesignInfo
        Design information for predictor and response variables.
    formula : str
        Patsy formula specifying the model.
    f : ExponentialFamily
        The distribution family to use.
    param_labels : list
        Labels for the model parameters.
    beta_init, phi_init : ndarray
        Initial values for model parameters.
    params_init : ndarray
        Initial values for model parameters.
    scale_estimator : str
        Scale estimator method.
    """
    def __init__(self, formula=None, data=None, X=None, y=None,
                 family=Gaussian, scale_estimator="M", weights=None, *args,
                 **kwargs):
        super().__init__(formula=formula, data=data, X=X, y=y, weights=weights, 
                         *args, **kwargs)
        self.xinds, self.yinds = self.model_data.indexes
        self.xcols, self.ycols = self.model_data.columns
        self.X, self.y, self.weights = self.model_data
        self.n = self.n_obs = self.X.shape[0]
        self.p = self.n_var = self.X.shape[1]
        self.x_design_info, self.y_design_info = self.model_data.design_info
        self.formula = formula
        #self.data = data
        if isinstance(family, ExponentialFamily) is False:
            try:
                family = family()
            except TypeError:
                pass

        self.f = family
        self.param_labels = list(self.xcols)
        self.beta_init, self.phi_init = self.get_start_values()
        self.params_init = self.beta_init

        if isinstance(self.f, (Binomial, Poisson)) \
            or self.f.name in ["Binomial", "Poisson"]:
            self.scale_estimator = 'fixed'
        else:
            self.scale_estimator = scale_estimator

        if self.scale_estimator == 'NR':
            self.params_init = np.r_[self.params_init, np.log(self.phi_init)]
            self.param_labels += ['log_scale']
        
        
    @staticmethod
    def _unpack_params(params, scale_estimator):
        if scale_estimator == "NR":
            beta, phi, tau = params[:-1], np.exp(params[-1]), params[-1]
        else:
            beta, phi, tau = params, 1.0,  0.0
        return beta, phi, tau
    
    @staticmethod
    def _unpack_params_data(params, data, scale_estimator, f):
        X, y, weights = data
        if scale_estimator == "NR":
            beta, phi, tau = params[:-1], np.exp(params[-1]), params[-1]
        else:
            beta, phi, tau = params, 1.0,  0.0
            
        mu = f.inv_link(np.dot(X, beta))
        
        if f.name == "NegativeBinomial":
            dispersion, phi = phi, 1.0
        else:
            dispersion = 1.0
        
        if scale_estimator == "M":
            phi = f.pearson_chi2(y, mu=mu, dispersion=dispersion) / (X.shape[0] - X.shape[1])
        elif scale_estimator != "NR":
            phi = 1.0
        return X, y, weights, mu, beta, phi, tau, dispersion

    @staticmethod
    def _loglike(params, data, scale_estimator, f):
        X, y, weights, mu, beta, phi, tau, dispersion = GLM._unpack_params_data(params, data, 
                                                           scale_estimator, f)
        ll = f.loglike(y, weights=weights, mu=mu, phi=phi, dispersion=dispersion)
        return ll

    def loglike(self, params, data=None, scale_estimator=None, f=None):
        data = self.model_data if data is None else data  #data = (self.X, self.y, None) if data is None else data
        s = self.scale_estimator if scale_estimator is None \
            else scale_estimator
        f = self.f if f is None else f
        ll = self._loglike(params=params, data=data, scale_estimator=s, f=f)
        return ll

    @staticmethod
    def _full_loglike(params, data, scale_estimator, f):
        X, y, weights, mu, beta, phi, tau, dispersion = GLM._unpack_params_data(params, data,
                                                           scale_estimator, f)
        ll = f.full_loglike(y, weights=weights, mu=mu, phi=phi, dispersion=dispersion)
        return ll

    def full_loglike(self, params, data=None, scale_estimator=None, f=None):
        data = self.model_data if data is None else data  #data = (self.X, self.y, None) if data is None else data
        s = self.scale_estimator if scale_estimator is None\
            else scale_estimator
        f = self.f if f is None else f
        ll = self._full_loglike(params=params, data=data, scale_estimator=s, f=f)
        return ll

    @staticmethod
    def _gradient(params, data, scale_estimator, f):
        X, y, weights, mu, beta, phi, tau, dispersion = GLM._unpack_params_data(params, data, 
                                                           scale_estimator, f)
        w = f.gw(y, mu=mu, phi=phi, dispersion=dispersion, weights=weights)
        g = np.dot(X.T, w)
        if scale_estimator == 'NR':
            dt = np.atleast_1d(np.sum(f.dtau(tau, y, mu, weights=weights)))
            g = np.concatenate([g, dt])
        return g

    def gradient(self, params, data=None, scale_estimator=None, f=None):
        data = self.model_data if data is None else data  #data = (self.X, self.y, None) if data is None else data
        s = self.scale_estimator if scale_estimator is None\
            else scale_estimator
        f = self.f if f is None else f
        ll = self._gradient(params=params, data=data, scale_estimator=s, f=f)
        return ll
    
    @staticmethod
    def _gradient_i(params, data, scale_estimator, f):
        X, y, weights, mu, beta, phi, tau, dispersion = GLM._unpack_params_data(params, data, 
                                                           scale_estimator, f)
        w = f.gw(y, mu=mu, phi=phi, dispersion=dispersion, weights=weights)
        g = X * w.reshape(-1, 1)
        if scale_estimator == 'NR':
            dt = f.dtau(tau, y, mu, weights=weights, reduce=False).reshape(-1, 1)
            g = np.concatenate([g, dt], axis=1)
        return g

    def gradient_i(self, params, data=None, scale_estimator=None, f=None):
        data = self.model_data if data is None else data  #data = (self.X, self.y, None) if data is None else data
        s = self.scale_estimator if scale_estimator is None\
            else scale_estimator
        f = self.f if f is None else f
        ll = self._gradient_i(params=params, data=data, scale_estimator=s, f=f)
        return ll

    @staticmethod
    def _hessian(params, data, scale_estimator, f):
        X, y, weights, mu, beta, phi, tau, dispersion = GLM._unpack_params_data(params, data, 
                                                           scale_estimator, f)
        gw, hw = f.get_ghw(y, mu=mu, phi=phi, dispersion=dispersion, weights=weights)
        H = np.dot((X * hw.reshape(-1, 1)).T, X)
        if isinstance(f, NegativeBinomial) or f.name=="NegativeBinomial":
            dbdt = -np.dot(X.T, dispersion * (y - mu) /
                           ((1 + dispersion * mu)**2 * f.dlink(mu)))
        else:
            dbdt = np.dot(X.T, gw)
        #dbdt = np.dot(X.T, gw)
        if scale_estimator == 'NR':
            d2t = np.atleast_2d(f.d2tau(tau, y, mu, weights=weights))
            dbdt = -np.atleast_2d(dbdt)
            H = np.block([[H, dbdt.T], [dbdt, d2t]])
        return H

    def hessian(self, params, data=None, scale_estimator=None, f=None):
        data = self.model_data if data is None else data  #data = (self.X, self.y, None) if data is None else data
        s = self.scale_estimator if scale_estimator is None\
            else scale_estimator
        f = self.f if f is None else f
        ll = self._hessian(params=params, data=data, scale_estimator=s, f=f)
        return ll

    def _optimize(self, t_init=None, opt_kws=None, data=None, s=None, f=None):
        t_init = self.params_init if t_init is None else t_init
        data = self.model_data if data is None else data #(self.X, self.y, None) if data is None else data
        s = self.scale_estimator if s is None else s
        f = self.f if f is None else f
        opt_kws = {} if opt_kws is None else opt_kws
        default_kws = dict(method='trust-constr',
                           options=dict(verbose=0, gtol=1e-6, xtol=1e-6))
        opt_kws = {**default_kws, **opt_kws}
        args = (data, s, f)
        optimizer = sp.optimize.minimize(self.loglike, t_init, args=args,
                                         jac=self.gradient, hess=self.hessian,
                                         **opt_kws)
        return optimizer

    def get_start_values(self):
        if isinstance(self.f, Binomial):
            mu_init = (self.y * self.f.weights + 0.5) / (self.f.weights + 1)
        else:
            mu_init = (self.y + self.y.mean()) / 2.0
        eta_init = self.f.link(mu_init)
        gp = self.f.dlink(mu_init)
        vmu = self.f.var_func(mu=mu_init)
        eta_init[np.isnan(eta_init) | np.isinf(eta_init)] = 1.0
        gp[np.isnan(gp) | np.isinf(gp)] = 1.0
        vmu[np.isnan(vmu) | np.isinf(vmu)] = 1.0
        den = vmu * (gp**2)
        we = 1 / den
        we[den == 0] = 0.0
        if isinstance(self.f, Binomial):
            z = eta_init + (self.y - mu_init) * gp
        else:
            z = eta_init
        if np.any(we < 0):
            beta_init = nwls(self.X, z, we)
        else:
            beta_init = wls_qr(self.X, z, we)
        mu = self.f.inv_link(self.X.dot(beta_init))
        phi_init = self.f.pearson_chi2(self.y, mu=mu) / (self.n - self.p)
        return beta_init, phi_init

    def _fit(self, method=None, opt_kws=None):
        opt = self._optimize(opt_kws=opt_kws)
        params = opt.x
        return params, opt

    def fit(self, method=None, opt_kws=None):
        self.params, self.opt = self._fit(method, opt_kws)
        f, scale_estimator, X, y = self.f, self.scale_estimator, self.X, self.y
        weights = self.model_data.weights
        n, n_params = self.n, len(self.params)
        self.n_params = n_params
        self.params_hess = self.hessian(self.params)
        self.params_cov = np.linalg.inv(self.params_hess)
        self.params_se = np.sqrt(np.diag(self.params_cov))
        self.res = self._parameter_inference(self.params, self.params_se,
                                             n-n_params,
                                             self.param_labels)
        if scale_estimator == "NR":
            beta, phi = self.params[:-1], np.exp(self.params[-1])
            if f.name == "NegativeBinomial":
                dispersion, phi = phi, 1.0
            else:
                dispersion = 1.0
            beta_cov = self.params_cov[:-1, :-1]
            beta_se = self.params_se[:-1]
            beta_labels = self.param_labels[:-1]
        else:
            beta = self.params
            beta_cov = self.params_cov
            beta_se = self.params_se
            beta_labels = self.param_labels
            dispersion, phi = 1.0, 1.0
        self.beta_cov = self.coefs_cov = beta_cov
        self.beta_se = self.coefs_se = beta_se
        self.beta = self.coefs = beta
        self.n_beta = len(beta)
        self.beta_labels = self.coefs_labels = beta_labels
        eta = np.dot(X, beta)
        self.eta = self.linpred = eta
        mu = f.inv_link(eta)
        self.chi2 = f.pearson_chi2(y, mu=mu, phi=1.0, dispersion=dispersion, weights=weights)
        if scale_estimator == "M":
            phi = self.chi2 / (X.shape[0] - X.shape[1])
        elif scale_estimator != "NR":
            phi = 1.0
        self.phi = phi
        self.dispersion = dispersion
        WX = X * np.sqrt(f.get_ehw(y, mu, phi=phi, dispersion=dispersion).reshape(-1, 1))
        h = self._compute_leverage_qr(WX)

        self.resid_raw = y - mu
        self.resid_pearson = self.f.pearson_resid(y, mu=mu, phi=1.0, dispersion=dispersion)
        self.resid_deviance = self.f.deviance_resid(y, mu=mu, phi=1.0, dispersion=dispersion)
        self.resid_signed = self.f.signed_resid(y, mu=mu, phi=1.0, dispersion=dispersion)

        self.resid_pearson_s = self.resid_pearson / np.sqrt(phi)
        self.resid_pearson_sst = self.resid_pearson_s / np.sqrt(1 - h)
        self.resid_deviance_s = self.resid_deviance / np.sqrt(phi)
        self.resid_deviance_sst = self.resid_deviance_s / np.sqrt(1 - h)

        self.resid_likelihood = np.sign(self.resid_raw) \
            * np.sqrt(h * self.resid_pearson_sst**2
                      + (1 - h) * self.resid_deviance_sst**2)
        self.cooks_distance = (
            h * self.resid_pearson_sst**2) / (self.p * (1 - h))

        self.resids = np.vstack([self.resid_raw,
                                 self.resid_pearson,
                                 self.resid_deviance,
                                 self.resid_signed,
                                 self.resid_pearson_s,
                                 self.resid_pearson_sst,
                                 self.resid_deviance_s,
                                 self.resid_deviance_sst,
                                 self.resid_likelihood,
                                 self.cooks_distance]).T
        self.resids = pd.DataFrame(self.resids,
                                   columns=["Raw",
                                            "Pearson",
                                            "Deviance",
                                            "Signed",
                                            "PearsonS",
                                            "PearsonSS",
                                            "DevianceS",
                                            "DevianceSS",
                                            "Likelihood",
                                            "Cooks"])

        self.llf = self.f.full_loglike(y, mu=mu, phi=phi, dispersion=dispersion, weights=weights)
        if self.f.name == "NegativeBinomial":
            opt_null = self._optimize(t_init=np.zeros(
                2), data=(np.ones((self.n, 1)), self.y, None))
            self.lln = self.full_loglike(
                opt_null.x, data=(np.ones((self.n, 1)), self.y, weights))
        else:
            self.lln = self.f.full_loglike(
                y, mu=np.ones(mu.shape[0])*y.mean(), phi=phi, weights=weights)

        k = len(self.params)
        sumstats = {}
        self.aic, self.aicc, self.bic, self.caic = self._get_information(
            self.llf, k, self.n_obs)
        self.r2_cs, self.r2_nk, self.r2_mc, self.r2_mb, self.llr = \
            self._get_pseudo_rsquared(self.llf, self.lln, k, self.n_obs)
        self.r2 = self._rsquared(y, mu)
        sumstats["AIC"] = self.aic
        sumstats["AICC"] = self.aicc
        sumstats["BIC"] = self.bic
        sumstats["CAIC"] = self.caic
        sumstats["R2_CS"] = self.r2_cs
        sumstats["R2_NK"] = self.r2_nk
        sumstats["R2_MC"] = self.r2_mc
        sumstats["R2_MB"] = self.r2_mb
        sumstats["R2_SS"] = self.r2
        sumstats["LLR"] = self.llr
        sumstats["LLF"] = self.llf
        sumstats["LLN"] = self.lln
        sumstats["Deviance"] = np.sum(f.deviance(y=y, mu=mu, phi=1.0, 
                                                 dispersion=dispersion,
                                                 weights=weights))
        sumstats["Chi2"] = self.chi2
        sumstats = pd.DataFrame(sumstats, index=["Statistic"]).T
        self.sumstats = sumstats
        self.mu = mu
        self.h = h

    
    def _one_step_approx(self, WX, h, rp):
        y = rp / np.sqrt(1 - h)
        db = WX.dot(np.linalg.inv(np.dot(WX.T, WX))) * y.reshape(-1, 1)
        return db
        
    def get_robust_res(self, grad_kws=None):
        default_kws = dict(params=self.params,
                           data=(self.X, self.y, self.f.weights), 
                           scale_estimator=self.scale_estimator,
                           f=self.f)
        grad_kws = handle_default_kws(grad_kws, default_kws)
        G = self.gradient_i(**grad_kws)
        B = np.dot(G.T, G)
        A = self.params_cov
        V = A.dot(B).dot(A)
        res = self._parameter_inference(self.params, np.sqrt(np.diag(V)),
                                        self.n-len(self.params),
                                        self.param_labels)
        return res

    @staticmethod
    def _predict(coefs, X, f, phi=1.0, dispersion=1.0, coefs_cov=None, linpred=True,
                 linpred_se=True,  mean=True, mean_ci=True, mean_ci_level=0.95,
                 predicted_ci=True, predicted_ci_level=0.95):
        res = {}
        eta = np.dot(X, coefs)
        if linpred_se or mean_ci:
            eta_se = wdiag_outer_prod(X, coefs_cov, X)

        if mean or mean_ci or predicted_ci:
            mu = f.inv_link(eta)

        if linpred:
            res["eta"] = eta
        if linpred_se:
            res["eta_se"] = eta_se
        if mean or mean_ci or predicted_ci:
            mu = f.inv_link(eta)
            res["mu"] = mu

        if mean_ci:
            mean_ci_level = symmetric_conf_int(mean_ci_level)
            mean_ci_lmult = sp.special.ndtri(mean_ci_level)
            res["eta_lower_ci"] = eta - mean_ci_lmult * eta_se
            res["eta_upper_ci"] = eta + mean_ci_lmult * eta_se
            res["mu_lower_ci"] = f.inv_link(res["eta_lower_ci"])
            res["mu_upper_ci"] = f.inv_link(res["eta_upper_ci"])

        if predicted_ci:
            predicted_ci_level = symmetric_conf_int(predicted_ci_level)
            v = f.variance(mu=mu, phi=phi, dispersion=dispersion)
            res["predicted_lower_ci"] = f.ppf(
                1-predicted_ci_level, mu=mu, scale=v)
            res["predicted_upper_ci"] = f.ppf(
                predicted_ci_level, mu=mu, scale=v)
        return res
    
    def _jacknife(self, method="optimize", verbose=True):
        if type(self.f.weights) is np.ndarray:
            weights = self.f.weights
        else:
            weights = np.ones(self.n_obs)
        if method == "optimize":
            jacknife_samples = np.zeros((self.n_obs, self.n_params))
            ii = np.ones(self.n_obs, dtype=bool)
            pbar = tqdm.tqdm(total=self.n_obs) if verbose else None
            for i in range(self.n_obs):
                ii[i] = False
                opt = self._optimize(data=(self.X[ii], self.y[ii], weights[ii]), f=self.f)
                jacknife_samples[i] = opt.x
                ii[i] = True
                if verbose:
                    pbar.update(1)
            if verbose:
                pbar.close()
        elif method == "one-step":    
            w = self.f.get_ehw(self.y, self.mu, phi=self.phi,
                               dispersion=self.dispersion,
                               weights=weights).reshape(-1, 1)
            WX = self.X * w
            h = self._compute_leverage_qr(WX)
            one_step = self._one_step_approx(WX, h, self.resid_pearson_s)
            jacknife_samples = self.params.reshape(1, -1) - one_step
        jacknife_samples = pd.DataFrame(jacknife_samples, index=self.xinds,
                                        columns=self.param_labels)
        return jacknife_samples
    
    
    def _bootstrap(self, n_boot=1000, verbose=True, rng=None):
        rng = np.random.default_rng() if rng is None else rng 
        n, q = self.n_obs, self.n_params
        weights = self.f.weights
        w = weights if type(weights) is np.ndarray else np.ones(n)
        X, y = self.X, self.y
        pbar = tqdm.tqdm(total=n_boot) if verbose else None
        boot_samples = np.zeros((n_boot, q))
        for i in range(n_boot):
            ii = rng.choice(n, size=n, replace=True)
            opt = self._optimize(data=(X[ii], y[ii], w[ii]), f=self.f)
            boot_samples[i] = opt.x
            pbar.update(1)
        if verbose:
            pbar.close()
        boot_samples = pd.DataFrame(boot_samples,columns=self.param_labels)

        return boot_samples
    
