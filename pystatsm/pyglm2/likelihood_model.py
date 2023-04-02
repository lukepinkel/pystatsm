#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 01:15:40 2023

@author: lukepinkel
"""


import numpy as np
import scipy as sp
import scipy.stats
import pandas as pd
from abc import ABCMeta, abstractmethod
from ..utilities import output
from ..utilities.func_utils import handle_default_kws
              
                                           

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