#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 06:48:04 2023

@author: lukepinkel
"""
import tqdm
import numba
import numpy as np
import scipy as sp
from .regression_model import RegressionMixin
from .likelihood_model import LikelihoodModel
from ..utilities import output

LN2PI = np.log(2 * np.pi)

class OLS(RegressionMixin, LikelihoodModel):
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
            Whether to use Restricted Maximum Likelihood (REML) estimation.
            Default is True.
    
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
        Compute the Hessian of the log-likelihood function with respect to 
        the parameters.
        
        Parameters
        ----------
        params : array_like
            Model parameters.
        reml : bool, optional
            Whether to use Restricted Maximum Likelihood (REML) estimation. 
            Default is True.
        
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
        Linv, _ = sp.linalg.lapack.dtrtri(L, lower=1)
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
        Fit the linear model
        
        Parameters
        ----------
        reml : bool, optional
            Whether to use Restricted Maximum Likelihood (REML) estimation.
            Default is True.
        
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
            Tuple containing the predictor variables (X) and 
            response variable (y).
        C : ndarray
            Constraint matrix.
        d : array_like, optional
            Vector of constants for the constraints. Default is None 
            (assumed to be a zero vector).
        L : ndarray, optional
            Lower-triangular Cholesky factor of the Gram matrix. Default is None.
        Linv : ndarray, optional
            Inverse of the lower-triangular Cholesky factor. Default is None.
        
        Returns
        -------
        tuple
            Tuple containing the constrained coefficients and the sum of
            squared errors for the constrained model.
        """
        if Linv is None:
            if L is None:
                X, y = data
                L = np.linalg.cholesky(X.T.dot(X))
            Linv, _ = sp.linalg.lapack.dtrtri(L, lower=1)
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
        Perform a non-parametric bootstrap to estimate model coefficients
        and error variance.
        
        Parameters
        ----------
        n_boot : int
            Number of bootstrap samples to generate.
        verbose : bool, optional
            Whether to display a progress bar during bootstrapping.
            Default is True.
        
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
        Perform a non-parametric bootstrap using Cholesky decomposition 
        (JIT-compiled with Numba).
        
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
        Perform a non-parametric bootstrap using the JIT-compiled Cholesky 
        decomposition method.
    
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
            Tuple containing p-values adjusted for family-wise error 
            rate (FWER) and unadjusted p-values.
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
    
    