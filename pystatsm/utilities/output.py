#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 23 03:54:38 2020

@author: lukepinkel
"""
import numpy as np
import scipy as sp
import scipy.stats
import pandas as pd
from .data_utils import _check_shape

def get_param_table(params, 
                    se_params,
                    degfree=None,
                    index=None,
                    parameter_label=None,
                    pdist=None, 
                    p_const=2.0,
                    alpha=0.05):
    """
    Creates a parameter summary table with confidence intervals.

    Parameters
    ----------
    params : array-like
        A 1D array of parameter estimates.
    se_params : array-like
        A 1D array of standard errors corresponding to the parameter estimates.
    degfree : int, optional
        The degrees of freedom for the t-distribution, if applicable.
    index : array-like, optional
        The index for the resulting DataFrame.
    parameter_label : str, optional
        The label for the parameter column in the resulting DataFrame. Default is 'parameter'.
    pdist : scipy.stats.rv_continuous, optional
        The probability distribution to use for calculating p-values and confidence intervals.
        Default is the t-distribution with the specified degrees of freedom.
    p_const : float, optional
        The constant to multiply the p-values by. Default is 2.0.
    alpha : float, optional
        The significance level for the confidence intervals. Default is 0.05.

    Returns
    -------
    df : pandas.DataFrame
        A DataFrame containing the parameter summary table with confidence intervals.
    """
    parameter_label = 'parameter' if parameter_label is None else parameter_label
    degfree = np.inf if degfree is None else degfree
    pdist = sp.stats.t(degfree) if pdist is None else pdist
    arr = np.vstack((_check_shape(params, 1), _check_shape(se_params, 1))).T
    df = pd.DataFrame(arr, index=index, columns=[parameter_label, 'SE'])
    df['t'] = df[parameter_label] / df['SE']

    df['p'] = pdist.sf(np.abs(df['t'])) * p_const
    ci_lower = df[parameter_label] + pdist.ppf(alpha/2) * df["SE"]
    ci_upper = df[parameter_label] + pdist.ppf(1 - alpha/2) * df["SE"]
    ci_label = f"CI{100*(1-alpha):g}"
    df[[f"Lower{ci_label}", f"Upper{ci_label}"]] = np.column_stack((ci_lower, ci_upper))
    return df
def compute_bias_corrected_bootstrap_intervals(params, jack_samples, boot_samples, alpha=0.05):
    """
    Compute bias-corrected and accelerated (BCa) bootstrap confidence intervals for the given parameters.

    Parameters
    ----------
    params : array-like
        A 1D array of parameter estimates (n_params,)
    jack_samples : array-like
        A 2D array of leave-one-out jackknife resamples (n_obs, n_params)
    boot_samples : array-like
        A 2D array of bootstrap resamples (n_samples, n_params)
    alpha : float, optional, default: 0.05
        The significance level for the confidence intervals.

    Returns
    -------
    conf_ints : numpy.ndarray
        A 2D array containing the lower and upper confidence intervals for each parameter.
    """
    params = np.asarray(params)
    jack_samples = np.asarray(jack_samples)
    boot_samples = np.asarray(boot_samples)

    z0 = sp.special.ndtri(np.mean(params < boot_samples, axis=0)).reshape(-1, 1)
    jack_mean = np.mean(jack_samples, axis=0)
    d = jack_mean - jack_samples
    d2 = d * d
    d3 = d2 * d
    num = np.sum(d3, axis=0).reshape(-1, 1)
    den = 6.0 * np.power(np.sum(d2, axis=0).reshape(-1, 1), 3 / 2)
    a = np.divide(num, den, out=np.zeros_like(num), where=den != 0)

    # Calculate left and right z-values
    zl, zr = sp.special.ndtri([alpha / 2, 1 - alpha / 2]) 
    
    # Combine left and right z-values and compute their common expressions
    z_values = np.array([[zl, zr]])
    cse = z0 + z_values
    
    # Calculate left and right adjusted z-values
    za_values = z0 + np.divide(cse, 1.0 - a * cse, where=(1.0 - a * cse) != 0)
    
    # Calculate left and right quantiles
    q_values = sp.special.ndtr(za_values)

    conf_ints = np.zeros((len(params), 2))
    for i in range(len(params)):
        conf_ints[i] = np.quantile(boot_samples[:, i], q_values[i])
    return conf_ints