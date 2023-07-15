#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 07:59:44 2023

@author: lukepinkel
"""

import numpy as np
import pandas as pd
import scipy as sp

from ..utilities.func_utils import handle_default_kws
from ..utilities.linalg_operations import _vech, _invech
from ..utilities.output import get_param_table
from .derivatives import _dloglike_mu, _d2loglike_mu, _dsigma_mu, _d2sigma_mu, _dloglike_mu_alt
from .model_specification import ModelSpecification

pd.set_option("mode.chained_assignment", None)


class SEM(ModelSpecification):
    def __init__(self, formula, data, group_col=None, model_spec_kws=None, group_kws=None):
        """
        Class for fitting structural equation models
        Parameters
        ----------
        formula : str
            Model formula
        data : pd.DataFrame
            Dataframe containing the data
        group_col : str, optional
            Column name of the grouping variable, by default None
        model_spec_kws : dict, optional
            Keyword arguments for ModelSpecification, by default None and not
            currently implemented
        group_kws : dict, optional
            Keyword arguments for the group parameters, by default None
            and not fully implemented
        """
        group_kws = dict(shared=[True] * 6) if group_kws is None else group_kws
        super().__init__(formula, data, group_col, **group_kws)

    def fit_func(self, theta, per_group=False):
        """
        Function to be minimized in the optimization problem.
        Currently this is the negative log-likelihood function.
        Parameters
        ----------
        theta : np.ndarray
            Vector of free parameters
        per_group
            If True, the function is evaluated for each group separately
        Returns
        -------
        f : float or np.ndarray
            Value of the function
        """
        free = self.transform_theta_to_free(theta)
        if per_group:
            f = np.zeros(self.n_groups)
            if np.iscomplexobj(theta):
                f = f.astype(complex)
        else:
            f = 0.0
        for i in range(self.n_groups):
            group_free = self.transform_free_to_group_free(free, i)
            par = self.group_free_to_par(group_free, i)
            mats = self.par_to_model_mats(par)
            Sigma, mu = self._implied_cov_mean(*mats)
            r = (self.model_data.sample_mean[i] - mu).flatten()
            V = np.linalg.inv(Sigma)
            trSV = np.trace(V.dot(self.model_data.sample_cov[i]))
            rVr = np.dot(r.T.dot(V), r)
            if np.any(np.iscomplex(Sigma)):
                s, lnd = np.linalg.slogdet(Sigma)
                lndS = np.log(s) + lnd
            else:
                s, lndS = np.linalg.slogdet(Sigma)
            fi = (rVr + lndS + trSV +
                  self.model_data.const[i]) * self.gweights[i]
            if (s == -1) or (fi < -1):
                fi += np.inf
            if per_group:
                f[i] = fi
            else:
                f += fi
        f = self.n / 2 * f
        return f

    def gradient(self, theta, per_group=False, method=0):
        """
        Gradient of the fit function
        Parameters
        ----------
        theta : np.ndarray
            Vector of free parameters
        per_group
            If True, the function is evaluated for each group separately
        method : int, optional
            Method to use for computing the gradient, by default 0

        Returns
        -------
        g : np.ndarray
            Gradient of the fit function. Either of size (n_groups, n_group_theta)
            or (n_group_theta, )
        """
        free = self.transform_theta_to_free(theta)
        if per_group:
            g = np.zeros((self.n_groups, self.n_group_theta))
        else:
            g = np.zeros(self.n_group_theta)
        for i in range(self.n_groups):
            group_free = self.transform_free_to_group_free(free, i)
            par = self.group_free_to_par(group_free, i)
            L, B, F, P, a, b = self.par_to_model_mats(par)
            B = np.linalg.inv(np.eye(B.shape[0]) - B)
            LB = np.dot(L, B)
            mu = (a + LB.dot(b.T).T).reshape(-1)
            a, b = a.flatten(), b.flatten()
            Sigma = LB.dot(F).dot(LB.T) + P
            R = self.model_data.sample_cov[i] - Sigma
            V = np.linalg.inv(Sigma)
            VRV = V.dot(R).dot(V)
            rtV = (self.model_data.sample_mean[i].flatten() - mu).dot(V)
            gi = np.zeros(self.nf)
            kws = self._grad_kws
            if method == 0:
                gi = _dloglike_mu(gi, L, B, F, b, VRV, rtV, **kws) * self.gweights[i]
            else:
                kws = self._grad_kws1
                gi = _dloglike_mu_alt(gi, L, B, F, b, VRV, rtV, **kws) * self.gweights[i]
            gi = self.jac_free_to_theta(self.jac_group_free_to_free(gi, i))
            if per_group:
                g[i] = gi
            else:
                g = g + gi
        g = self.n / 2 * g
        return g

    def hessian(self, theta, per_group=False):
        """
        Hessian of the fit function
        Parameters
        ----------
        theta : np.ndarray
            Vector of free parameters
        per_group : bool, optional
            If True, the function is evaluated for each group separately, by default False

        Returns
        -------
        H : np.ndarray
            Hessian of the fit function. Either of size (n_groups, n_group_theta, n_group_theta)
            or (n_group_theta, n_group_theta)
        """
        free = self.transform_theta_to_free(theta)
        if per_group:
            H = np.zeros(
                (self.n_groups, self.n_group_theta, self.n_group_theta))
        else:
            H = np.zeros((self.n_group_theta, self.n_group_theta))
        for i in range(self.n_groups):
            group_free = self.transform_free_to_group_free(free, i)
            par = self.group_free_to_par(group_free, i)
            L, B, F, P, a, b = self.par_to_model_mats(par)
            B = np.linalg.inv(np.eye(B.shape[0]) - B)
            LB = np.dot(L, B)
            mu = (a + LB.dot(b.T).T).reshape(-1)
            a, b = a.flatten(), b.flatten()
            kws = self._hess_kws
            Sigma = LB.dot(F).dot(LB.T) + P
            S = self.model_data.sample_cov[i]
            R = S - Sigma
            V = np.linalg.inv(Sigma)
            VRV = V.dot(R).dot(V)
            rtV = (self.model_data.sample_mean[i].flatten() - mu).dot(V)
            Hi = np.zeros((self.nf,) * 2)
            d1Sm = np.zeros((self.nf, self.p, self.p + 1))
            Hi = _d2loglike_mu(H=Hi, d1Sm=d1Sm, L=L, B=B, F=F, a=a, b=b, VRV=VRV, rtV=rtV,
                               V=V, **kws)
            Hi = Hi * self.gweights[i]
            Hi = self.jac_group_free_to_free(Hi, i, axes=(0, 1))
            Hi = self.jac_free_to_theta(Hi, axes=(0, 1))
            if per_group:
                H[i] = Hi
            else:
                H = H + Hi
        H = self.n / 2 * H
        return H

    def dsigma_mu(self, theta):
        """
        Derivative of the covariance and mean with respect to the free parameters
        Covariance and mean are organized [vech(Cov), mean] and the derivatives are
        therefore of  size (p*(p+1)/2 + p, nf)
        Parameters
        ----------
        theta : np.ndarray
            Vector of free parameters

        Returns
        -------
        dSm : np.ndarray
            Derivative of the covariance and mean with respect to the free parameters

        """
        free = self.transform_theta_to_free(theta)
        dSm = self.dSm.copy()
        for i in range(self.n_groups):
            group_free = self.transform_free_to_group_free(free, i)
            par = self.group_free_to_par(group_free, i)
            L, B, F, _, a, b = self.par_to_model_mats(par)
            B = np.linalg.inv(np.eye(B.shape[0]) - B)
            kws = dict(dA=self.dA, m_size=self.m_size, m_type=self.m_kind,
                       vind=self._vech_inds, n=self.nf, p2=self.p2)
            a, b = a.flatten(), b.flatten()
            dSm[i] = _dsigma_mu(dSm[i], L, B, F, b, **kws)
        return dSm

    def d2sigma_mu(self, theta):
        """
        Second derivative of the covariance and mean with respect to the free parameters
        Covariance and mean are organized [vech(Cov), mean] and the derivatives are
        therefore of  size (p*(p+1)/2 + p, nf, nf)
        Parameters
        ----------
        theta : np.ndarray
            Vector of free parameters

        Returns
        -------
        d2Sm : np.ndarray
            Second derivative of the covariance and mean with respect to the free parameters

        """
        free = self.transform_theta_to_free(theta)
        d2Sm = self.d2Sm.copy()
        for i in range(self.n_groups):
            group_free = self.transform_free_to_group_free(free, i)
            par = self.group_free_to_par(group_free, i)
            L, B, F, _, a, b = self.par_to_model_mats(par)
            B = np.linalg.inv(np.eye(B.shape[0]) - B)
            a, b = a.flatten(), b.flatten()
            kws = dict(dA=self.dA, m_size=self.m_size, m_type=self.d2_kind,
                       d2_inds=self.d2_inds, vind=self._vech_inds, n=self.nf2,
                       p2=self.p2)
            d2Sm[i] = _d2sigma_mu(d2Sm[i], L, B, F, b, **kws)
        return d2Sm

    def _implied_cov_mean(self, L, B, F, P, a, b):
        B = np.linalg.inv(np.eye(B.shape[0]) - B)
        LB = np.dot(L, B)
        Sigma = LB.dot(F).dot(LB.T) + P
        mu = (a + LB.dot(b.T).T).reshape(-1)
        return Sigma, mu

    def implied_cov_mean(self, theta):
        """
        Compute the implied covariance and mean from the free parameters but
        instead results in two arrays of size
        Parameters
        ----------
        theta : np.ndarray
            Vector of free parameters

        Returns
        -------
        Sigma : np.ndarray
            Implied covariance matrix
        mu : np.ndarray
            Implied mean vector
        """
        free = self.transform_theta_to_free(theta.copy())
        Sigma = np.zeros((self.n_groups, self.p, self.p))
        mu = np.zeros((self.n_groups, self.p))
        for i in range(self.n_groups):
            group_free = self.transform_free_to_group_free(free, i)
            par = self.group_free_to_par(group_free, i)
            mats = self.par_to_model_mats(par)
            Sigma[i], mu[i] = self._implied_cov_mean(*mats)
        return Sigma, mu

    def implied_sample_stats(self, theta):
        """
        Compute the implied sample statistics from the free parameters
        organized as [vech(Cov), mean]
        Parameters
        ----------
        theta : np.ndarray
            Vector of free parameters

        Returns
        -------
        Sigmamu : np.ndarray
            Implied covariance and mean vector
        """
        free = self.transform_theta_to_free(theta.copy())
        Sigmamu = np.zeros((self.n_groups, self.p2 + self.p))
        if np.iscomplexobj(theta):
            Sigmamu = Sigmamu.astype(complex)
        for i in range(self.n_groups):
            group_free = self.transform_free_to_group_free(free, i)
            par = self.group_free_to_par(group_free, i)
            L, B, F, P, a, b = self.par_to_model_mats(par)
            B = np.linalg.inv(np.eye(B.shape[0]) - B)
            LB = np.dot(L, B)
            Sigma = LB.dot(F).dot(LB.T) + P
            mu = (a + LB.dot(b.T).T).reshape(-1)
            s, mu = _vech(Sigma), mu.flatten()
            Sigmamu[i, :self.p2] = s
            Sigmamu[i, self.p2:] = mu
        return Sigmamu

    def gradient_obs(self, theta):
        """
        Gradient of the log-likelihood with respect to the free parameters
        for each observation
        Parameters
        ----------
        theta : np.ndarray
            Vector of free parameters
        Returns
        -------
        g : np.ndarray
            Gradient of the log-likelihood with respect to the free parameters
            of size (n, n_total_free)
        """
        g = np.zeros((self.n, self.n_total_free))
        Sigmas, mus = self.implied_cov_mean(theta)
        dSms = self.dsigma_mu(theta)
        for i in range(self.n_groups):
            Sigma, mu = Sigmas[i], mus[i]
            dSm = dSms[i]
            dS = dSm[:-self.p]
            dm = dSm[-self.p:]
            dS = _invech(dS.T)
            V = np.linalg.inv(Sigma)
            DS = dS.reshape(self.nf, -1, order='F')
            t1 = DS.dot(V.reshape(-1, order='F')).reshape(1, -1)
            Y = self.model_data.data[self.model_data.group_indices[i]] - mu
            YV = Y.dot(V)
            t2 = YV.dot(dm)
            t3 = np.einsum("ij,hjk,ik->ih", YV, dS, YV, optimize=True)
            gi = -2 * (-t1 / 2 + t2 + t3 / 2)
            gi = self.jac_group_free_to_free(gi.T, i).T
            g[self.model_data.group_indices[i]] = gi
        g = self.jac_free_to_theta(g, axes=(1,)) / 2
        return g

    def loglike(self, theta, level="sample"):
        """
        Log-likelihood of the model at various levels
        Parameters
        ----------
        theta : np.ndarray
            Vector of free parameters
        level : str
            Level at which to compute the log-likelihood.  One of "sample",
            "observation", or "group"

        Returns
        -------
        ll : float or np.ndarray
            Log-likelihood of the model at the specified level. If level is
            "sample" or "observation", then ll is a vector of length n. If
            level is "group", then ll is a vector of length n_groups.
        """
        free = self.transform_theta_to_free(theta)
        if level == "group":
            f = np.zeros(self.n_groups)
        elif level == "observation" or level == "sample":
            f = np.zeros(self.n)
        if np.iscomplexobj(theta):
            f = f.astype(complex)
        for i in range(self.n_groups):
            ix = self.model_data.group_indices[i]
            group_free = self.transform_free_to_group_free(free, i)
            par = self.group_free_to_par(group_free, i)
            mats = self.par_to_model_mats(par)
            Sigma, mu = self._implied_cov_mean(*mats)
            L = sp.linalg.cholesky(Sigma)
            Y = self.model_data.data[ix] - mu
            # np.dot(X, np.linalg.inv(L.T))
            Z = sp.linalg.solve_triangular(L, Y.T, trans=1).T
            t1 = 2.0 * np.log(np.diag(L)).sum()
            t2 = np.sum(Z ** 2, axis=1)
            ll = (t1 + t2 + self.ll_const) / 2
            if level == "group":
                f[i] = np.sum(ll)
            else:
                f[ix] = ll
        if level == "sample":
            f = np.sum(f)
        return f

    def _fit(self, theta_init=None, minimize_kws=None, minimize_options=None, use_hess=False):
        bounds = self.bounds
        fit_func = self.fit_func
        grad = self.gradient
        if use_hess:
            hess = self.hessian
        else:
            hess = None
        theta = self.theta if theta_init is None else theta_init
        if "method" in minimize_options:
            if minimize_options["method"] == "trust-constr":
                default_minimize_options = dict(initial_tr_radius=1.0, verbose=3)
            else:
                default_minimize_options = {}
        else:
            default_minimize_options = {}
        minimize_options = handle_default_kws(
            minimize_options, default_minimize_options)

        default_minimize_kws = dict(
            method="trust-constr", options=minimize_options)
        minimize_kws = handle_default_kws(minimize_kws, default_minimize_kws)
        res = sp.optimize.minimize(fit_func, x0=theta, jac=grad, hess=hess,
                                   bounds=bounds, **minimize_kws)
        return res

    def get_standard_errors(self, theta, robust=False):
        cov_params = np.linalg.inv(self.hessian(theta))
        if robust:
            G = self.gradient_obs(theta)
            M = G.T.dot(G)
            V = cov_params
            cov_params = np.dot(V, M.dot(V))
        se_params = np.sqrt(np.diag(cov_params))
        return se_params

    def fit(self, robust_se=False, theta_init=None, minimize_kws=None, minimize_options=None, use_hess=False):
        res = self._fit(theta_init=theta_init, minimize_kws=minimize_kws,
                        minimize_options=minimize_options, use_hess=use_hess)
        if np.linalg.norm(res.get("grad", res.get("jac", self.gradient(res.x)))) > 1e16:
            if minimize_options is None:
                minimize_options = {}
            minimize_options["initial_tr_radius"] = 0.01
            res = self._fit(minimize_kws=minimize_kws,
                            minimize_options=minimize_options, use_hess=use_hess)
        self.opt_res = res
        self.theta = res.x
        self.free = self.transform_theta_to_free(self.theta)
        params = res.x
        se_params = self.get_standard_errors(params, robust=robust_se)
        self.res = get_param_table(params, se_params, degfree=self.n - 2,
                                   index=self.theta_names)
        self.res_free = pd.DataFrame(self.transform_theta_to_free(self.res.values),
                                     index=self.free_names, columns=self.res.columns)
        mats = {}
        free = self.transform_theta_to_free(self.theta)
        for i in range(self.n_groups):
            group_free = self.transform_free_to_group_free(free, i)
            mat_list = self.group_free_to_model_mats(group_free, i)
            mats[i] = {}
            for j, mat in enumerate(mat_list):
                mats[i][j] = pd.DataFrame(mat, index=self.mat_labels[0][j], 
                                              columns=self.mat_labels[1][j])
        self.mats = mats
