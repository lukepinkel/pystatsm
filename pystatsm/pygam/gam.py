# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 17:01:11 2021

@author: lukepinkel
"""

import patsy
import numpy as np
import scipy as sp
import scipy.stats
import scipy.linalg
import pandas as pd
import scipy.interpolate
import matplotlib.pyplot as plt
from .smooth_setup import (parse_smooths, get_parametric_formula,
                           get_smooth_terms, get_smooth_matrices)
from ..pyglm.families import Gaussian, InverseGaussian, Gamma
from ..utilities.splines import (crspline_basis, bspline_basis, ccspline_basis,
                                 absorb_constraints)
from ..utilities.numerical_derivs import so_gc_cd
from ..utilities.linalg_operations import wcrossp



class GAM:

    def __init__(self, formula, data, family=None):
        """
        parameters
        ----------

        formula: str
            Formula for model

        data: dataframe
            Model data

        family: Family, optional
            Model family. Defaults to Gaussian.

        """
        if family is None:
            family = Gaussian()
        smooth_info = parse_smooths(formula, data)
        formula = get_parametric_formula(formula)
        y, Xp = patsy.dmatrices(formula, data, return_type='dataframe', eval_env=1)
        smooths, n_smooth_terms, n_total_params, varnames = get_smooth_terms(
                                                                smooth_info, Xp)
        X, S, ranks, ldS = get_smooth_matrices(Xp, smooths, n_smooth_terms,
                                               n_total_params)
        self.X, self.Xp, self.y = np.concatenate(X, axis=1), Xp.values, y.values[:, 0]
        self.S = S
        self.ranks = np.asarray(ranks, dtype=float)
        self.ldS = np.asarray(ldS, dtype=float)
        self.ldS_sum = self.ldS.sum()
        self.f, self.smooths = family, smooths
        self.ns, self.n_obs, self.nx = n_smooth_terms, Xp.shape[0], n_total_params
        self.mp = self.nx - self.ranks.sum()
        self.data = data
        theta = np.zeros(self.ns+1)
        for i, (var, s) in enumerate(smooths.items()):
            ix = smooths[var]['ix']
            a = self.S[i][ix, ix[:, None].T]
            d = np.diag(self.X[:, ix].T.dot(self.X[:, ix]))
            lam = (1.5 * (d / a)[a>0]).mean()
            theta[i] = np.log(lam)
            varnames += [f"log_smooth_{var}"]
        theta[-1] = 1.0
        varnames += ["log_scale"]
        self.theta = theta
        self.varnames = varnames
        self.smooth_info = smooth_info


    def get_wz(self, eta):
        """
        Parameters
        ----------
        eta: array of shape (n_obs, )
            Linear predictor X*beta

        Returns
        -------
        z: array of shape (n_obs, )
            Pseudo data / working variate
            z = eta + (y - mu) dg(mu)

        w: array of shape (n_obs, )
            Regression weights

        """
        mu = self.f.inv_link(eta)
        v0, v1 = self.f.var_func(mu=mu), self.f.dvar_dmu(mu)
        g1, g2 = self.f.dlink(mu),  self.f.d2link(mu)
        r = self.y - mu
        a = 1.0 + r * (v1 / v0 + g2 / g1)
        z = eta + r * g1 / a
        w = a / (g1**2 * v0)
        return z, w

    def solve_pls(self, eta, S):
        """
        Parameters
        ----------
        eta: array of shape (n_obs, )
            Linear predictor X*beta

        S: array of shape (nx, nx)
            Penalty matrix

        Returns
        -------
        beta_new: array of shape (nx, )
            Solution to the penalized least squares equation

        """
        z, w = self.get_wz(eta)
        Xw = self.X * w[:, None]
        XtWX = Xw.T.dot(self.X)
        rhs = Xw.T.dot(z)
        try:
            c, low = sp.linalg.cho_factor(XtWX + S, lower=True, check_finite=False)
            beta_new = sp.linalg.cho_solve((c, low), rhs, check_finite=False)
        except np.linalg.LinAlgError:
            beta_new = np.linalg.solve(XtWX + S, rhs)
        return beta_new

    def pirls(self, lam, n_iters=200, tol=1e-12):
        """
        Parameters
        ----------
        lam: array of shape (ns, )
            Smoothing penalty

        n_iters: int, optional, default=200
            Maximum number of penalized iteratively reweighted
            least squares iterations

        tol: float, optional, default=1e-12
            Tolerance for change in deviance over iterations before
            convergence is declared

        Returns
        -------
        beta: array of shape (nx, )
            Array of coefficients

        eta: array of shape (n_obs, )
            Array of linear predictors

        mu: array of shape (n_obs, )
            Modeled mean

        dev: float
            Deviance at convergence

        convergence: bool
            Whether or not convergence was reached

        i: int
            Number of PIRLS iterations

        """
        S = self.get_penalty_mat(lam)
        eta_prev = self.f.link(self.y)
        dev_prev = 1e16 #self.f.deviance(self.y, mu=self.f.inv_link(eta)).sum()
        convergence = False
        beta_prev = np.zeros(self.X.shape[1])
        for i in range(n_iters):
            beta = self.solve_pls(eta_prev, S)
            eta = self.X.dot(beta)
            mu = self.f.inv_link(eta)
            dev = self.f.deviance(self.y, mu=mu).sum()+beta.T.dot(S).dot(beta)

            if abs(dev - dev_prev) / (abs(dev_prev)+1e-6) < tol:
                convergence = True
                break
            elif dev > dev_prev:
                j = 0
                while ((j < 15)&(dev > dev_prev)):
                    beta = (beta + beta_prev) / 2.0
                    eta = self.X.dot(beta)
                    mu = self.f.inv_link(eta)
                    dev = self.f.deviance(self.y, mu=mu).sum()+beta.T.dot(S).dot(beta)
                    j+=1
                if j==15:
                    convergence = False
                    break
            beta_prev, eta_prev, dev_prev = beta, eta, dev
        return beta, eta, mu, dev, convergence, i

    def get_penalty_mat(self, lam):
        """
        Parameters
        ----------
        lam: array of shape (ns, )
            Smoothing penalty

        Returns
        -------
        Sa: array of shape (nx, nx)
            Smoothing penalty matrix

        """
        Sa = np.einsum('i,ijk->jk', lam, self.S)
        return Sa

    def logdetS(self, lam, phi):
        """
        Parameters
        ----------
        lam: array of shape (ns, )
            Smoothing penalty

        phi: float
            Scale

        Returns
        -------
        logdet: float
            log determinant of penalty matrix

        """
        logdet = self.ranks.dot(np.log(lam) - np.log(phi)) + self.ldS_sum
        return logdet

    def grad_beta_rho(self, beta, lam):
        """
        Parameters
        ----------
        beta: array of shape (nx,)
            Model coefficients

        lam: array of shape (ns, )
            Smoothing penalty


        Returns
        -------
        dbdr: array of shape (nx, ns)
            Derivative of beta with respect to log smoothing parameters

        """
        S = self.get_penalty_mat(lam)
        _, Dp2 = self.hess_dev_beta(beta, S)
        Sb = np.einsum('ikm,m->ki', self.S, beta)  # (nx, ns); column i is S_i β
        dbdr = -lam[None, :] * np.linalg.solve(Dp2, Sb)
        return dbdr

    def hess_beta_rho(self, beta, lam):
        """
        Parameters
        ----------
        beta: array of shape (nx,)
            Model coefficients

        lam: array of shape (ns, )
            Smoothing penalty


        Returns
        -------
        b2: array of shape (ns, ns, nx)
            Second derivative of beta with respect to log
            smoothing parameters

        """
        S = self.get_penalty_mat(lam)
        b1 = self.grad_beta_rho(beta, lam)
        _, Dp2 = self.hess_dev_beta(beta, S)
        mu = self.f.inv_link(self.X.dot(beta))
        w1 = self.f.dw_deta(self.y, mu)
        eta1 = self.X.dot(b1)  # (n, ns)
        Tw = self.X * w1[:, None]
        Uij = eta1[:, :, None] * eta1[:, None, :]  # (n, ns, ns)
        Xfij = Tw.T.dot(Uij.reshape(self.n_obs, -1)).reshape(self.nx, self.ns, self.ns)
        Sb = np.einsum('ikm,mj->kij', self.S, b1)  # (nx, ns, ns); Sb[:, i, j] = S_i b1_j
        u = Xfij + lam[None, :, None] * Sb + lam[None, None, :] * Sb.transpose(0, 2, 1)
        Ainv_u = np.linalg.solve(Dp2, u.reshape(self.nx, -1)).reshape(self.nx, self.ns, self.ns)
        b2 = -Ainv_u.transpose(1, 2, 0)  # (ns, ns, nx)
        diag = np.arange(self.ns)
        b2[diag, diag, :] += b1.T
        return b2

    def grad_dev_beta(self, beta, S):
        """
        Parameters
        ----------
        beta: array of shape (nx,)
            Model coefficients

        S: array of shape (nx, nx)
            Smoothing penalty matrix


        Returns
        -------
        g: array of shape (nx, )
            Derivative of penalized deviance with respect to beta

        """
        mu = self.f.inv_link(self.X.dot(beta))
        gw = (self.y - mu) / (self.f.var_func(mu=mu) * self.f.dlink(mu))
        g = -self.X.T.dot(gw) + S.dot(beta)
        return g

    def hess_dev_beta(self, beta, S):
        """
        Parameters
        ----------
        beta: array of shape (nx,)
            Model coefficients

        S: array of shape (nx, nx)
            Smoothing penalty matrix


        Returns
        -------
        D2: array of shape (nx, nx)
            Second derivative of deviance with respect to beta

        Dp2: array of shape (nx, nx)
            Second derivative of penalized deviance with respect to beta

        """
        mu = self.f.inv_link(self.X.dot(beta))
        v0, g1 = self.f.var_func(mu=mu), self.f.dlink(mu)
        v1, g2 = self.f.dvar_dmu(mu), self.f.d2link(mu)
        r = self.y - mu
        w = (1.0 + r * (v1 / v0 + g2 / g1)) / (v0 * g1**2)
        D2 = (self.X * w[:, None]).T.dot(self.X)
        Dp2 = D2 + S
        return D2, Dp2

    def reml(self, theta):
        """
        Parameters
        ----------
        theta: array of shape (ns+1,)
            Parameter vector. First ns elements are log smoothing parameters
            and the last is scale

        Returns
        -------
        L: float
            REML criterion

        """
        lam, phi = np.exp(theta[:-1]), np.exp(theta[-1])
        S = self.get_penalty_mat(lam)
        beta, eta, mu, _, _, _ = self.pirls(lam)
        D2, Dp2 = self.hess_dev_beta(beta, S)

        D = self.f.deviance(y=self.y, mu=mu).sum()
        P = beta.T.dot(S).dot(beta)
        _, ldh = np.linalg.slogdet( Dp2 / phi)
        lds = self.logdetS(lam, phi)
        Dp = (D + P) / phi
        K = ldh - lds
        ls = self.f.llscale(phi, self.y) * 2.0
        L = (Dp + K + ls) / 2.0
        return L

    def gradient(self, theta):
        """
        Parameters
        ----------
        theta: array of shape (ns+1,)
            Parameter vector. First ns elements are log smoothing parameters
            and the last is scale

        Returns
        -------
        g: array of shape (ns+1)
            Derivative of REML criterion with respect to theta

        """
        lam, phi = np.exp(theta[:-1]), np.exp(theta[-1])
        S = self.get_penalty_mat(lam)
        X = self.X
        beta, eta, mu, _, _, _ = self.pirls(lam)
        _, Dp2 = self.hess_dev_beta(beta, S)
        A = np.linalg.inv(Dp2)
        dw_deta = self.f.dw_deta(self.y, mu)
        b1 = self.grad_beta_rho(beta, lam)
        eta1 = X.dot(b1)
        bSb = np.einsum('m,imn,n->i', beta, self.S, beta)
        trAS = np.einsum('kj,ijk->i', A, self.S)
        XA = X.dot(A)
        q = (XA * X).sum(axis=1)
        trAH1 = eta1.T.dot(dw_deta * q)
        g = np.zeros_like(theta)
        g[:self.ns] = bSb * lam / phi + lam * trAS + trAH1 - self.ranks

        Dp = self.f.deviance(y=self.y, mu=mu).sum() + beta.T.dot(S).dot(beta)
        ls1 = self.f.dllscale(phi, self.y) * 2.0
        g[-1] = -Dp / phi + ls1 * phi - self.mp
        g /= 2.0
        return g

    def hessian(self, theta):
        """
        Parameters
        ----------
        theta: array of shape (ns+1,)
            Paramete vector. First ns elements are log smoothing parameters
            and the last is scale

        Returns
        -------
        H: array of shape (ns+1, ns+1)
            Second derivative of REML criterion with respect to theta

        """
        lam, phi = np.exp(theta[:-1]), np.exp(theta[-1])
        X, S = self.X, self.get_penalty_mat(lam)
        beta, eta, mu, _, _, _ = self.pirls(lam)
        _, Dp2 = self.hess_dev_beta(beta, S)
        A = np.linalg.inv(Dp2)
        b1 = self.grad_beta_rho(beta, lam)
        b2 = self.hess_beta_rho(beta, lam)
        dw_deta = self.f.dw_deta(self.y, mu)
        d2w_deta2 = self.f.d2w_deta2(self.y, mu)
        D2r = b1.T.dot(Dp2).dot(b1)
        eta1 = X.dot(b1)
        XA = X.dot(A)
        q = (XA * X).sum(axis=1)  # x_n' A x_n

        # tr1[i, j] = trace(AM_i AM_j) where AM_k = A (H1_k + λ_k S_k)
        if self.ns:
            AM = np.stack([A.dot(wcrossp(X, dw_deta * eta1[:, k]) + lam[k] * self.S[k])
                           for k in range(self.ns)])
            tr1 = np.einsum('iab,jba->ij', AM, AM)
        else:
            tr1 = np.zeros((0, 0))

        # trAH2[i, j] = trace(A · H2_ij) computed without forming H2
        p1, p2 = d2w_deta2 * q, dw_deta * q
        Xtp2 = X.T.dot(p2)
        trAH2 = eta1.T.dot(eta1 * p1[:, None]) + np.einsum('ijk,k->ij', b2, Xtp2)
        trAS = np.einsum('kj,ijk->i', A, self.S)
        diag_corr = np.zeros((self.ns, self.ns))
        diag_corr[np.diag_indices(self.ns)] = lam * trAS
        tr2 = trAH2 + diag_corr

        bSb = np.einsum('m,imn,n->i', beta, self.S, beta)
        ldh2 = -(tr1 - tr2)
        H = np.zeros((self.ns + 1, self.ns + 1))
        H[:self.ns, :self.ns] = ldh2 / 2.0 - D2r / phi
        H[np.diag_indices(self.ns)] += lam * bSb / (2.0 * phi)
        H[-1, :self.ns] = -bSb * lam / (2.0 * phi)
        H[:self.ns, -1] = H[-1, :self.ns]

        Dp = self.f.deviance(y=self.y, mu=mu).sum() + beta.T.dot(S).dot(beta)
        ls1, ls2 = self.f.dllscale(phi, self.y), self.f.d2llscale(phi, self.y)
        H[-1, -1] = Dp / (2.0 * phi) + ls1 * phi + ls2 * phi**2
        return H

    def get_smooth_comps(self, smooth_names=None, data=None, beta=None, ci=90):
        """
        Parameters
        ----------
        beta: array of shape (nx,)
            Model coefficients

        ci: int or float, optional
            Confidence level to return intervals with


        Returns
        -------
        f: dict of arrays
            Each key-value pair corresponds to a smooth and an array ranging
            over the values at which the smooth is being evaluated, x,
            the estimated value, f(x), and the standard error.

        """
        beta = self.beta if beta is None else beta
        smooth_names = self.smooths.keys() if smooth_names is None else smooth_names
        z = sp.stats.norm(0, 1).ppf(1.0 - (100 - ci) / 200)
        return {key: np.vstack(self._eval_smooth(key, beta, data, z)).T
                for key in smooth_names}

    def _eval_smooth(self, key, beta, data, z):
        methods = {"cr":crspline_basis, "cc":ccspline_basis,"bs":bspline_basis}
        s = self.smooths[key]
        knots = s['knots']
        x = np.linspace(knots.min(), knots.max(), 200) if data is None else data[key]
        Xs = methods[s['kind']](x, knots, **s['fkws'])
        Xs, _ = absorb_constraints(s['q'], X=Xs)
        y = Xs.dot(beta[s['ix']])
        ix = s['ix'].copy()[:, None]
        Vc = self.Vc[ix, ix.T]
        se = np.sqrt((Xs.dot(Vc) * Xs).sum(axis=1)) * z
        return x, y, se

    def plot_smooth_comp(self, smooth_names=None, data=None, beta=None, single_fig=True,
                         subplot_map=None, ci=95, fig_kws={}):
        """
        Parameters
        ----------
        beta: array of shape (nx,), optional
            Model coefficients.  Defaults to estimated values

        single_fig: bool, optional
            Whether or not to plot smooth components on a single figure

        subplot_map: function, array, or dict, optional
            Maps the ith smooth to a subplot (e.g. subplot_map[0]=[0, 0],
            subplot_map[1]=[0, 1], subplot_map[2]=[1, 0], subplot_map[3]=[1, 1])

        ci: int or float, optional
            Confidence level of interval

        fig_kws: dict, optional
            Figure keyword arguments


        Returns
        -------
        fig: matplotlib object
            Figure object

        ax: matplotlib object
            Axis object

        """
        beta = self.beta if beta is None else beta
        smooth_names = self.smooths.keys() if smooth_names is None else smooth_names
        z = sp.stats.norm(0, 1).ppf(1.0 - (100 - ci) / 200)
        if single_fig:
            fig, ax = plt.subplots(**fig_kws)
        if subplot_map is None:
            subplot_map = dict(zip(np.arange(self.ns), np.arange(self.ns)))
        for i, key in enumerate(smooth_names):
            x, y, se = self._eval_smooth(key, beta, data, z)
            if not single_fig:
                fig, ax = plt.subplots()
                ax.plot(x, y)
                ax.fill_between(x, y-se, y+se, color='b', alpha=0.4)
            else:
                ax[subplot_map[i]].plot(x, y)
                ax[subplot_map[i]].fill_between(x, y-se, y+se, color='b', alpha=0.4)
        return fig, ax

    def optimize_penalty(self, approx_hess=False, opt_kws={}):
        """
        Parameters
        ----------
        approx_hess: bool, optional
            Whether to calculate the hessian or use a numerical approximation.
            Defaults to False (i.e. calculate exact hessian)

        opt_kws: dict, optional
            scipy.optimize.minimize keyword arguments

        """
        if approx_hess:
            hess = lambda x: so_gc_cd(self.gradient, x)
        else:
            hess = self.hessian
        x = self.theta.copy()
        opt = sp.optimize.minimize(self.reml, x, jac=self.gradient,
                                   hess=hess, method='trust-constr',
                                   **opt_kws)
        theta = opt.x.copy()
        rho, logscale = theta[:-1], theta[-1]
        lambda_, scale = np.exp(rho), np.exp(logscale)
        beta, eta, mu, dev, _, _ = self.pirls(lambda_)
        _, w = self.get_wz(eta)
        X, Slambda = self.X, self.get_penalty_mat(lambda_)
        # Convention (b): observed information per φ unit for the data part
        # (Hphi) and full bread K = (X'WX + S)/φ. Vb is the inverse of K
        # directly; the score (when used downstream) carries the matching 1/φ.
        Hbeta = wcrossp(X, w)
        Hphi = Hbeta / scale
        K = Hphi + Slambda / scale
        Vb = np.linalg.inv(K)
        Vp = np.linalg.inv(self.hessian(theta))
        Jb = self.grad_beta_rho(beta, lambda_)
        C = Jb.dot(Vp[:-1, :-1]).dot(Jb.T)
        Vc = Vb + C
        Vs = Vb.dot(Hphi).dot(Vb)
        Vf = Vs + C
        F = Vb.dot(Hphi)
        self.Slambda, self.K = Slambda, K
        self.Vb, self.Vp, self.Vc, self.Vf, self.Vs = Vb, Vp, Vc, Vf, Vs
        self.opt, self.theta, self.scale = opt, theta, scale
        self.beta, self.eta, self.dev, self.mu = beta, eta, dev, mu
        self.F, self.edf, self.Hbeta = F, np.trace(F), Hbeta
        s_table = {}
        for term in self.smooths.keys():

            ix = self.smooths[term]['ix']
            Vb = self.Vb[ix, ix[:, None]]
            X = self.X[:, ix]
            b = self.beta[ix]
            Q, R = np.linalg.qr(X)
            Rb = R.dot(b)
            W = np.linalg.pinv(R.dot(Vb).dot(R.T))
            Tr = Rb.dot(W).dot(Rb)
            edf = np.sum(self.F[ix, ix])
            if (edf - np.floor(edf)) < 0.05:
                r = np.floor(edf)
            else:
                r = np.ceil(edf)
            p = sp.stats.chi2(r).sf(Tr)
            s_table[term] = {"edf":edf, "rdf":r, "chisq":Tr,
                             "p_approx":p}

        self.res_smooths = pd.DataFrame(s_table).T

    def fit(self, approx_hess=False, opt_kws={}, confint=95):
        """
        Parameters
        ----------
        approx_hess: bool, optional
            Whether to calculate the hessian or use a numerical approximation.
            Defaults to False (i.e. calculate exact hessian)

        opt_kws: dict, optional
            scipy.optimize.minimize keyword arguments

        confint: int, float, optional
            Confidence intervals for summary table

        """
        self.optimize_penalty(approx_hess=approx_hess, opt_kws=opt_kws)

        b, se = self.beta, np.sqrt(np.diag(self.Vc))
        b = np.concatenate((b, self.theta))
        se = np.concatenate((se, np.sqrt(np.diag(self.Vp))))

        c = sp.stats.norm(0, 1).ppf(1-(100-confint)/200)
        t = b/se
        p = sp.stats.t(self.edf).sf(np.abs(t))
        res = np.vstack((b, b-c*se, b+c*se, se, t, p)).T
        self.res = pd.DataFrame(res, index=self.varnames,
                                columns=['param', f'CI{confint}-', f'CI{confint}+',
                                         'SE', 't', 'p'])
        ymu = self.y.mean()
        self.null_deviance = self.f.deviance(self.y, mu=np.ones_like(self.y)*ymu).sum()
        self.model_deviance = self.f.deviance(self.y, mu=self.mu).sum()
        self.deviance_explained = (self.null_deviance - self.model_deviance) / self.null_deviance
        self.rsquared = 1.0 - np.sum((self.y - self.mu)**2) / np.sum((self.y - ymu)**2)
        self.ll_model = self.f.full_loglike(self.y, mu=self.mu, phi=self.model_deviance/self.n_obs)
        self.aic = 2.0 * self.ll_model + 2.0 + self.edf * 2.0
        self.sumstats = pd.DataFrame([self.deviance_explained, self.rsquared,
                                      self.ll_model, self.aic, self.edf],
                                     index=['Explained Deviance', 'Rsquared',
                                            'Loglike', 'AIC', 'EDF'])


