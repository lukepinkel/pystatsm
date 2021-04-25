# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 17:01:11 2021

@author: lukepinkel
"""

import patsy
import numpy as np
import scipy as sp
import scipy.stats
import pandas as pd
import scipy.interpolate
import matplotlib.pyplot as plt
from .smooth_setup import (parse_smooths, get_parametric_formula, 
                           get_smooth_terms, get_smooth_matrices)
from ..pyglm.families import Gaussian, InverseGaussian, Gamma
from ..utilities.splines import (crspline_basis, bspline_basis, ccspline_basis,
                                 absorb_constraints)

from ..utilities.numerical_derivs import so_gc_cd

def wcrossp(X, w):
    Y =  (X * w.reshape(-1, 1)).T.dot(X)
    return Y


class GAM:
    
    def __init__(self, formula, data, family=None):
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
        self.S, self.ranks, self.ldS = S, ranks, ldS
        self.f, self.smooths = family, smooths
        self.ns, self.n_obs, self.nx = n_smooth_terms, Xp.shape[0], n_total_params
        self.mp = self.nx - np.sum(self.ranks)
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
        beta_new = np.linalg.solve(Xw.T.dot(self.X)+S, Xw.T.dot(z))
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
        logdet = 0.0
        for i, (r, lds) in enumerate(list(zip(self.ranks, self.ldS))):
            logdet += r * np.log(lam[i]/phi) + lds
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
        Dp, Dp2 = self.hess_dev_beta(beta, S)
        A = np.linalg.inv(2.0 * Dp2)
        dbdr = np.zeros((beta.shape[0], lam.shape[0]))
        for i in range(self.ns):
            Si, ai = self.S[i], lam[i]
            dbdr[:, i] = -ai * A.dot(Si.dot(beta))*2.0
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
        S, b1 = self.get_penalty_mat(lam), self.grad_beta_rho(beta, lam)
        _, Dp2 = self.hess_dev_beta(beta, S)
        A = np.linalg.inv(Dp2)
        mu = self.f.inv_link(self.X.dot(beta))
        b2 = np.zeros((self.ns, self.ns, beta.shape[0]))
        for i in range(self.ns):
            Si, ai, b1i = self.S[i], lam[i], b1[:, i]
            eta1i = self.X.dot(b1i)
            for j in range(i, self.ns):
                Sj, aj, b1j = self.S[j], lam[j], b1[:, j]
                eta1j = self.X.dot(b1j)
                w1 = self.f.dw_deta(self.y, mu)
                fij = 1.0 * eta1j * eta1i * w1
                u = self.X.T.dot(fij) + ai * Si.dot(b1j) + aj * Sj.dot(b1i)
                b2[i, j] = b2[j, i] = (i==j)*b1[:, j] - A.dot(u)
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
            Paramete vector. First ns elements are log smoothing parameters
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
            Paramete vector. First ns elements are log smoothing parameters
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
        D2, Dp2 = self.hess_dev_beta(beta, S)
        A = np.linalg.inv(Dp2)
        dw_deta = self.f.dw_deta(self.y, mu)
        b1 = self.grad_beta_rho(beta, lam)
        g = np.zeros_like(theta)
        for i in range(self.ns):
            Si, ai, b1i = self.S[i], lam[i], b1[:, i]
            w1i = (dw_deta * X.dot(b1i)).reshape(-1, 1)
            H1 = (X * w1i).T.dot(X)
            dbsb = beta.T.dot(Si).dot(beta) * ai / phi
            dldh = np.trace(A.dot(Si*ai + H1))
            dlds = self.ranks[i]
            g[i] = dbsb + dldh - dlds
        
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
        D2, Dp2 = self.hess_dev_beta(beta, S)
        A = np.linalg.inv(Dp2)
        b1, b2 = self.grad_beta_rho(beta, lam), self.hess_beta_rho(beta, lam)
        dw_deta, d2w_deta2 = self.f.dw_deta(self.y, mu), self.f.d2w_deta2(self.y, mu)
        D2r =  b1.T.dot(Dp2).dot(b1)
        H = np.zeros((self.ns+1, self.ns+1))
        for i in range(self.ns):
            Si, ai , b1i = self.S[i], lam[i], b1[:, i]
            eta1i = X.dot(b1i)
            w1i = dw_deta * eta1i
            H1i = wcrossp(X, w1i)
            for j in range(i, self.ns):
                 Sj, aj, b1j, eta2 = self.S[j], lam[j], b1[:, j], X.dot(b2[i, j])
                 eta1j = self.X.dot(b1j)
                 w1j = dw_deta * eta1j
                 w2 = eta1j * eta1i * d2w_deta2 + dw_deta * eta2
                 H1j = wcrossp(X, w1j)
                 H2 = wcrossp(X, w2)
                 d = (i==j)
                 ldh2 = -(np.trace(A.dot(H1i+Si*ai).dot(A).dot(H1j+aj*Sj))\
                          -np.trace(A.dot(H2+d*ai*Si)))
                 t1 = d * ai / (2.0 * phi) * beta.T.dot(Si).dot(beta)
                 t2 = -D2r[i, j] / (phi)
                 H[i, j] = H[j, i] = t1 + t2 + ldh2/2
                 if d:
                     H[-1, j] = H[j, -1] = -np.dot(beta.T, Si.dot(beta)) * ai / (2*phi)
    
        Dp = self.f.deviance(y=self.y, mu=mu).sum() + beta.T.dot(S).dot(beta)
        ls1, ls2 = self.f.dllscale(phi, self.y), self.f.d2llscale(phi, self.y)
        
        H[-1, -1] = Dp / (2.0 * phi) + ls1*phi  + ls2*phi**2 
        return H
    
    def get_smooth_comps(self, beta, ci=90):
        methods = {"cr":crspline_basis, "cc":ccspline_basis,"bs":bspline_basis} 
        f = {}
        ci = sp.stats.norm(0, 1).ppf(1.0 - (100 - ci) / 200)
        for i, (key, s) in enumerate(self.smooths.items()):
            knots = s['knots']         
            x = np.linspace( knots.min(),  knots.max(), 200)
            X = methods[s['kind']](x, knots, **s['fkws'])
            X, _ = absorb_constraints(s['q'], X=X)
            y = X.dot(beta[s['ix']])
            ix = s['ix'].copy()[:, None]
            Vc = self.Vc[ix, ix.T]
            se = np.sqrt(np.diag(X.dot(Vc).dot(X.T))) * ci
            f[key] = np.vstack((x, y, se)).T
        return f
            
    def plot_smooth_comp(self, beta, single_fig=True, subplot_map=None, 
                         ci=95, fig_kws={}):
        ci = sp.stats.norm(0, 1).ppf(1.0 - (100 - ci) / 200)
        methods = {"cr":crspline_basis, "cc":ccspline_basis,"bs":bspline_basis} 
        if single_fig:
            fig, ax = plt.subplots(**fig_kws)
            
        if subplot_map is None:
            subplot_map = dict(zip(np.arange(self.ns), np.arange(self.ns)))
        for i, (key, s) in enumerate(self.smooths.items()):
            knots = s['knots']         
            x = np.linspace( knots.min(),  knots.max(), 200)
            X = methods[s['kind']](x, knots, **s['fkws'])
            X, _ = absorb_constraints(s['q'], X=X)
            y = X.dot(beta[s['ix']])
            ix = s['ix'].copy()[:, None]
            Vc = self.Vc[ix, ix.T]
            se = np.sqrt(np.diag(X.dot(Vc).dot(X.T))) * ci
            if not single_fig: 
                fig, ax = plt.subplots()
                ax.plot(x, y)
                ax.fill_between(x, y-se, y+se, color='b', alpha=0.4)
            else:
                ax[subplot_map[i]].plot(x, y)
                ax[subplot_map[i]].fill_between(x, y-se, y+se, color='b', alpha=0.4)
        return fig, ax
    
    def optimize_penalty(self, approx_hess=False, opt_kws={}):
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
        Hbeta = wcrossp(X, w)
        Vb = np.linalg.inv(Hbeta + Slambda) * scale
        Vp = np.linalg.inv(self.hessian(theta))
        Jb = self.grad_beta_rho(beta, lambda_)
        C = Jb.dot(Vp[:-1, :-1]).dot(Jb.T)
        Vc = Vb + C
        Vs = Vb.dot(Hbeta/scale).dot(Vb)
        Vf = Vs + C
        F = Vb.dot(Hbeta / scale)
        self.Slambda = Slambda
        self.Vb, self.Vp, self.Vc, self.Vf, self.Vs = Vb, Vp, Vc, Vf, Vs
        self.opt, self.theta, self.scale = opt, theta, scale
        self.beta, self.eta, self.dev, self.mu = beta, eta, dev, mu
        self.F, self.edf, self.Hbeta = F, np.trace(F), Hbeta
        
    def fit(self, approx_hess=False, opt_kws={}, confint=95):
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
        self.ll_model = self.f.full_loglike(self.y, mu=self.mu, scale=self.model_deviance/self.n_obs)
        self.aic = 2.0 * self.ll_model + 2.0 + self.edf * 2.0
        self.sumstats = pd.DataFrame([self.deviance_explained, self.rsquared,
                                      self.ll_model, self.aic, self.edf], 
                                     index=['Explained Deviance', 'Rsquared',
                                            'Loglike', 'AIC', 'EDF'])                                  
    
    
        