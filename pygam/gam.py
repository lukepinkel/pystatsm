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
from .smooth_setup import parse_smooths, get_parametric_formula, get_smooth
from ..pyglm.families import Gaussian, InverseGaussian, Gamma
from ..utilities.splines import (crspline_basis, bspline_basis, ccspline_basis,
                                 absorb_constraints)


def wcrossp(X, w):
    Y =  (X * w.reshape(-1, 1)).T.dot(X)
    return Y

class GAM:
    
    def __init__(self, formula, data, family=None):
        if family is None:
            family = Gaussian()
        smooth_info = parse_smooths(formula, data)
        formula = get_parametric_formula(formula)
        y, Xp = patsy.dmatrices(formula, data, return_type='dataframe', 
                                eval_env=1)
        varnames = Xp.columns.tolist()
        smooths = {}
        start = p = Xp.shape[1]
        ns = 0
        for key, val in smooth_info.items():
            slist = get_smooth(**val)
            if len(slist)==1:
                smooths[key], = slist
                p_i = smooths[key]['X'].shape[1]
                varnames += [f"{key}{j}" for j in range(1, p_i+1)]
                p += p_i
                ns += 1
            else:
                for i, x in enumerate(slist):
                    by_key = f"{key}_{x['by_cat']}"
                    smooths[by_key] = x
                    p_i = x['X'].shape[1]
                    varnames += [f"{by_key}_{j}" for j in range(1, p_i+1)]
                    p += p_i
                    ns += 1
        X, S, Sj, ranks, ldS = [Xp], np.zeros((ns, p, p)), [], [], []
        for i, (var, s) in enumerate(smooths.items()):
            p_i = s['X'].shape[1]
            Si, ix = np.zeros((p, p)), np.arange(start, start+p_i)
            start += p_i
            Si[ix, ix.reshape(-1, 1)] = s['S']
            smooths[var]['ix'], smooths[var]['Si'] = ix, Si
            X.append(smooths[var]['X'])
            S[i] = Si
            Sj.append(s['S'])
            ranks.append(np.linalg.matrix_rank(Si))
            u = np.linalg.eigvals(s['S'])
            ldS.append(np.log(u[u>np.finfo(float).eps]).sum())
        self.X, self.Xp, self.y = np.concatenate(X, axis=1), Xp.values, y.values[:, 0]
        self.S, self.Sj, self.ranks, self.ldS = S, Sj, ranks, ldS
        self.f, self.smooths = family, smooths
        self.ns, self.n_obs, self.nx = ns, self.X.shape[0], self.X.shape[1]
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
        mu = self.f.inv_link(eta)
        v = self.f.var_func(mu=mu)
        dg = self.f.dinv_link(eta)
        r = self.y - mu
        a = 1.0 + r * (self.f.dvar_dmu(mu) / v + self.f.d2link(mu) * dg)
        z = eta + r / (dg * a)
        w = a * dg**2 / v
        return z, w
    
    def solve_pls(self, eta, S):
        z, w = self.get_wz(eta)
        Xw = self.X * w[:, None]
        beta_new = np.linalg.solve(Xw.T.dot(self.X)+S, Xw.T.dot(z))
        return beta_new
        
    def pirls(self, lam, n_iters=200, tol=1e-7):
        beta = np.zeros(self.X.shape[1])
        S = self.get_penalty_mat(lam)
        eta = self.X.dot(beta)
        dev = self.f.deviance(self.y, mu=self.f.inv_link(eta)).sum()
        success = False
        for i in range(n_iters):
            beta_new = self.solve_pls(eta, S)
            eta_new = self.X.dot(beta_new)
            dev_new = self.f.deviance(self.y, mu=self.f.inv_link(eta_new)).sum()
            if dev_new > dev:
                success=False
                break
            if abs(dev - dev_new) / dev_new < tol:
                success = True
                break
            eta = eta_new
            dev = dev_new
            beta = beta_new
        mu = self.f.inv_link(eta)
        return beta, eta, mu, dev, success, i

    def get_penalty_mat(self, lam):
        Sa = np.einsum('i,ijk->jk', lam, self.S)
        return Sa
    
    def logdetS(self, lam, phi):
        logdet = 0.0
        for i, (r, lds) in enumerate(list(zip(self.ranks, self.ldS))):
            logdet += r * np.log(lam[i]/phi) + lds
        return logdet
    
    def grad_beta_rho(self, beta, lam):
        S = self.get_penalty_mat(lam)
        Dp, Dp2 = self.hess_dev_beta(beta, S)
        A = np.linalg.inv(2.0 * Dp2)
        dbdr = np.zeros((beta.shape[0], lam.shape[0]))
        for i in range(self.ns):
            Si, ai = self.S[i], lam[i]
            dbdr[:, i] = -ai * A.dot(Si.dot(beta))*2.0
        return dbdr
    
    def hess_beta_rho(self, beta, lam):
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
                fij = 0.5 * eta1j * eta1i * w1
                u = self.X.T.dot(fij) + ai * Si.dot(b1j) + aj * Sj.dot(b1i)
                b2[i, j] = b2[j, i] = (i==j)*b1[:, j] - A.dot(u)
        return b2
                
        
    def hess_dev_beta(self, beta, S):
        mu = self.f.inv_link(self.X.dot(beta))
        v0, g1 = self.f.var_func(mu=mu), self.f.dlink(mu)
        v1, g2 = self.f.dvar_dmu(mu), self.f.d2link(mu)
        r = self.y - mu
        w = (1.0 + r * (v1 / v0 + g2 / g1)) / (v0 * g1**2)
        D2 = (self.X * w[:, None]).T.dot(self.X) 
        Dp2 = D2 + S
        return D2, Dp2
    
    def reml(self, theta):
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
    
    def _hessian(self, theta):
        lam, phi = np.exp(theta[:-1]), np.exp(theta[-1])
        S = self.get_penalty_mat(lam)
        beta, eta, mu, _, _, _ = self.pirls(lam)
        D2, Dp2 = self.hess_dev_beta(beta, S)
        A = np.linalg.inv(Dp2)
        db = self.grad_beta_rho(beta, lam)

        H = np.zeros((self.ns+1, self.ns+1))
        for i in range(self.ns):
            Si, ai = self.S[i], lam[i]
            Sib = Si.dot(beta)
            for j in range(i, self.ns):
                krd = (i==j)
                Sj, aj, dbj = self.S[j], lam[j], db[:, j]
                t1 = krd * np.dot(beta.T, Sib) * ai / (2*phi)
                
                t2 = ai / (2*phi) * (dbj.T.dot(Sib) + Sib.T.dot(dbj))
                t3 = -ai * aj /2.0 * np.trace(A.dot(Si).dot(A).dot(Sj))
                t4 = krd* ai / 2.0 * np.trace(A.dot(Si))
                #t5 = -1.0 / phi * db[:, i].T.dot(V).dot(db[:, j])
                H[i, j] = H[j, i] = t1+t2+t3+t4#+t5
                if krd:
                    H[-1, j] = H[j, -1] = -t1
                    
        Dp = self.f.deviance(y=self.y, mu=mu).sum() + beta.T.dot(S).dot(beta)
        ls1, ls2 = self.f.dllscale(phi, self.y), self.f.d2llscale(phi, self.y)
        
        H[-1, -1] = Dp / (2.0 * phi) - 2.0 * ls2*phi**2 - 2.0 * ls1*phi
        return H
    
    def hessian(self, theta):
        lam, phi = np.exp(theta[:-1]), np.exp(theta[-1])
        S = self.get_penalty_mat(lam)
        X = self.X
        beta, eta, mu, _, _, _ = self.pirls(lam)
        D2, Dp2 = self.hess_dev_beta(beta, S)
        A = np.linalg.inv(Dp2)
        b1 = self.grad_beta_rho(beta, lam)
        b2 = self.hess_beta_rho(beta, lam)
        dw_deta = self.f.dw_deta(self.y, mu)
        d2w_deta2 = self.f.d2w_deta2(self.y, mu)
        #Ddb = X.T.dot((self.y - mu) / (self.f.var_func(mu=mu) * self.f.dlink(mu)))
        #D2r = b1.T.dot(D2).dot(b1) + np.einsum('ijk,k->ij', b2, Ddb) 
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
                 # bsb2 = b2[i, j].dot(S).dot(beta) + b1i.dot(Sj).dot(beta) * aj\
                 #      + b1j.T.dot(Si).dot(beta) * ai + b1i.T.dot(S).dot(b1j)\
                 #      + d * ai * beta.T.dot(Si).dot(beta)
                 #t1 = np.trace(A.dot(H2))
                 #t2 = d * ai * np.trace(A.dot(Si))
                 #t3 = -np.trace(A.dot(H1i).dot(A).dot(H1j))
                 #t4 = -np.trace(A.dot(H1i).dot(A).dot(Sj)) * aj
                 #t5 = -np.trace(A.dot(H1j).dot(A).dot(Si)) * ai
                 #t6 = -np.trace(A.dot(Si).dot(A).dot(Sj)) * ai * aj
                 #ldh2 = t1 + t2 + t3 + t4 + t5 + t6
                 ldh2 = -(np.trace(A.dot(H1i+Si*ai).dot(A).dot(H1j+aj*Sj))\
                          -np.trace(A.dot(H2+d*ai*Si)))
                 #H[i, j] = H[j, i] = ldh2 + (bsb2 + D2r[i, j]) / (2.0 * phi)
                 t1 = d * ai / (2.0 * phi) * beta.T.dot(Si).dot(beta)
                 t2 = -D2r[i, j] / (phi)
                 H[i, j] = H[j, i] = t1 + t2 + ldh2/2
                 if d:
                     H[-1, j] = H[j, -1] = -np.dot(beta.T, Si.dot(beta)) * ai / (2*phi)
    

        Dp = self.f.deviance(y=self.y, mu=mu).sum() + beta.T.dot(S).dot(beta)
        ls1, ls2 = self.f.dllscale(phi, self.y), self.f.d2llscale(phi, self.y)
        
        H[-1, -1] = Dp / (2.0 * phi) - 2.0 * ls2*phi**2 - 2.0 * ls1*phi
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
    
    def optimize_penalty(self, opt_kws={}):
        x = self.theta.copy()
        opt = sp.optimize.minimize(self.reml, x, jac=self.gradient, 
                                   hess=self.hessian, method='trust-constr',
                                   **opt_kws)
        theta = opt.x.copy()
        rho, logscale = theta[:-1], theta[-1]
        lambda_, scale = np.exp(rho), np.exp(logscale)
        beta, eta, mu, dev, _, _ = self.pirls(lambda_)
        X, Slambda = self.X, self.get_penalty_mat(lambda_)
        XtX = X.T.dot(X)
        Vb = np.linalg.inv(XtX + Slambda) * scale
        Vp = np.linalg.inv(self.hessian(theta))
        Jb = self.grad_beta_rho(beta, lambda_)
        C = Jb.dot(Vp[:-1, :-1]).dot(Jb.T)
        Vc = Vb + C
        Vf = Vb.dot(XtX*scale).dot(Vb) + C
        self.Slambda = Slambda
        self.Vb, self.Vp, self.Vc, self.Vf = Vb, Vp, Vc, Vf
        self.opt, self.theta, self.scale = opt, theta, scale
        self.beta, self.eta, self.dev = beta, eta, dev
        self.edf = np.trace(np.linalg.inv(Vb*XtX/scale))
        
    def fit(self, opt_kws={}, confint=95):
        self.optimize_penalty(opt_kws=opt_kws)

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
        
        
        
        
    
        