# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 12:58:08 2021

@author: lukepinkel
"""

import patsy
import numpy as np
import scipy as sp
import scipy.stats
import pandas as pd
import scipy.optimize
import matplotlib.pyplot as plt
from ..pyglm.links import IdentityLink, Link
from .smooth_setup import parse_smooths, get_parametric_formula,  get_smooth_terms, get_smooth_matrices
from ..utilities.splines import crspline_basis, bspline_basis, ccspline_basis, absorb_constraints
from ..utilities.numerical_derivs import so_gc_cd



def wcrossp(X, w):
    Y =  (X * w.reshape(-1, 1)).T.dot(X)
    return Y


class LogbLink(Link):
    
    def __init__(self, b=0.01):
        self.b = 0.01
    
    def link(self, mu):
        return np.log(1.0 / mu - self.b)
    
    def inv_link(self, eta):
        return 1.0 / (np.exp(eta) + self.b)
    
    def dinv_link(self, eta):
        u = np.exp(eta)
        return -u / (u + self.b)**2
    
    def d2link(self, mu):
        u = 1.0 - mu * self.b
        d2mu = (2.0 * u - 1.0) / (mu * u)**2
        return d2mu
    
    def d3link(self, mu):
        u = 1.0 - mu * self.b
        d3mu = ((1.0 - u) * u * 6.0 - 2.0) / (mu * u)**3
        return d3mu
    
    def d4link(self, mu):
        u = 1.0 - mu * self.b
        d4mu = (((24.0 * u - 36.0) * u + 24.0) * u - 6.0) / (mu * u)**4
        return d4mu
    
        

class PredictorTerm:
    
    def __init__(self, formula, data, link):
        if not isinstance(link, Link):
            link = link()
        smooth_info = parse_smooths(formula, data)
        formula = get_parametric_formula(formula)
        y, Xp = patsy.dmatrices(formula, data, return_type='dataframe', eval_env=1)
        smooths, n_smooth_terms, n_total_params, varnames = get_smooth_terms(
                                                                smooth_info, Xp)
        X, S, ranks, ldS = get_smooth_matrices(Xp, smooths, n_smooth_terms,
                                               n_total_params)
        self.X, self.Xp, self.y = np.concatenate(X, axis=1), Xp.values, y.values[:, 0]
        self.S, self.ranks, self.ldS = S, ranks, ldS
        self.smooths = smooths
        self.ns, self.n_obs, self.nx = n_smooth_terms, Xp.shape[0], n_total_params
        self.mp = self.nx - np.sum(self.ranks)
        self.data = data
        theta = np.zeros(self.ns)
        self.x_varnames = varnames
        self.t_varnames = []
        for i, (var, s) in enumerate(smooths.items()):
            ix = smooths[var]['ix']
            a = self.S[i][ix, ix[:, None].T]
            d = np.diag(self.X[:, ix].T.dot(self.X[:, ix]))
            lam = (1.5 * (d / a)[a>0]).mean()
            theta[i] = np.log(lam)
            self.t_varnames += [f"log_smooth_{var}"]
        self.theta = theta
        self.smooth_info = smooth_info
        self.link = link
        
class GauLS:
    
    def __init__(self, m_formula, s_formula, data, m_link=IdentityLink,
                 s_link=LogbLink):
        
        self.m = PredictorTerm(m_formula, data, m_link)
        self.s = PredictorTerm(s_formula, data, s_link)
        self.ns_m = len(self.m.theta)
        self.ns_s = len(self.s.theta)
        self.y = self.m.y
        self.ixm = np.arange(0, self.m.X.shape[1])
        self.ixs = np.arange(self.m.X.shape[1], 
                             self.m.X.shape[1]+self.s.X.shape[1])
        self.nxm = self.m.X.shape[1]
        self.nxs = self.s.X.shape[1]
        self.nx = self.nxm+self.nxs
        self.ns = self.ns_m + self.ns_s
        Sm, Ss = self.m.S.copy(), self.s.S.copy()
        p, q = Sm.shape[1], Ss.shape[1]
        Sm = np.pad(Sm, ((0, 0), (0, q), (0, q)))
        Ss = np.pad(Ss, ((0, 0), (p, 0), (p, 0)))
        key = np.arange(self.ns)
        val = np.zeros(self.ns)
        val[:self.ns_m] = 0
        val[self.ns_m:] = 1 
        hix = np.zeros((2, 2, 2)).astype(int)
        hix[0, 0] = np.array([0, 1])
        hix[0, 1] = np.array([1, 2])
        hix[1, 1] = np.array([2, 3])
        self.hix = hix
        self.xix = dict(zip(key, val.astype(int)))
        self.X = {0:self.m.X, 1:self.s.X}
        self.S = np.concatenate([Sm, Ss], axis=0)
        self.Xt = np.concatenate([self.m.X, self.s.X], axis=1)
        self.mp = self.m.mp + self.s.mp
        self.ranks = self.m.ranks + self.s.ranks
        self.theta = np.ones(self.ns)
        x_varnames = [f"m({x})" for x in self.m.x_varnames] + \
                     [f"s({x})" for x in self.s.x_varnames]
        t_varnames = [f"m({x})" for x in self.m.t_varnames] + \
                     [f"s({x})" for x in self.s.t_varnames]
        self.varnames = x_varnames + t_varnames
        
    def get_mutau(self, beta_m, beta_s):
        etam, etas = self.m.X.dot(beta_m), self.s.X.dot(beta_s)
        mu, tau = self.m.link.inv_link(etam), self.s.link.inv_link(etas)
        return mu, tau
    
    def loglike(self, beta):
        beta_m, beta_s = beta[self.ixm], beta[self.ixs]
        mu, tau = self.get_mutau(beta_m, beta_s)
        r = self.y - mu
        r2, t2 = r**2, tau**2
        ll_elementwise = -0.5 * r2 * t2 - 0.5 * np.log(2.0*np.pi) + np.log(tau)
        ll = np.sum(ll_elementwise)
        return -ll
    
    def penalized_loglike(self, beta, S):
        ll = self.loglike(beta)
        llp = ll + beta.T.dot(S).dot(beta) / 2.0
        return llp
        
    def d1loglike(self, beta_m, beta_s):
        mu, tau = self.get_mutau(beta_m, beta_s)
        r = self.y - mu
        r2, t2 = r**2, tau**2
        dm = t2 * r
        ds = 1/tau - tau * r2
        L1 = np.vstack((dm, ds)).T
        return L1

    def d2loglike(self, beta_m, beta_s):
        mu, tau = self.get_mutau(beta_m, beta_s)
        r = self.y - mu
        r2, t2 = r**2, tau**2
        
        dmm = -t2
        dms = 2.0 * tau * r
        dss = -r2 - 1.0 / t2
        L2 = np.vstack((dmm, dms, dss)).T
        return L2        
    
    def d3loglike(self, beta_m, beta_s):
        mu, tau = self.get_mutau(beta_m, beta_s)
        r = self.y - mu
        t2 = tau**2
        
        dmmm = np.zeros_like(r)
        dmms = -2.0 * tau
        dmss = 2.0 * r
        dsss = 2.0 / (t2 * tau)
        
        L3 = np.vstack((dmmm, dmms, dmss, dsss)).T
        return L3
    
    def d4loglike(self, beta_m, beta_s):
        mu, tau = self.get_mutau(beta_m, beta_s)
        r = self.y - mu
        t2 = tau**2
        
        dmmmm = np.zeros_like(r)
        dmmms = np.zeros_like(r)
        dmmss = np.zeros_like(r) + -2.0
        dmsss = np.zeros_like(r)
        dssss = -6.0 / (t2**2)
        L4 = np.vstack((dmmmm, dmmms, dmmss, dmsss, dssss)).T
        return L4
    
    def ll_eta_derivs(self, beta_m, beta_s):
        etam, etas = self.m.X.dot(beta_m), self.s.X.dot(beta_s)
        mu, tau = self.m.link.inv_link(etam), self.s.link.inv_link(etas)
        r = self.y - mu
        r2, t2 = r**2, tau**2
        
        lm = t2 * r
        ls = 1/tau - tau * r2

         
        lmm = -t2
        lms = 2.0 * tau * r
        lss = -r2 - 1.0 / t2

        lmmm = np.zeros_like(r)
        lmms = -2.0 * tau
        lmss = 2.0 * r
        lsss = 2.0 / (t2 * tau)
        
        lmmmm = np.zeros_like(r)
        lmmms = np.zeros_like(r)
        lmmss = np.zeros_like(r) + -2.0
        lmsss = np.zeros_like(r)
        lssss = -6.0 / (t2**2)
    


        g1m, g1s = self.m.link.dinv_link(etam), self.s.link.dinv_link(etas)
        g2m, g2s = self.m.link.d2link(mu), self.s.link.d2link(tau)
        g3m, g3s = self.m.link.d3link(mu), self.s.link.d3link(tau)
        g4m, g4s = self.m.link.d4link(mu), self.s.link.d4link(tau)
        
        dm = lm * g1m
        ds = ls * g1s
        
        dmm = (lmm - lm * g2m * g1m) * g1m**2
        dms = lms * g1s * g1m
        dss = (lss - ls * g2s * g1s) * g1s**2
        
        dmmm = (lmmm - 3.0 * lmm * g2m * g1m +\
                lm * (3.0 * g2m**2 * g1m**2 - g3m * g1m)) * g1m**3
        dmms = (lmms - lms * g2m * g1m) * g1s * g1m**2
        dmss = (lmss - lms * g2s * g1s) * g1m * g1s**2
        dsss = (lsss - 3.0 * lss * g2s * g1s +\
                ls * (3.0 * g2s**2 * g1s**2 - g3s * g1s)) * g1s**3
        
        dmmmm = (lmmmm - 6.0 * lmmm * g2m * g1m +\
                 lmm * (15 * g2m**2 * g1m**2 - 4 * g3m * g1m)-\
                 lm * (15 * g2m**3 * g1m**3 - 10*g2m*g3m*g1m**2\
                       +g4m*g1m))*g1m**4
        dmmms = (lmmms - 3.0 * lmms * g2m * g1m +\
                 lms * (3 * g2m**2 * g1m**2 - g3m*g1m)) * g1s * g1m**3
        dmmss = (lmmss - lmss * g2m * g1m - lmms * g2s * g1s +\
                 lms * g2m * g2s * g1m * g1s) * g1m**2 * g1s**2
        dmsss = (lmsss - 3.0 * lmss * g2s * g1s +\
                 lms * (3 * g2s**2 * g1s**2 - g3s*g1s)) * g1m * g1s**3
        dssss = (lssss - 6.0 * lsss * g2s * g1s +\
                 lss * (15 *g2s**2 * g1s**2 - 4 * g3s * g1s)-\
                 ls * (15 * g2s**3 * g1s**3 - 10 * g2s * g3s * g1s**2\
                       +g4s * g1s)) * g1s**4
        
        L1 = np.vstack((dm, ds)).T
        L2 = np.vstack((dmm, dms, dss)).T
        L3 = np.vstack((dmmm, dmms, dmss, dsss)).T
        L4 = np.vstack((dmmmm, dmmms, dmmss, dmsss, dssss)).T
        return L1, L2, L3, L4
    
    def grad_ll_beta(self, beta):
        beta_m, beta_s = beta[self.ixm], beta[self.ixs]
        L1, _, _, _ = self.ll_eta_derivs(beta_m, beta_s)
        g1 = self.m.X.T.dot(L1[:, 0])
        g2 = self.s.X.T.dot(L1[:, 1])
        g = np.concatenate([g1, g2])
        return -g
    
    def grad_pll_beta(self, beta, S):
        return self.grad_ll_beta(beta)+S.dot(beta)
    
    
    def hess_ll_beta(self, beta):
        beta_m, beta_s = beta[self.ixm], beta[self.ixs]
        _, L2, _, _  = self.ll_eta_derivs(beta_m, beta_s)
        wmm, wms, wss = L2[:, 0], L2[:, 1], L2[:, 2]
        wmm, wms, wss = wmm.reshape(-1, 1), wms.reshape(-1, 1), wss.reshape(-1, 1)
        Xm, Xs = self.m.X, self.s.X
        H11 = (Xm * wmm).T.dot(Xm)
        H12 = (Xm * wms).T.dot(Xs)
        H22 = (Xs * wss).T.dot(Xs)
        H1 = np.concatenate([H11, H12], axis=1)
        H2 = np.concatenate([H12.T, H22], axis=1)
        H = np.concatenate([H1, H2], axis=0)
        return -H
    
    def hess_pll_beta(self, beta, S):
        return self.hess_ll_beta(beta)+S
    
    def get_penalty_mat(self, lam):
        Sa = np.einsum('i,ijk->jk', lam, self.S)
        return Sa
    
    def outer_step(self, S, beta_init=None, n_iters=200, tol=1e-25):
        if beta_init is None:
            beta_init = np.zeros(self.nx)
        beta = beta_init.copy()
        ll_prev = self.loglike(beta)
        convergence = False
        etol = np.finfo(float).eps**(1/3)
        for i in range(n_iters):
            H, g = self.hess_ll_beta(beta)+S, self.grad_ll_beta(beta)+S.dot(beta)
            u, V = np.linalg.eigh(H)
            if (u<etol).any():
                u[u<etol] = abs(u[u<etol])+etol
            H = V.dot(np.diag(u)).dot(V.T)
            d = np.linalg.solve(H, g)
            ll_curr = self.penalized_loglike(beta - d, S)
            if (abs(ll_curr - ll_prev) / (abs(ll_prev)+1e-6)) < tol:
                beta = beta - d
                convergence = True
                break
            elif ll_curr > ll_prev:
                j = 0
                while ((j < 15)&(ll_curr > ll_prev)):
                    d = d / 2.0
                    ll_curr = self.penalized_loglike(beta - d, S)
                    j+=1
                if j==15:
                    convergence = False
                    break
            beta = beta - d
            ll_prev = self.penalized_loglike(beta, S)
        return beta, i, convergence
    
    def beta_rho(self, rho):
        lam = np.exp(rho)
        S = self.get_penalty_mat(lam)
        beta, i, convergence = self.outer_step(S)
        return beta
    
    def grad_beta_rho(self, beta, lam):
        S = self.get_penalty_mat(lam)
        H = self.hess_ll_beta(beta)
        Hp = np.linalg.inv(H + S)
        dbdr = np.zeros((beta.shape[0], lam.shape[0]))
        for i in range(self.ns):
            Si, ai = self.S[i], lam[i]
            dbdr[:, i] = -ai * Hp.dot(Si.dot(beta))
        return dbdr
    
    
    def dhess(self, beta, lam):
        b1 = self.grad_beta_rho(beta, lam)
        L1, L2, L3, L4  = self.ll_eta_derivs(beta[self.ixm], beta[self.ixs])
        Xm, Xs, ixm, ixs = self.m.X, self.s.X, self.ixm, self.ixs
        etam1, etas1 = Xm.dot(b1[ixm]), Xs.dot(b1[ixs])
        dH = np.zeros((self.ns, self.nx, self.nx))
        for i in range(self.ns):
            etam1i, etas1i = etam1[:, i], etas1[:, i]
            v1 = (L3[:, 0] * etam1i + L3[:, 1] * etas1i).reshape(-1, 1)
            v2 = (L3[:, 1] * etam1i + L3[:, 2] * etas1i).reshape(-1, 1)
            v3 = (L3[:, 2] * etam1i + L3[:, 3] * etas1i).reshape(-1, 1)
            #v = v1 + v2 + v3
            dHmm = (Xm * v1).T.dot(Xm)
            dHms = (Xm * v2).T.dot(Xs)
            dHss = (Xs * v3).T.dot(Xs)
            dH[i] = np.block([[dHmm, dHms], [dHms.T, dHss]])
        return -dH
    
    def d2hess(self, beta, lam):
        b1 = self.grad_beta_rho(beta, lam)
        b2 = self.hess_beta_rho(beta, lam)
        L1, L2, L3, L4  = self.ll_eta_derivs(beta[self.ixm], beta[self.ixs])
        Xm, Xs, ixm, ixs = self.m.X, self.s.X, self.ixm, self.ixs
        etam1, etas1 = Xm.dot(b1[ixm]), Xs.dot(b1[ixs])
        etam2 = np.einsum("ij,jkl->ikl", Xm, b2.T[ixm])
        etas2 = np.einsum("ij,jkl->ikl", Xs, b2.T[ixs])
        d2H = np.zeros((self.ns, self.ns, self.nx, self.nx))
        L3, L4 = L3.T, L4.T
        for i in range(self.ns):
            mi, si = etam1[:, i], etas1[:, i]
            for j in range(i, self.ns):
                mj, sj = etam1[:, j], etas1[:, j]
                mij, sij = etam2[:, i, j], etas2[:, i, j]
                
                v1 = L4[0] * mi * mj + L4[1] * mi * sj + L4[1] * si * mj + \
                     L4[2] * si * sj + L3[0] * mij + L3[1] * sij
                v2 = L4[1] * mi * mj + L4[2] * mi * sj + L4[2] * si * mj + \
                     L4[3] * si * sj + L3[1] * mij + L3[2] * sij
                v3 = L4[2] * mi * mj + L4[3] * mi * sj + L4[3] * si * mj + \
                     L4[4] * si * sj + L3[2] * mij + L3[3] * sij
                
                v1, v2, v3 = v1.reshape(-1, 1), v2.reshape(-1, 1), v3.reshape(-1, 1)
                dHmm = (Xm * v1).T.dot(Xm)
                dHms = (Xm * v2).T.dot(Xs)
                dHss = (Xs * v3).T.dot(Xs)
                d2H[i, j] = d2H[j, i] = np.block([[dHmm, dHms], [dHms.T, dHss]])
        return -d2H
    
    
    def hess_beta_rho(self, beta, lam):
        S, b1 = self.get_penalty_mat(lam), self.grad_beta_rho(beta, lam)
        H = self.hess_ll_beta(beta)
        Hp = np.linalg.inv(H + S)
        dH = self.dhess(beta, lam)
        b2 = np.zeros((self.ns, self.ns, beta.shape[0]))
        for i in range(self.ns):
            Si, b1i, ai = self.S[i], b1[:, i], lam[i]
            for j in range(i, self.ns):
                b1j = b1[:, j]
                Sj, aj = self.S[j], lam[j]
                u = dH[i].dot(b1[:, j]) + ai * Si.dot(b1j) + aj * Sj.dot(b1i)
                b2[i, j] = b2[j, i] = (i==j)*b1i - Hp.dot(u)
        return b2
    
    def _grad_beta_rho(self, rho):
        lam = np.exp(rho)
        beta = self.beta_rho(rho)
        return self.grad_beta_rho(beta, lam)
    
    def logdetS(self, rho):
        logdet = 0.0
        for i, (r, lds) in enumerate(list(zip(self.m.ranks, self.m.ldS))):
            logdet += r * rho[i] + lds
        for j, (r, lds) in enumerate(list(zip(self.s.ranks, self.s.ldS))):
            logdet += r * rho[j+i+1] + lds    
        return logdet
    
    def reml(self, rho):
        lam = np.exp(rho)
        beta = self.beta_rho(rho)
        S = self.get_penalty_mat(lam)
        H = self.hess_ll_beta(beta)
        ldetS = self.logdetS(rho)
        _, ldetH = np.linalg.slogdet(H + S)
        ll = self.penalized_loglike(beta, S)
        L = -ll + ldetS / 2.0 - ldetH / 2.0 + self.mp * np.log(2.0*np.pi) / 2.0
        return -L
    
    def gradient(self, rho):
        lam = np.exp(rho)
        beta, S = self.beta_rho(rho), self.get_penalty_mat(lam)
        H = self.hess_ll_beta(beta)
        dH = self.dhess(beta, lam)
        A = np.linalg.inv(H + S)
        bsb = np.zeros_like(rho)
        ldh = np.zeros_like(rho)
        lds = np.zeros_like(rho)
        for i in range(self.ns):
            Si, ai = self.S[i], lam[i]
            dbsb = beta.T.dot(Si).dot(beta) * ai
            dldh = np.trace(A.dot(Si*ai + dH[i]))
            dlds = self.ranks[i]
            bsb[i] = dbsb
            ldh[i] = dldh
            lds[i] = dlds
        g = bsb / 2.0 - lds / 2.0 + ldh / 2.0
        return g
    
    def hessian(self, rho):
        lam = np.exp(rho)
        beta, S = self.beta_rho(rho), self.get_penalty_mat(lam)
        Hb = self.hess_ll_beta(beta)
        dHb = self.dhess(beta, lam)
        d2Hb = self.d2hess(beta, lam)
        b1 = self.grad_beta_rho(beta, lam)
        Hp = Hb + S
        A = np.linalg.inv(Hp)
        D2r = b1.T.dot(Hp).dot(b1)
        H = np.zeros((self.ns, self.ns))
        for i in range(self.ns):
            Si, ai = self.S[i], lam[i]
            for j in range(i, self.ns):
                Sj, aj = self.S[j], lam[j]
                d = (i==j)
                H1i, H1j, H2ij = dHb[i], dHb[j], d2Hb[i, j]
                ldh2 = -(np.trace(A.dot(H1i+Si*ai).dot(A).dot(H1j+aj*Sj))\
                          -np.trace(A.dot(H2ij+d*ai*Si)))
                t1 = d * ai / (2.0) * beta.T.dot(Si).dot(beta)
                t2 = -D2r[i, j]
                H[i, j] = H[j, i] = t1 + t2 + ldh2/2
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
            
    def plot_smooth_comp(self, beta=None, single_fig=True, subplot_map=None, 
                         ci=95, fig_kws={}):
        beta = self.beta if beta is None else beta
        ci = sp.stats.norm(0, 1).ppf(1.0 - (100 - ci) / 200)
        methods = {"cr":crspline_basis, "cc":ccspline_basis,"bs":bspline_basis} 
        if single_fig:
            fig, ax = plt.subplots(**fig_kws)
            
        if subplot_map is None:
            subplot_map = dict(zip(np.arange(self.ns), np.arange(self.ns)))
        for i, (key, s) in enumerate(self.m.smooths.items()):
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
    
    def plot_smooth_quantiles(self, m_comp=None, s_comp=None, quantiles=None, figax=None):
        if quantiles is None:
            quantiles = [5, 10, 20, 30, 40]
        if figax is None:
            fig, ax = plt.subplots()
        else:
            fig, ax = figax
            
        methods = {"cr":crspline_basis, "cc":ccspline_basis,"bs":bspline_basis} 
        
        m, s = self.m.smooths[m_comp], self.s.smooths[s_comp]
        mk = m['knots']  
        x = np.linspace(mk.min(), mk.max(), 200)
        Xm = methods[m['kind']](x, mk, **m['fkws'])
        Xs = methods[s['kind']](x, mk, **s['fkws'])
        Xm, _ = absorb_constraints(m['q'], X=Xm)
        Xs, _ = absorb_constraints(s['q'], X=Xs)
        mu = self.m.link.inv_link(Xm.dot(self.beta[m['ix']]))
        tau = 1.0 / self.s.link.inv_link(Xs.dot(self.beta[self.ixs][s['ix']]))
        for q in quantiles:
            c = sp.stats.norm(0, 1).ppf(q/100)
            ax.fill_between(x, mu+c*tau, mu-c*tau, color='b', alpha=0.2, label=f"{2*q}th Quantile")
        ax.set_xlim(x.min(), x.max())
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
        rho = opt.x.copy()
        lambda_ = np.exp(rho)
        beta = self.beta_rho(rho)
        Slambda = self.get_penalty_mat(lambda_)
        Hbeta = self.hess_ll_beta(beta)
        Vb = np.linalg.inv(Hbeta + Slambda)
        Vp = np.linalg.inv(self.hessian(rho))
        Jb = self.grad_beta_rho(beta, lambda_)
        C = Jb.dot(Vp).dot(Jb.T)
        Vc = Vb + C
        Vs = Vb.dot(Hbeta).dot(Vb)
        Vf = Vs + C
        F = Vb.dot(Hbeta)
        self.Slambda = Slambda
        self.Vb, self.Vp, self.Vc, self.Vf, self.Vs = Vb, Vp, Vc, Vf, Vs
        self.opt, self.theta = opt, rho
        self.beta = beta
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
        yhat = self.m.X.dot(self.beta[self.ixm])
        resids = self.y - yhat
        ssr = np.sum(resids**2)
        sst =  np.sum((self.y - self.y.mean())**2)
        self.rsquared = 1.0 - ssr / sst
        self.ll_model = self.loglike(self.beta)
        self.aic = 2.0 * self.ll_model + 2.0 + self.edf * 2.0
        self.sumstats = pd.DataFrame([self.rsquared, self.ll_model, self.aic, self.edf], 
                                     index=['Rsquared', 'Loglike', 'AIC', 'EDF'])                                  
    
    
        


# rng = np.random.default_rng(123)


# n_obs = 20000
# df = pd.DataFrame(np.zeros((n_obs, 4)), columns=['x0', 'x1', 'x2', 'y'])
 
# df['x0'] = rng.choice(np.arange(5), size=n_obs, p=np.ones(5)/5)
# df['x1'] = rng.uniform(-1, 1, size=n_obs)
# df['x2'] = rng.uniform(-1, 1, size=n_obs)
# df['x3'] = rng.uniform(-2, 2, size=n_obs)

# u0 =  dummy(df['x0']).dot(np.array([-0.2, 0.2, -0.2, 0.2, 0.0]))
# f1 = (3.0 * df['x1']**3 - 2.43 * df['x1'])
# f2 = -(3.0 * df['x2']**3 - 2.43 * df['x2']) 
# f3 = (df['x3'] - 1.0) * (df['x3'] + 1.0)
# eta =  u0 + f1 + f2
# mu = eta.copy() 
# tau = 1.0 / (np.exp(f3) + 0.1)
# shape = mu * 2.0
# df['y'] = rng.normal(loc=mu, scale=tau)


# mod = GauLS("y~C(x0)+s(x1, kind='cr')+s(x2, kind='cr')", "y~1+s(x3, kind='cr')", df)

# theta = np.ones(3)
# g1 = mod.gradient(theta)
# g2 = numerical_derivs.fo_fc_cd(mod.reml, theta)

# H1 = mod.hessian(theta)
# H2 = numerical_derivs.so_gc_cd(mod.gradient, theta)


# sp.optimize.minimize(mod.reml, np.ones(3), jac=mod.gradient, hess=mod.hessian,
#                      method='trust-constr', options=dict(verbose=3))



# def fprime(f, x, eps=None):
#     if eps is None:
#         eps = np.finfo(float).eps**(1.0/2.0)
#     y = f(x)
#     p, q = y.shape[0], x.shape[0]
#     D = np.zeros((p, q))
#     h = np.zeros(q)
#     for i in range(q):
#         h[i] = eps
#         D[:, i] = (f(x+h) - f(x - h)) / (2.0 * eps)
#         h[i] = 0.0
#     return D
    

# def f2prime(f, x, eps=None):
#     eps = np.finfo(float).eps**(1/3) if eps is None else eps
#     J = f(x)
#     n, p = J.shape
#     Hn = np.zeros((p, p, n))
#     Hp = np.zeros((p, p, n))
#     H = np.zeros((p, p, n))
#     h = np.zeros(p)
#     for i in range(p):
#         h[i] = eps
#         Hp[i] = f(x+h).T
#         Hn[i] = f(x-h).T
#         h[i] = 0.0
#     for i in range(p):
#         for j in range(i+1):
#             H[i, j] = (Hp[i, j] - Hn[i, j] + Hp[j, i] - Hn[j, i]) / (4 * eps)
#             H[j, i] = H[i, j]
#     return H

# def f3prime(f, x, eps=None):
#     if eps is None:
#         eps = np.finfo(float).eps**(1.0/3.0)
#     y = f(x)
#     p, q = y.shape[0], x.shape[0]
#     D = np.zeros((q, p, p))
#     h = np.zeros(q)
#     for i in range(q):
#         h[i] = eps
#         D[i] = (f(x+h) - f(x - h)) / (2.0 * eps)
#         h[i] = 0.0
#     return D

# def f4prime(f, x, eps=None):
#     eps = np.finfo(float).eps**(1/3) if eps is None else eps
#     J = f(x)
#     p, n, n = J.shape
#     Hn = np.zeros((p, p, n, n))
#     Hp = np.zeros((p, p, n, n))
#     H =  np.zeros((p, p, n, n))
#     h = np.zeros(p)
#     for i in range(p):
#         h[i] = eps
#         Hp[i] = f(x+h)
#         Hn[i] = f(x-h)
#         h[i] = 0.0
#     for i in range(p):
#         for j in range(i+1):
#             H[i, j] = (Hp[i, j] - Hn[i, j] + Hp[j, i] - Hn[j, i]) / (4 * eps)
#             H[j, i] = H[i, j]
#     return H

# X = mod.Xt
# atol = np.finfo(float).eps**(1/3)
# rtol = 1e-5

# atolh = np.finfo(float).eps**(1/4)
# rtolh = 1e-3

# lam = np.array([ 83.30492, 1319.09376,   92.54213   ])
# rho = np.log(lam)

# rho = np.array([4.511599, 4.501194, 4.556914 ])
# lam = np.exp(rho)

# beta = mod.beta_rho(rho)
# S = mod.get_penalty_mat(lam)
# mod.outer_step(S)

# g1 = mod.grad_ll_beta(beta)
# g2 = numerical_derivs.fo_fc_cd(mod.loglike, beta)
# np.allclose(g1, g2, atol=atol)

# H1 = mod.hess_ll_beta(beta)
# H2 = numerical_derivs.so_gc_cd(mod.grad_ll_beta, beta)
# np.allclose(H1, H2, atol=atol)

# g1 = mod.grad_pll_beta(beta, S)
# g2 = numerical_derivs.fo_fc_cd(lambda x: mod.penalized_loglike(x, S), beta)
# np.allclose(g1, g2, atol=atol)


# H1 = mod.hess_pll_beta(beta, S)
# H2 = numerical_derivs.so_gc_cd(lambda x: mod.grad_pll_beta(x, S), beta)
# np.allclose(H1, H2, atol=atol)

# D1 = mod.grad_beta_rho(beta, lam)
# D2 = fprime(mod.beta_rho, rho)
# np.allclose(D1, D2, atol=atol)

# H1 = mod.hess_beta_rho(beta, lam)
# H2 = f2prime(mod._grad_beta_rho, rho, np.finfo(float).eps**(1/4))

# np.allclose(H1, H2, atol=atolh)


# mu = mod.m.X.dot(beta[mod.ixm])

# etas = mod.s.X.dot(beta[mod.ixs])
# L1, L2, L3, L4  = mod.ll_eta_derivs(beta[mod.ixm], beta[mod.ixs])



# f = lambda rho: mod.hess_ll_beta(mod.beta_rho(rho))

# dH1 = mod.dhess(beta, lam)
# dH2 = f3prime(f, rho, eps=np.finfo(float).eps**(1/3))
# np.allclose(dH1, dH2, atol=atolh, rtol=rtolh)



# f = lambda rho: mod.dhess(mod.beta_rho(rho), np.exp(rho))

# d2H1 = mod.d2hess(beta, lam)
# d2H2 = f4prime(f, rho, eps=np.finfo(float).eps**(1/3))


# theta = rho - 2.0

# mod.gradient(theta)

# T1 = np.eye(33)
# for key in mod.m.smooths.keys():
#     D, U = np.linalg.eigh(mod.m.smooths[key]['S'])
#     U = U[:, np.argsort(D)[::-1]]
#     D = D[np.argsort(D)[::-1]]
#     D[-1] = 1
#     D = np.sqrt(1.0 / D)
#     V = U.dot(np.diag(D))
#     Vi = U.dot(np.diag(1/D))
#     T1[mod.m.smooths[key]['ix'],  mod.m.smooths[key]['ix'][:, None]] = V.T

# p = mod.m.S.shape[1]

# for key in mod.s.smooths.keys():
#     D, U = np.linalg.eigh(mod.s.smooths[key]['S'])
#     U = U[:, np.argsort(D)[::-1]]
#     D = D[np.argsort(D)[::-1]]
#     D[-1] = 1
#     D = np.sqrt(1.0 / D)
#     V = U.dot(np.diag(D))
#     Vi = U.dot(np.diag(1/D))
#     T1[mod.s.smooths[key]['ix']+p,  mod.s.smooths[key]['ix'][:, None]+p] = V.T

# Xt = X.dot(T1)
# L1, L2, L3, L4  = mod.ll_eta_derivs(beta[mod.ixm], beta[mod.ixs])

# wmm, wms, wss = L2[:, 0], L2[:, 1], L2[:, 2]
# wmm, wms, wss = wmm.reshape(-1, 1), wms.reshape(-1, 1), wss.reshape(-1, 1)
# Xm, Xs = Xt[:, mod.ixm], Xt[:, mod.ixs]
# H11 = (Xm * wmm).T.dot(Xm)
# H12 = (Xm * wms).T.dot(Xs)
# H22 = (Xs * wss).T.dot(Xs)
# H1 = np.concatenate([H11, H12], axis=1)
# H2 = np.concatenate([H12.T, H22], axis=1)
# H = np.concatenate([H1, H2], axis=0)




