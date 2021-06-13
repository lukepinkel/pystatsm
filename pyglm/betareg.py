# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 20:40:59 2021

@author: lukepinkel
"""
import patsy
import numpy as np
import scipy as sp
import scipy.stats
import pandas as pd
from scipy.special import gammaln, digamma, polygamma
from .links import LogitLink, LogLink, Link


def wdprod(X, w, y):
    XWy =  (X * w.reshape(-1, 1)).T.dot(y)
    return XWy


class BetaReg:
    
    def __init__(self, X=None, Z=None, y=None, m_formula=None, s_formula=None, 
                 data=None, m_link=LogitLink, s_link=LogLink):
        
        if not isinstance(m_link, Link):
            m_link = m_link()
            
        if not isinstance(s_link, Link):
            s_link = s_link()
        
        if m_formula is not None and s_formula is not None and data is not None:
            y, X = patsy.dmatrices(m_formula, data=data, return_type='dataframe')
            _, Z = patsy.dmatrices(s_formula, data=data, return_type='dataframe')
            xcols, xinds = X.columns, X.index
            zcols, zinds = Z.columns, Z.index
            ycols, yinds = y.columns, y.index
            X, Z, y = X.values, Z.values, y.values[:, 0]
        elif X is not None and Z is not None and y is not None:
            if type(X) not in [pd.DataFrame, pd.Series]:
                xcols = [f'x{i}' for i in range(1, X.shape[1]+1)]
                xinds = np.arange(X.shape[0])
            else:
                xcols, xinds = X.columns, X.index
                X = X.values
        
            if type(Z) not in [pd.DataFrame, pd.Series]:
                zcols = [f'z{i}' for i in range(1, Z.shape[1]+1)]
                zinds = np.arange(Z.shape[0])
            else:
                zcols, zinds = Z.columns, Z.index
                Z = Z.values
    
            if type(y) not in [pd.DataFrame, pd.Series]:
                ycols = ['y']
                yinds = np.arange(y.shape[0])
            else:
                 ycols, yinds = y.columns, y.index
                 y = y.values
        
        self.X, self.Z, self.y = X, Z, y
        self.xcols, self.zcols, self.ycols = xcols, zcols, ycols
        self.xinds, self.zinds, self.yinds = xinds, zinds, yinds
        self.m_formula, self.s_formula = m_formula, s_formula
        self.m_link, self.s_link = m_link, s_link
        self.n_obs, self.n_xvar = X.shape
        self.n_zvar = Z.shape[1]
        self.theta_init = np.zeros(self.n_xvar+self.n_zvar)
        self.m_ix = np.arange(self.n_xvar)
        self.s_ix = np.arange(self.n_xvar, self.n_xvar+self.n_zvar)
        self.ys = np.log(y / (1.0 - y))
        self.logy = np.log(y)
        self.log1y = np.log(1.0 - y)
        
    def loglike_elementwise(self, theta):
        betam, betas = theta[self.m_ix], theta[self.s_ix]
        etam, etas = self.X.dot(betam), self.Z.dot(betas)
        mu, phi = self.m_link.inv_link(etam), self.s_link.inv_link(etas)
        mu_phi = mu * phi
        ll_i = gammaln(phi) - gammaln(mu_phi) - gammaln(phi - mu_phi) +\
               (mu_phi - 1.0) * self.logy + (phi - mu_phi - 1.0) * self.log1y
        return -ll_i
    
    def loglike(self, theta):
        return np.sum(self.loglike_elementwise(theta))
    
    def gradient(self, theta):
        betam, betas = theta[self.m_ix], theta[self.s_ix]
        etam, etas = self.X.dot(betam), self.Z.dot(betas)
        mu, phi = self.m_link.inv_link(etam), self.s_link.inv_link(etas)
        mu_phi = mu * phi
        wm = phi / self.m_link.dlink(mu)
        ws = 1.0 / self.s_link.dlink(phi)
        
        d = digamma(phi - mu_phi)
        u = digamma(mu_phi) - d
        rm = self.ys - u
        rs = mu * rm + digamma(phi) - d + self.log1y
        
        gm = wdprod(self.X, wm, rm)
        gs = wdprod(self.Z, ws, rs)
        g = -np.concatenate([gm, gs])
        return g
    
    def hessian(self, theta):
        betam, betas = theta[self.m_ix], theta[self.s_ix]
        etam, etas = self.X.dot(betam), self.Z.dot(betas)
        mu, phi = self.m_link.inv_link(etam), self.s_link.inv_link(etas)
        mu_phi = mu * phi
        phi_mu = phi - mu_phi
        
        dg0, dg1, dg2 = digamma(phi), digamma(mu_phi), digamma(phi_mu)
        pg0, pg1, pg2 = polygamma(1, phi), polygamma(1, mu_phi), polygamma(1, phi_mu)
        u = dg1 - dg2
        dL_dmu = -phi * (self.ys - u)
        d2L_dmu2 = phi**2 * (pg1 + pg2)
        g1m = 1.0 / self.m_link.dlink(mu)
        g2m = -self.m_link.d2link(mu) *  g1m**2
        
        g1s = 1.0 / self.s_link.dlink(phi)
        g2s = -self.s_link.d2link(phi) * g1s**2
        #gs2 = self.s_link.d2link(phi)
        u1 = mu * pg1 - (1 - mu) * pg2
        
        dL_dphi = mu * (self.ys - (dg1 - dg2)) + dg0 - dg2 + self.log1y
        d2L_dphi2 = -mu**2 * pg1 + (mu-1)**2*(-pg2)+pg0
        
        #a = -(1.0 - mu) * pg2 + mu * ((1 - mu)*pg2 - u*pg1) + pg0
        #b = mu * (self.ys - dg1 + dg2) - dg2 + dg0 + self.log1y
        
        wmm = (d2L_dmu2 * g1m + dL_dmu * g2m) * g1m
        wms = -((self.ys - u) - phi * u1) * g1m * g1s
        #wss = -(a * g1s - b * g1s**2 * gs2)
        wss = -(d2L_dphi2 * g1s + dL_dphi * g2s) * g1s
        
        Hmm = wdprod(self.X, wmm, self.X)
        Hms = wdprod(self.X, wms, self.Z)
        Hss = wdprod(self.Z, wss, self.Z)
        H = np.block([[Hmm, Hms], [Hms.T, Hss]])
        return H

    def fit(self):
        opt = sp.optimize.minimize(self.loglike, self.theta_init, hess=self.hessian, 
                                   jac=self.gradient, method='trust-constr', 
                                   options=dict(verbose=3))
        H = self.hessian(opt.x)
        se = np.sqrt(np.diag(np.linalg.inv(H)))
        theta = opt.x
        res = pd.DataFrame(np.vstack((theta, se, theta/se)).T, columns=['param', 'SE', 't'])
        res.index = self.xcols.tolist() + self.zcols.tolist()
        res['p'] = sp.stats.t(df=self.n_obs-self.n_xvar-self.n_zvar).sf(np.abs(res['t']))*2.0
        self.res = res
        self.theta = theta
        self.optimizer = opt
        
        
        
        