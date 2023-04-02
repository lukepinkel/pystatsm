#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 21:21:57 2020

@author: lukepinkel
"""
import numpy as np
import scipy as sp
import scipy.stats
import scipy.special
from abc import ABCMeta, abstractmethod
from scipy.special import digamma, polygamma, loggamma
from ..utilities.data_utils import _check_np, _check_shape
from ..utilities.func_utils import logbinom
from .links import (Link, IdentityLink, ReciprocalLink, LogLink, LogitLink,
                    PowerLink)

LN2PI = np.log(2.0 * np.pi)
FOUR_SQRT2 = 4.0 * np.sqrt(2.0)


class ExponentialFamily(metaclass=ABCMeta):

    def __init__(self, link=IdentityLink, weights=1.0, phi=1.0, dispersion=1.0):
        if not isinstance(link, Link):
            link = link()
        self._link = link
        self.weights = weights
        self.phi = phi
        self.dispersion = dispersion

    @abstractmethod
    def _loglike(self, y, mu, weights=1.0, phi=1.0, dispersion=1.0):
        pass

    @abstractmethod
    def _full_loglike(self, y, mu, weights=1.0, phi=1.0, dispersion=1.0):
        pass

    @abstractmethod
    def var_func(self, mu, dispersion=1.0):
        pass
    
    @abstractmethod
    def deviance(self, y, mu, weights=1.0, phi=1.0, dispersion=1.0):
        pass
    
    @abstractmethod
    def dvar_dmu(self, mu, dispersion=1.0):
        pass

    @abstractmethod
    def d2var_dmu2(self, mu, dispersion=1.0):
        pass
    
    @abstractmethod
    def d3var_dmu3(self, mu, dispersion=1.0):
        pass
    
    @abstractmethod
    def d4var_dmu4(self, mu, dispersion=1.0):
        pass
    
    @abstractmethod
    def var_derivs(self, mu, dispersion=1.0, order=1):
        pass

    def link(self, mu):
        return self._link.link(mu)

    def inv_link(self, eta):
        return self._link.inv_link(eta)

    def dinv_link(self, eta):
        return self._link.dinv_link(eta)

    def d2inv_link(self, eta):
        return self._link.d2inv_link(eta)

    def dlink(self, mu):
        return self._link.dlink(mu)

    def d2link(self, mu):
        res = self._link.d2link(mu)
        return res

    def d3link(self, mu):
        return self._link.d3link(mu)

    def d4link(self, mu):
        return self._link.d4link(mu)
        
    def loglike(self, y, mu, weights=1.0, phi=1.0, dispersion=1.0):
        ll_i = self._loglike(y, mu, weights, phi, dispersion)
        return np.sum(ll_i)

    def full_loglike(self, y, mu, weights=1.0, phi=1.0, dispersion=1.0):
        ll_i = self._full_loglike(y, mu, weights, phi, dispersion)
        return np.sum(ll_i)
    
    def pearson_resid(self, y, mu, weights=1.0, phi=1.0, dispersion=1.0):
        v = self.var_func(mu, dispersion)
        w = np.sqrt(weights / v)
        r_p = w * (y - mu)
        return r_p
    
    def variance(self, mu, weights=1.0, phi=1.0, dispersion=1.0):
        v = self.var_func(mu, dispersion) * phi / weights
        return v
    
    def pearson_chi2(self, y, mu, weights=1.0, phi=1.0, dispersion=1.0):
        v = self.var_func(mu, dispersion)
        w = weights / v
        chi2 = np.sum(w * (y - mu)**2)
        return chi2

    def signed_resid(self, y, mu, weights=1.0, phi=1.0, dispersion=1.0):
        d = self.deviance(y, mu=mu, phi=phi, dispersion=dispersion)
        r_s = np.sign(y - mu) * np.sqrt(d)
        return r_s

    def deviance_resid(self, y, mu, weights=1.0, phi=1.0, dispersion=1.0):
        d = self.deviance(y, mu=mu, phi=phi, dispersion=dispersion, weights=weights)
        r = np.sign(y - mu) * np.sqrt(d)
        return r

    def gw(self, y, mu, weights=1.0, phi=1.0, dispersion=1.0):
        num = (y - mu)
        den = self.variance(mu, weights, phi, dispersion) * self.dlink(mu)
        res = -num / den
        return res
    
    def hw(self, y, mu, weights=1.0, phi=1.0, dispersion=1.0):
        V0, V1 = self.var_derivs(mu, dispersion, 1)
        g1, g2 = self.dlink(mu), self.d2link(mu)
        g1V0 = g1 * V0 * phi
        r = y - mu
        a = 1.0 + r * (V1 / V0 + g2 / g1)
        hw = weights * a / (g1 * g1V0)
        return hw
 
    def get_ghw(self, y, mu, weights=1.0, phi=1.0, dispersion=1.0):
        V0, V1 = self.var_derivs(mu, dispersion, 1)
        g1, g2 = self.dlink(mu), self.d2link(mu)
        g1V0 = g1 * V0 * phi
        r = y - mu
        a = 1.0 + r * (V1 / V0 + g2 / g1)
        rw = weights * r
        gw = -rw / g1V0
        hw = weights * a / (g1 * g1V0)
        return gw, hw
        
    def get_a(self, y, mu, weights=1.0, phi=1.0, dispersion=1.0):
        V0, V1 = self.var_derivs(mu, dispersion, 1)
        g1, g2 = self.dlink(mu), self.d2link(mu)
        a = 1.0 + (y - mu) * (V1 / V0 + g2 / g1)
        return a

    def get_g(self, y, mu, weights=1.0, phi=1.0, dispersion=1.0):
        g = self.dlink(mu) / self._get_a(y=y, mu=mu, phi=phi, dispersion=dispersion)
        return g
    
    def get_w(self, y, mu, weights=1.0, phi=1.0, dispersion=1.0):
        a = self._get_a(y=y, mu=mu, weights=weights, phi=phi, dispersion=dispersion)
        v = self.var_func(mu, dispersion)
        dmu = self.dlink(mu)
        res = a / (dmu**2 * v)
        return res
    
    def get_ehw(self, y, mu, weights=1.0, phi=1.0, dispersion=1.0):
        den = self.variance(mu=mu, weights=weights, phi=phi, dispersion=dispersion) 
        den = den * self.dlink(mu)**2
        w = weights / den
        return w

    def da_dmu(self, y, mu=None, weights=None, phi=1.0, dispersion=1.0, **kwargs):
        y, mu, weights = self._check_args(**dict(y=y, mu=mu, weights=weights, **kwargs))
        V0, V1, V2 = self.var_derivs(mu, dispersion, 2)
        g1, g2, g3 = self.dlink(mu), self.d2link(mu), self.d3link(mu)
        u = (V1 / V0 + g2 / g1)
        v = (V2 / V0 - (V1 / V0)**2 + g3 / g1 - (g2 / g1)**2)
        res = (y - mu) * v - u
        return res

    def d2a_dmu2(self, y, mu=None, weights=None, phi=1.0, dispersion=1.0, **kwargs):
        y, mu, weights = self._check_args(**dict(y=y, mu=mu, weights=weights, **kwargs))
        V0, V1, V2, V3 = self.var_derivs(mu, dispersion, 3)
        g1 = self.dlink(mu)
        g2 = self.d2link(mu)
        g3 = self.d3link(mu)
        g4 = self.d4link(mu)
        u = V2 / V0 - (V1 / V0)**2 + g3 / g1 - (g2 / g1)**2
        v1 = V3 / V0 - 3.0 * (V1 * V2) / V0**2 + 2.0 * (V1 / V0)**3
        v2 = g4 / g1 - 3.0 * (g3 * g2) / g1**2 + 2.0 * (g2 / g1)**3
        res = (y - mu) * (v1 + v2) - 2.0 * u
        return res

    def dw_deta(self, y, mu=None, weights=None, phi=1.0, dispersion=1.0, **kwargs):
        y, mu, weights = self._check_args(**dict(y=y, mu=mu, weights=weights, **kwargs))
        a0 = self.get_a(y=y, mu=mu, dispersion=dispersion)
        a1 = self.da_dmu(y=y, mu=mu, dispersion=dispersion)
        w = self.get_w(y=y, mu=mu, dispersion=dispersion)
        V0, V1 = self.var_derivs(mu, dispersion, 1)
        g1, g2 = self.dlink(mu), self.d2link(mu)
        res = (w / g1) * (a1 / a0 - V1 / V0 - 2.0 * g2 / g1)
        return res

    def d2w_deta2(self, y, mu=None, weights=None, phi=1.0, dispersion=1.0, **kwargs):
        y, mu, weights = self._check_args(**dict(y=y, mu=mu, weights=weights, **kwargs))
        V0, V1, V2, V3 = self.var_derivs(mu, dispersion, 3)
        g1, g2, g3, g4 = self.dlink(mu), self.d2link(mu), self.d3link(mu), self.d4link(mu)
        g21 = g2 / g1
        g31 = g3 / g1
        V10 = V1 / V0        
        V20 = V2 / V0
        V30 = V3 / V0
        V12 = V1 * V2
        V10_sq = V10**2
        V10_cb = V10 * V10_sq
        g21_sq = g21**2
        g21_cb = g21 * g21_sq 
        a0 = 1.0 + (y - mu) * (V10 + g21)
        a1 = (y - mu) *  (V20 - V10_sq + g31 - g21_sq) -  (V10 + g21)
        u = V20 - (V10)**2 + g31 - (g21)**2
        v1 = V30 - 3.0 * (V12) / V0**2 + 2.0 * V10_cb
        v2 = g4 / g1 - 3.0 * (g3 * g2) / g1**2 + 2.0 * g21_cb
        a2 = (y - mu) * (v1 + v2) - 2.0 * u
        
        dmu = self.dlink(mu)
        w0 = a0 / (dmu**2 * V0)
        w1 = (w0 / g1) * (a1 / a0 - V10 - 2.0 * g21)
        t1 = w1**2 / w0
        t2 = w1 * g2 / g1**2
        t3 = a2 / a0 - (a1 / a0)**2 - V2 / V0 + (V1 / V0)**2 \
            - 2.0 * (g3 / g1) + 2.0 * (g2 / g1)**2
        t3 *= w0 / (g1**2)
        res = t1 - t2 + t3
        return res


class Gaussian(ExponentialFamily):

    def __init__(self, link=IdentityLink, weights=1.0, phi=1.0):
        self.name = "Gaussian"
        super().__init__(link, weights, phi)

    def _loglike(self, y, mu, weights=1.0, phi=1.0, dispersion=1.0):
        w = weights / phi
        ll = w * np.power((y - mu), 2) + np.log(1.0 / w)
        ll = ll / 2.0
        return ll

    def _full_loglike(self, y, mu, weights=None, phi=1.0, dispersion=1.0):
        ll = self._loglike(y, mu, weights, phi, dispersion)
        llf = ll + LN2PI / 2.0
        return llf
    
    def var_func(self, mu, dispersion=1.0):
        V = np.ones_like(mu)
        return V

    def deviance(self, y, mu, weights=1.0, phi=1.0, dispersion=1.0):
        d = weights * np.power((y - mu), 2.0)
        return d

    def dtau(self, tau, y, mu, weights=1.0, reduce=True):
        phi = np.exp(tau)
        g = -(weights * np.power((y - mu), 2) / phi - 1) / 2
        if reduce:
            g = np.sum(g)
        return g

    def d2tau(self, tau, y, mu, weights=1.0, reduce=True):
        phi = np.exp(tau)
        h = weights * np.power((y - mu), 2) / (2 * phi)
        if reduce:
            h = np.sum(h)
        return h

    def dvar_dmu(self, mu, dispersion=1.0):
        return np.zeros_like(mu)

    def d2var_dmu2(self, mu, dispersion=1.0):
        return np.zeros_like(mu)

    def d3var_dmu3(self, mu, dispersion=1.0):
        return np.zeros_like(mu)

    def d4var_dmu4(self, mu, dispersion=1.0):
        return np.zeros_like(mu)
    
    def var_derivs(self, mu, dispersion=1.0, order=0):
        V0 = np.ones_like(mu)
        if order==0:
            return V0
        V1 = np.zeros_like(mu)
        if order==1:
            return V0, V1
        V2 = np.zeros_like(mu)
        if order==2:
            return V0, V1, V2
        V3 = np.zeros_like(mu)
        if order==3:
            return V0, V1, V2, V3
        V4 = np.zeros_like(mu)
        if order==4:
            return V0, V1, V2, V3, V4

    def llscale(self, phi, y):
        ls = len(y) * (np.log(phi) / 2.0 + np.log(2.0*np.pi) / 2.0)
        return ls

    def dllscale(self, phi, y):
        ls1 = len(y) / (2.0 * phi)
        return ls1

    def d2llscale(self, phi, y):
        ls2 = -len(y) / (2.0 * phi**2)
        return ls2

    def rvs(self, mu, phi=1.0, rng=None, seed=None):
        rng = np.random.default_rng(seed) if rng is None else rng
        return rng.normal(loc=mu, scale=np.sqrt(phi))

    def ppf(self, q, mu, phi):
        return sp.stats.norm(loc=mu, scale=np.sqrt(phi)).ppf(q)


class InverseGaussian(ExponentialFamily):

    def __init__(self, link=PowerLink(-2), weights=1.0, phi=1.0):
        self.name = "InverseGaussian"
        super().__init__(link, weights, phi)

    def _loglike(self, y, mu, weights=1.0, phi=1.0, dispersion=1.0):
        num = (y - mu)**2
        den = (y * mu**2 * phi)
        ll = weights * num / den + np.log((phi * y**3) / weights)
        ll = ll / 2.0
        return ll

    def _full_loglike(self, y, mu, weights=1.0, phi=1.0, dispersion=1.0):
        ll = self._loglike(y, mu, weights, phi, dispersion)
        llf = ll + LN2PI / 2.0
        return llf
    
    def var_func(self, mu, dispersion=1.0):
        V = mu**3
        return V
    
    def var_derivs(self, mu, dispersion=1.0, order=0):
        V0 = np.power(_check_shape(mu, 1), 3.0)
        if order==0:
            return V0
        V1 = 3.0 * mu**2
        if order==1:
            return V0, V1
        V2 = 6.0 * mu
        if order==2:
            return V0, V1, V2
        V3 = np.ones_like(mu) * 6.0
        if order==3:
            return V0, V1, V2, V3
        V4 = np.zeros_like(mu)
        if order==4:
            return V0, V1, V2, V3, V4

    def deviance(self, y, mu, weights=1.0, phi=1.0, dispersion=1.0):
        d = weights * np.power((y - mu), 2.0) / (y * np.power(mu, 2))
        return d

    def dtau(self, tau, y, mu, weights=1.0, reduce=True):
        phi = np.exp(tau)
        num = weights * np.power((y - mu), 2)
        den = (phi * y * np.power(mu, 2))
        g = -(num / den - 1) / 2
        if reduce:
            g = np.sum(g)
        return g

    def d2tau(self, tau, y, mu, weights=1.0, reduce=True):
        phi = np.exp(tau)
        h = weights * np.power((y - mu), 2) / (2 * phi * y * mu**2)
        if reduce:
            h = np.sum(h)
        return h

    def dvar_dmu(self, mu, dispersion=1.0):
        return 3.0 * mu**2

    def d2var_dmu2(self, mu, dispersion=1.0):
        return 6.0 * mu

    def d3var_dmu3(self, mu, dispersion=1.0):
        return np.ones_like(mu) * 6.0

    def d4var_dmu4(self, mu, dispersion=1.0):
        return np.zeros_like(mu)

    def llscale(self, phi, y):
        ls = np.sum(np.log(phi) / 2.0 + np.log(2.0*np.pi*y**3) / 2.0)
        return ls

    def dllscale(self, phi, y):
        ls1 = len(y) / (2.0 * phi)
        return ls1

    def d2llscale(self, phi, y):
        ls2 = -len(y) / (2.0 * phi**2)
        return -ls2

    def rvs(self, mu, phi=1.0, rng=None, seed=None):
        rng = np.random.default_rng(seed) if rng is None else rng
        return rng.wald(mean=mu, scale=np.sqrt(phi))

    def ppf(self, q, mu, phi=1.0):
        return sp.stats.wald(loc=mu, scale=np.sqrt(phi)).ppf(q)


class Gamma(ExponentialFamily):

    def __init__(self, link=ReciprocalLink, weights=1.0, phi=1.0):
        self.name = "Gamma"
        super().__init__(link, weights, phi)

    def _loglike(self, y, mu, weights=1.0, phi=1.0, dispersion=1.0):
        w = weights / phi
        z = w * y / mu
        ll = z - w * np.log(z) + sp.special.gammaln(weights/phi)
        return ll

    def _full_loglike(self, y, mu, weights=1.0, phi=1.0, dispersion=1.0):
        ll = self._loglike(y, mu, weights, phi, dispersion)
        llf = ll + np.log(y)
        return llf

    def var_func(self, mu, dispersion=1.0):
        V = mu**2
        return V


    def deviance(self, y, mu, weights=1.0, phi=1.0, dispersion=1.0):
        d = 2 * weights * ((y - mu) / mu - np.log(y / mu))
        return d

    def dtau(self, tau, y, mu, weights=1.0, reduce=True):
        phi = np.exp(tau)
        T0 = np.log(weights * y / (phi * mu))
        T1 = (1 - y / mu)
        T2 = -sp.special.digamma(weights / phi)
        g = (weights / phi) * (T0 + T1 + T2)
        if reduce:
            g = np.sum(g)
        return g

    def d2tau(self, tau, y, mu, weights=1.0, reduce=True):
        phi = np.exp(tau)
        T0 = np.log(weights * y / (phi * mu))
        T1 = (2 - y / mu)
        T2 = sp.special.digamma(weights / phi)
        T3 = weights / phi * sp.special.polygamma(1, weights / phi)
        h = weights / phi * (T3+T2-T1-T0)
        if reduce:
            h = np.sum(h)
        return h

    def dvar_dmu(self, mu, dispersion=1.0):
        return 2.0 * mu

    def d2var_dmu2(self, mu, dispersion=1.0):
        return np.ones_like(mu) * 2.0

    def d3var_dmu3(self, mu, dispersion=1.0):
        return np.zeros_like(mu)

    def d4var_dmu4(self, mu, dispersion=1.0):
        return np.zeros_like(mu)
    
    def var_derivs(self, mu, dispersion=1.0, order=0):
        V0 = mu**2
        if order==0:
            return V0
        V1 = 2.0 * mu
        if order==1:
            return V0, V1
        V2 = np.ones_like(mu) * 2.0
        if order==2:
            return V0, V1, V2
        V3 = np.zeros_like(mu)
        if order==3:
            return V0, V1, V2, V3
        V4 = np.zeros_like(mu)
        if order==4:
            return V0, V1, V2, V3, V4

    def llscale(self, phi, y):
        v = 1.0 / phi
        ls = (sp.special.gammaln(v) + np.log(phi) * v + v + np.log(y)).sum()
        return ls

    def dllscale(self, phi, y):
        v = 1.0 / phi
        ls1 = -(sp.special.digamma(v) + np.log(phi)) * v**2 * len(y)
        return ls1

    def d2llscale(self, phi, y):
        v = 1.0 / phi
        ls2 = (sp.special.polygamma(1, v) * v + 2.0 * sp.special.digamma(v)
               - (1.0 - 2.0 * np.log(phi))) * v**3 * len(y)
        return ls2

    def rvs(self, mu, phi=1.0, rng=None, seed=None):
        rng = np.random.default_rng(seed) if rng is None else rng
        return rng.gamma(shape=1/phi, scale=mu*phi)

    def ppf(self, q, mu, phi=1.0):
        a = 1 / phi
        b = 1 / (mu * phi)
        return sp.stats.gamma(a=a, scale=1/b).ppf(q)


class NegativeBinomial(ExponentialFamily):

    def __init__(self, link=LogLink, weights=1.0, phi=1.0):
        self.name = "NegativeBinomial"
        super().__init__(link, weights, phi)

    def _loglike(self, y, mu, weights=1.0, phi=1.0, dispersion=1.0):
        a = 1 / dispersion
        kmu = dispersion * mu
        ypa = y + a
        llk = -weights * (y * np.log(kmu) - ypa * np.log(1 + kmu) + loggamma(ypa) - loggamma(a))
        return llk

    def _full_loglike(self, y, mu, weights=1.0, phi=1.0, dispersion=1.0):
        ll = self._loglike(y, mu, weights, phi, dispersion)
        llf = ll + weights / 1.0 * loggamma(y + 1.0)
        return llf

    def var_func(self, mu, dispersion=1.0):
        V = mu + np.power(mu, 2) * dispersion
        return V

    def deviance(self, y, mu, weights=1.0, phi=1.0, dispersion=1.0):
        d = np.zeros(y.shape[0])
        ix = (y == 0)
        v = 1.0 / dispersion
        d[ix] = np.log(1 + dispersion * mu[ix]) / dispersion
        yb, mb = y[~ix], mu[~ix]
        u = (yb + v) / (mb + v)
        d[~ix] = (yb*np.log(yb / mb) - (yb + v) * np.log(u))
        d *= 2*weights
        return d

    def dtau(self, tau, y, mu, weights=1.0, reduce=True):
        dispersion = np.exp(tau)
        T0 = dispersion * (y - mu) / (1 + dispersion * mu)
        T1 = np.log(1 + dispersion * mu)
        T2 = -sp.special.digamma(y + 1 / dispersion)
        T3 = sp.special.digamma(1 / dispersion)
        g = -(weights / dispersion) * (T0 + T1 + T2 + T3)
        if reduce:
            g = np.sum(g)
        return g

    def d2tau(self, tau, y, mu, weights=1.0, reduce=True):
        dispersion = np.exp(tau)
        a = 1.0 / dispersion
        ypa = y + a
        u = 1 + dispersion * mu
        denom = -y * dispersion * mu + mu + 2 * dispersion * mu**2
        numer = u**2
        v = denom / numer
        dtt = v - a * np.log(u) \
            + a * (sp.special.digamma(ypa) - sp.special.digamma(a)) \
            + a**2 * (sp.special.polygamma(1, ypa) -
                      sp.special.polygamma(1, a))
  
        h = -weights * dtt
        if reduce:
            h = np.sum(h)
        return h

    def dvar_dmu(self, mu, dispersion=1.0):
        return 2.0 * dispersion * mu + 1.0

    def d2var_dmu2(self, mu, dispersion=1.0):
        return 2.0 * dispersion

    def d3var_dmu3(self, mu, dispersion=1.0):
        return np.zeros_like(mu)

    def d4var_dmu4(self, mu, dispersion=1.0):
        return np.zeros_like(mu)
    
    def var_derivs(self, mu, dispersion=1.0, order=0):
        V0 = mu + mu**2 * dispersion
        if order==0:
            return V0
        V1 = 2.0 * dispersion * mu + 1.0
        if order==1:
            return V0, V1
        V2 = 2.0 * np.ones_like(mu) * dispersion 
        if order==2:
            return V0, V1, V2
        V3 = np.zeros_like(mu)
        if order==3:
            return V0, V1, V2, V3
        V4 = np.zeros_like(mu)
        if order==4:
            return V0, V1, V2, V3, V4

    def llscale(self, phi, y):
        return None

    def dllscale(self, phi, y):
        return None

    def d2llscale(self, phi, y):
        return None

    def rvs(self, mu, phi=1.0, dispersion=1.0, rng=None, seed=None):
        rng = np.random.default_rng(seed) if rng is None else rng
        var = mu + dispersion * mu**2
        n = - mu**2 / (mu - var)
        p = mu / var
        y = rng.negative_binomial(n=n, p=p)
        return y
    
    def ppf(self, q, mu, dispersion=1.0):
        var = mu + dispersion * mu**2
        n = - mu**2 / (mu - var)
        p = mu / var
        return sp.stats.nbinom(n=n, p=p).ppf(q)


class Poisson(ExponentialFamily):

    def __init__(self, link=LogLink, weights=1.0, phi=1.0):
        self.name = "Poisson"
        super().__init__(link, weights, phi)

    def _loglike(self, y, mu, weights=1.0, phi=1.0, dispersion=1.0):
        w = weights / phi
        ll = -w * (y * np.log(mu) - mu)
        return ll

    def _full_loglike(self, y, mu, weights=1.0, phi=1.0, dispersion=1.0):
        ll = self._loglike(y, mu, weights, phi, dispersion)
        llf = ll + weights / phi * sp.special.gammaln(y+1)
        return llf

    def var_func(self, mu, dispersion=1.0):
        V = mu
        return V

    def deviance(self, y, mu, weights=1.0, phi=1.0, dispersion=1.0):
        d = np.zeros(y.shape[0])
        ixa = y == 0
        ixb = ~ixa
        d[ixa] = mu[ixa]
        d[ixb] = (y[ixb]*np.log(y[ixb]/mu[ixb]) - (y[ixb] - mu[ixb]))
        d *= 2.0 * weights
        return d

    def dvar_dmu(self, mu, dispersion=1.0):
        return np.ones_like(mu)

    def d2var_dmu2(self, mu, dispersion=1.0):
        return np.zeros_like(mu)

    def d3var_dmu3(self, mu, dispersion=1.0):
        return np.zeros_like(mu)

    def d4var_dmu4(self, mu, dispersion=1.0):
        return np.zeros_like(mu)

    def var_derivs(self, mu, dispersion=1.0, order=0):
        V0 = mu 
        if order==0:
            return V0
        V1 = np.ones_like(mu)
        if order==1:
            return V0, V1
        V2 =  np.zeros_like(mu)
        if order==2:
            return V0, V1, V2
        V3 = np.zeros_like(mu)
        if order==3:
            return V0, V1, V2, V3
        V4 = np.zeros_like(mu)
        if order==4:
            return V0, V1, V2, V3, V4
        
    def llscale(self, phi, y):
        return None

    def dllscale(self, phi, y):
        return None

    def d2llscale(self, phi, y):
        return None

    def rvs(self, mu, scale=1.0, rng=None, seed=None):
        rng = np.random.default_rng(seed) if rng is None else rng
        y = rng.poisson(lam=mu)
        return y

    def ppf(self, q, mu, scale=1.0):
        return sp.stats.poisson(mu=mu, scale=np.sqrt(scale)).ppf(q)


class Binomial(ExponentialFamily):

    def __init__(self, link=LogitLink, weights=1.0, phi=1.0):
        self.name = "Binomial"
        super().__init__(link, weights, phi)

    def _loglike(self, y, mu, weights=1.0, phi=1.0, dispersion=1.0):
        w = weights / phi
        ll = -w * (y * np.log(mu) + (1 - y) * np.log(1 - mu))
        return ll

    def _full_loglike(self, y, mu, weights=1.0, phi=1.0, dispersion=1.0):
        ll = self._loglike(y, mu, weights, phi, dispersion)
        r = weights * y
        llf = ll - logbinom(weights, r)
        return llf

    def var_func(self, mu, dispersion=1.0):
        V = mu * (1 - mu)
        return V

    def deviance(self, y, mu, weights=1.0, phi=1.0, dispersion=1.0, **kwargs):
        ixa = y == 0
        ixb = (y != 0) & (y != 1)
        ixc = y == 1
        d = np.zeros(y.shape[0])
        u = (1 - y)[ixb]
        v = (1 - mu)[ixb]
        d[ixa] = -np.log(1-mu[ixa])
        d[ixc] = -np.log(mu[ixc])
        d[ixb] = y[ixb]*np.log(y[ixb]/mu[ixb]) + u*np.log(u/v)
        return 2*weights*d

    def dvar_dmu(self, mu, dispersion=1.0):
        return 1.0 - 2.0 * mu

    def d2var_dmu2(self, mu, dispersion=1.0):
        return np.ones_like(mu) * -2.0

    def d3var_dmu3(self, mu, dispersion=1.0):
        return np.zeros_like(mu)

    def d4var_dmu4(self, mu, dispersion=1.0):
        return np.zeros_like(mu)

    def var_derivs(self, mu, dispersion=1.0, order=0):
        V0 =  mu * (1 - mu) 
        if order==0:
            return V0
        V1 = 1.0 - 2.0 * mu
        if order==1:
            return V0, V1
        V2 =  np.ones_like(mu) * -2.0
        if order==2:
            return V0, V1, V2
        V3 = np.zeros_like(mu)
        if order==3:
            return V0, V1, V2, V3
        V4 = np.zeros_like(mu)
        if order==4:
            return V0, V1, V2, V3, V4
        
    def llscale(self, phi, y):
        return None

    def dllscale(self, phi, y):
        return None

    def d2llscale(self, phi, y):
        return None

    def rvs(self, mu=None, loc=None, scale=1.0, rng=None, seed=None):
        mu = loc if mu is None else mu
        rng = np.random.default_rng(seed) if rng is None else rng
        y = rng.binomial(n=self.weights, p=mu) / self.weights
        return y

    def ppf(self, q, mu, scale=1.0):
        return sp.stats.binom(n=self.weights, p=mu).ppf(q) / self.weights
