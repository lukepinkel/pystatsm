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
from scipy.special import digamma, polygamma, loggamma
from ..utilities.data_utils import _check_np, _check_shape
from ..utilities.func_utils import logbinom
from .links import (Link, IdentityLink, ReciprocalLink, LogLink, LogitLink,
                    PowerLink)

LN2PI = np.log(2.0 * np.pi)
FOUR_SQRT2 = 4.0 * np.sqrt(2.0)


class ExponentialFamily(object):

    def __init__(self, link=IdentityLink, weights=1.0, phi=1.0, dispersion=1.0):
        if not isinstance(link, Link):
            link = link()
        self._link = link
        self.weights = weights
        self.phi = phi
        self.dispersion = dispersion

    def _to_mean(self, eta=None, T=None):
        if eta is not None:
            mu = self.inv_link(eta)
        else:
            mu = self.mean_func(T)
        return mu

    def link(self, mu):
        return self._link.link(mu)

    def inv_link(self, eta):
        return self._link.inv_link(eta)

    def dinv_link(self, eta):
        return self._link.dinv_link(eta)

    def d2inv_link(self, eta):
        return self._link.d2inv_link(eta)

    def dlink(self, mu):
        return 1.0 / self.dinv_link(self.link(mu))

    def d2link(self, mu):
        res = self._link.d2link(mu)
        return res

    def d3link(self, mu):
        return self._link.d3link(mu)

    def d4link(self, mu):
        return self._link.d4link(mu)

    def cshape(self, y, mu):
        y = _check_shape(_check_np(y), 1)
        mu = _check_shape(_check_np(mu), 1)
        return y, mu

    def loglike(self, y, eta=None, mu=None, T=None, phi=1.0, dispersion=1.0, weights=None):
        return np.sum(self._loglike(y, eta, mu, T, phi, dispersion, weights))

    def full_loglike(self, y, eta=None, mu=None, T=None, phi=1.0, dispersion=1.0, weights=None):
        return np.sum(self._full_loglike(y, eta, mu, T, phi, dispersion, weights))

    def pearson_resid(self, y, eta=None, mu=None, T=None, phi=1.0, dispersion=1.0, weights=None):
        mu = self._to_mean(eta=eta, T=T) if mu is None else mu
        weights = self.weights if weights is None else weights
        y, mu = self.cshape(y, mu)
        V = self.var_func(mu=mu, dispersion=dispersion)
        w = np.sqrt(self.weights / V)
        r_p = w * (y - mu)
        return r_p
    
    def variance(self, eta=None, mu=None, T=None, phi=1.0, dispersion=1.0):
        mu = self._to_mean(eta=eta, T=T) if mu is None else mu
        v = self.var_func(mu=mu, dispersion=dispersion) * phi
        return v

    def pearson_chi2(self, y, eta=None, mu=None, T=None, phi=1.0, dispersion=1.0, weights=None):
        mu = self._to_mean(eta=eta, T=T) if mu is None else mu
        weights = self.weights if weights is None else weights
        y, mu = self.cshape(y, mu)
        V = self.var_func(mu=mu, dispersion=dispersion)
        w = weights / V
        chi2 = np.sum(w * (y - mu)**2)
        return chi2

    def signed_resid(self, y, eta=None, mu=None, T=None, phi=1.0, dispersion=1.0, weights=None):
        mu = self._to_mean(eta=eta, T=T) if mu is None else mu
        y, mu = self.cshape(y, mu)
        d = self.deviance(y, mu=mu, phi=phi, dispersion=dispersion)
        r_s = np.sign(y - mu) * np.sqrt(d)
        return r_s

    def deviance_resid(self, y, eta=None, mu=None, T=None, phi=1.0, dispersion=1.0, weights=None):
        mu = self._to_mean(eta=eta, T=T) if mu is None else mu
        weights = self.weights if weights is None else weights
        d = self.deviance(y, mu=mu, phi=phi, dispersion=dispersion, weights=weights)
        r = np.sign(y - mu) * np.sqrt(d)
        return r

    def gw(self, y, mu, phi=1.0, dispersion=1.0, weights=None):
        weights = self.weights if weights is None else weights
        y, mu = self.cshape(y, mu)
        num = weights * (y - mu)
        den = self.variance(mu=mu, phi=phi, dispersion=dispersion) * self.dlink(mu)
        res = num / den
        return -res

    def hw(self, y, mu, phi=1.0, dispersion=1.0, weights=None):
        weights = self.weights if weights is None else weights
        y, mu = self.cshape(y, mu)
        V0 = self.var_func(mu=mu, dispersion=dispersion)
        V1 = self.dvar_dmu(mu, dispersion=dispersion)
        g1, g2 = self.dlink(mu), self.d2link(mu)
        g1V0 = g1 * V0 * phi
        r = y - mu
        a = 1.0 + r * (V1 / V0 + g2 / g1)
        hw = weights * a / (g1 * g1V0)
        return hw

    def get_ghw(self, y, mu, phi=1.0, dispersion=1.0, weights=None):
        weights = self.weights if weights is None else weights
        y, mu = self.cshape(y, mu)
        V0 = self.var_func(mu=mu, dispersion=dispersion)
        V1 = self.dvar_dmu(mu, dispersion=dispersion)
        g1, g2 = self.dlink(mu), self.d2link(mu)
        # if self.name == "NegativeBinomial":
        #     g1V0 = g1 * V0
        # else:
        #     g1V0 = g1 * V0 * phi
        g1V0 = g1 * V0 * phi
        r = y - mu
        a = 1.0 + r * (V1 / V0 + g2 / g1)
        rw = weights * r
        gw = -rw / g1V0
        hw = weights * a / (g1 * g1V0)
        return gw, hw

    def get_a(self, y, mu, dispersion=1.0):
        y, mu = self.cshape(y, mu)
        V0 = self.var_func(mu=mu, dispersion=dispersion)
        V1 = self.dvar_dmu(mu, dispersion=dispersion)
        g1, g2 = self.dlink(mu), self.d2link(mu)
        a = 1.0 + (y - mu) * (V1 / V0 + g2 / g1)
        return a

    def get_g(self, y, mu, dispersion=1.0):
        y, mu = self.cshape(y, mu)
        res = self.dlink(mu) / self.get_a(y=y, mu=mu, dispersion=dispersion)
        return res

    def get_w(self, y, mu, phi=1.0, dispersion=1.0):
        y, mu = self.cshape(y, mu)
        res = self.get_a(y=y, mu=mu, dispersion=dispersion)\
            / (self.dlink(mu)**2 * self.var_func(mu=mu, dispersion=dispersion))
        return res
    
    def get_ehw(self, y, mu, phi=1.0, dispersion=1.0, weights=None):
        weights = self.weights if weights is None else weights
        den = self.variance(mu=mu, phi=phi, dispersion=dispersion) 
        den = den * self.dlink(mu)**2
        w = weights / den
        return w

    def da_dmu(self, y, mu, dispersion=1.0):
        y, mu = self.cshape(y, mu)
        V0 = self.var_func(mu=mu, dispersion=dispersion)
        V1 = self.dvar_dmu(mu=mu, dispersion=dispersion)
        V2 = self.d2var_dmu2(mu=mu, dispersion=dispersion)
        g1, g2, g3 = self.dlink(mu), self.d2link(mu), self.d3link(mu)
        u = (V1 / V0 + g2 / g1)
        v = (V2 / V0 - (V1 / V0)**2 + g3 / g1 - (g2 / g1)**2)
        res = (y - mu) * v - u
        return res

    def d2a_dmu2(self, y, mu, dispersion=1.0):
        y, mu = self.cshape(y, mu)
        V0 = self.var_func(mu=mu, dispersion=dispersion)
        V1 = self.dvar_dmu(mu=mu, dispersion=dispersion)
        V2 = self.d2var_dmu2(mu=mu, dispersion=dispersion)
        V3 = self.d3var_dmu3(mu=mu, dispersion=dispersion)
        g1 = self.dlink(mu)
        g2 = self.d2link(mu)
        g3 = self.d3link(mu)
        g4 = self.d4link(mu)
        u = V2 / V0 - (V1 / V0)**2 + g3 / g1 - (g2 / g1)**2
        v1 = V3 / V0 - 3.0 * (V1 * V2) / V0**2 + 2.0 * (V1 / V0)**3
        v2 = g4 / g1 - 3.0 * (g3 * g2) / g1**2 + 2.0 * (g2 / g1)**3
        res = (y - mu) * (v1 + v2) - 2.0 * u
        return res

    def dw_deta(self, y, mu, dispersion=1.0):
        y, mu = self.cshape(y, mu)
        a0 = self.get_a(y=y, mu=mu, dispersion=dispersion)
        a1 = self.da_dmu(y=y, mu=mu, dispersion=dispersion)
        w = self.get_w(y=y, mu=mu, dispersion=dispersion)
        V0 = self.var_func(mu=mu, dispersion=dispersion)
        V1 = self.dvar_dmu(mu=mu, dispersion=dispersion)
        g1, g2 = self.dlink(mu), self.d2link(mu)
        res = (w / g1) * (a1 / a0 - V1 / V0 - 2.0 * g2 / g1)
        return res

    def d2w_deta2(self, y, mu, phi=1.0, dispersion=1.0):
        y, mu = self.cshape(y, mu)
        w0 = self.get_w(y=y, mu=mu, dispersion=dispersion)
        w1 = self.dw_deta(y=y, mu=mu, dispersion=dispersion)
        a0 = self.get_a(y=y, mu=mu, dispersion=dispersion)
        a1 = self.da_dmu(y=y, mu=mu, dispersion=dispersion)
        a2 = self.d2a_dmu2(y=y, mu=mu, dispersion=dispersion)
        g1, g2, g3 = self.dlink(mu), self.d2link(mu), self.d3link(mu)
        V0 = self.var_func(mu=mu, dispersion=dispersion)
        V1 =  self.dvar_dmu(mu=mu, dispersion=dispersion)
        V2 = self.d2var_dmu2(mu=mu, dispersion=dispersion)
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

    def _loglike(self, y, eta=None, mu=None, T=None, phi=1.0, dispersion=1.0, weights=None):
        mu = self._to_mean(eta=eta, T=T) if mu is None else mu
        weights = self.weights if weights is None else weights
        y, mu = self.cshape(y, mu)
        w = weights / phi
        ll = w * np.power((y - mu), 2) + np.log(phi/weights)
        ll = ll / 2.0
        return ll

    def _full_loglike(self, y, eta=None, mu=None, T=None, phi=1.0, dispersion=1.0, weights=None):
        ll = self._loglike(y, eta, mu, T, phi, dispersion, weights)
        llf = ll + LN2PI / 2.0
        return llf

    def canonical_parameter(self, mu, dispersion=1.0):
        T = mu
        return T

    def cumulant(self, T, dispersion=1.0):
        b = T**2 / 2.0
        return b

    def mean_func(self, T, dispersion=1.0):
        mu = T
        return mu

    def var_func(self, T=None, mu=None, eta=None, dispersion=1.0):
        mu = self._to_mean(eta=eta, T=T) if mu is None else mu
        V = np.ones_like(mu)
        return V
    
    def d2canonical(self, mu, dispersion=1.0):
        res = 0.0*mu+1.0
        return res

    def deviance(self, y, weights=None, T=None, mu=None, eta=None, phi=1.0, dispersion=1.0):
        mu = self._to_mean(eta=eta, T=T) if mu is None else mu
        y, mu = self.cshape(y, mu)
        w = self.weights if weights is None else weights
        d = w * np.power((y - mu), 2.0)
        return d

    def dtau(self, tau, y, mu, reduce=True, weights=None):
        y, mu = self.cshape(y, mu)
        w = self.weights if weights is None else weights
        phi = np.exp(tau)
        g = -(w * np.power((y - mu), 2) / phi - 1) / 2
        if reduce:
            g = np.sum(g)
        return g

    def d2tau(self, tau, y, mu, reduce=True, weights=None):
        y, mu = self.cshape(y, mu)
        w = self.weights if weights is None else weights
        phi = np.exp(tau)
        h = w * np.power((y - mu), 2) / (2 * phi)
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

    def _loglike(self, y, eta=None, mu=None, T=None, phi=1.0, dispersion=1.0, weights=None):
        mu = self._to_mean(eta=eta, T=T) if mu is None else mu
        y, mu = self.cshape(y, mu)
        w = self.weights if weights is None else weights
        num = (y - mu)**2
        den = (y * mu**2 * phi)
        ll = w * num / den + np.log((phi * y**3) / w)
        ll = ll / 2.0
        return ll

    def _full_loglike(self, y, eta=None, mu=None, T=None, phi=1.0, dispersion=1.0, weights=None):
        ll = self._loglike(y, eta, mu, T, phi, dispersion, weights)
        llf = ll + LN2PI / 2.0
        return llf

    def canonical_parameter(self, mu, dispersion=1.0):
        T = 1.0 / (np.power(mu, 2.0))
        return T

    def cumulant(self, T, dispersion=1.0):
        b = -np.sqrt(-2.0*T)
        return b

    def mean_func(self, T, dispersion=1.0):
        mu = 1.0 / np.sqrt(-2.0*T)
        return mu

    def var_func(self, T=None, mu=None, eta=None, dispersion=1.0):
        mu = self._to_mean(eta=eta, T=T) if mu is None else mu
        V = np.power(_check_shape(mu, 1), 3.0)
        return V

    def d2canonical(self, mu, dispersion=1.0):
        res = 3.0 / (np.power(mu, 4))
        return res

    def deviance(self, y, weights=None, T=None, mu=None, eta=None, phi=1.0, dispersion=1.0):
        if mu is None:
            mu = self._to_mean(eta=eta, T=T)
        y, mu = self.cshape(y, mu)
        w = self.weights if weights is None else weights
        d = w * np.power((y - mu), 2.0) / (y * np.power(mu, 2))
        return d

    def dtau(self, tau, y, mu, reduce=True, weights=None):
        y, mu = self.cshape(y, mu)
        w = self.weights if weights is None else weights
        phi = np.exp(tau)
        num = w * np.power((y - mu), 2)
        den = (phi * y * np.power(mu, 2))
        g = -(num / den - 1) / 2
        if reduce:
            g = np.sum(g)
        return g

    def d2tau(self, tau, y, mu, reduce=True, weights=None):
        y, mu = self.cshape(y, mu)
        w = self.weights if weights is None else weights
        phi = np.exp(tau)
        h = w * np.power((y - mu), 2) / (2 * phi * y * mu**2)
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

    def _loglike(self, y, eta=None, mu=None, T=None, phi=1.0, dispersion=1.0, weights=None):
        mu = self._to_mean(eta=eta, T=T) if mu is None else mu
        weights = self.weights if weights is None else weights
        y, mu = self.cshape(y, mu)
        w = weights / phi
        z = w * y / mu
        ll = z - w * np.log(z) + sp.special.gammaln(weights/phi)
        return ll

    def _full_loglike(self, y, eta=None, mu=None, T=None, phi=1.0, dispersion=1.0, weights=None):
        ll = self._loglike(y, eta, mu, T, phi, dispersion, weights)
        llf = ll + np.log(y)
        return llf

    def canonical_parameter(self, mu, dispersion=1.0):
        T = -1.0 / mu
        return T

    def cumulant(self, T, dispersion=1.0):
        b = -np.log(-T)
        return b

    def mean_func(self, T, dispersion=1.0):
        mu = -1 / T
        return mu

    def var_func(self, T=None, mu=None, eta=None, dispersion=1.0):
        mu = self._to_mean(eta=eta, T=T) if mu is None else mu
        V = _check_shape(mu, 1)**2
        return V

    def d2canonical(self, mu, dispersion=1.0):
        res = -2 / (mu**3)
        return res

    def deviance(self, y, weights=None, T=None, mu=None, eta=None, phi=1.0, dispersion=1.0):
        if mu is None:
            mu = self._to_mean(eta=eta, T=T)

        y, mu = self.cshape(y, mu)
        w = self.weights if weights is None else weights
        d = 2 * w * ((y - mu) / mu - np.log(y / mu))
        return d

    def dtau(self, tau, y, mu, reduce=True, weights=None):
        y, mu = self.cshape(y, mu)
        w = self.weights if weights is None else weights
        phi = np.exp(tau)
        T0 = np.log(w * y / (phi * mu))
        T1 = (1 - y / mu)
        T2 = -sp.special.digamma(w / phi)
        g = (w / phi) * (T0 + T1 + T2)
        if reduce:
            g = np.sum(g)
        return g

    def d2tau(self, tau, y, mu, reduce=True, weights=None):
        y, mu = self.cshape(y, mu)
        w = self.weights if weights is None else weights
        phi = np.exp(tau)
        T0 = np.log(w * y / (phi * mu))
        T1 = (2 - y / mu)
        T2 = sp.special.digamma(w / phi)
        T3 = w / phi * sp.special.polygamma(1, w / phi)
        h = w / phi * (T3+T2-T1-T0)
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

    def _loglike(self, y, eta=None, mu=None, T=None, phi=1.0, dispersion=1.0, weights=None):
        mu = self._to_mean(eta=eta, T=T) if mu is None else mu
        y, mu = self.cshape(y, mu)
        w = self.weights/1.0 if weights is None else weights/1.0
        a = 1 / dispersion
        kmu = dispersion * mu
        ypa = y + a
        llk = -w * (y * np.log(kmu) - ypa * np.log(1 + kmu) + loggamma(ypa) - loggamma(a))
        return llk

    def _full_loglike(self, y, eta=None, mu=None, T=None, phi=1.0, dispersion=1.0, weights=None):
        weights = self.weights if weights is None else weights
        ll = self._loglike(y, eta, mu, T, phi, dispersion, weights)
        llf = ll + weights / 1.0 * loggamma(y + 1.0)
        return llf

    def canonical_parameter(self, mu, dispersion=1.0):
        u = mu * dispersion
        T = np.log(u / (1.0 + u))
        return T

    def cumulant(self, T, dispersion=1.0):
        b = (-1.0 / dispersion) * np.log(1 - dispersion * np.exp(T))
        return b

    def mean_func(self, T, dispersion=1.0):
        u = np.exp(T)
        mu = -1.0 / dispersion * (u / (1 - u))
        return mu

    def var_func(self, T=None, mu=None, eta=None, dispersion=1.0):
        mu = self._to_mean(eta=eta, T=T) if mu is None else mu
        V = mu + np.power(mu, 2) * dispersion
        return V

    def d2canonical(self, mu, dispersion=1.0):
        res = -2 * dispersion * mu - 1
        res /= (np.power(mu, 2) * np.power((mu*dispersion+1.0), 2))
        return res

    def deviance(self, y, weights=None, T=None, mu=None, eta=None, phi=1.0, dispersion=1.0):
        mu = self._to_mean(eta=eta, T=T) if mu is None else mu
        y, mu = self.cshape(y, mu)
        w = self.weights if weights is None else weights
        d = np.zeros(y.shape[0])
        ix = (y == 0)
        v = 1.0 / dispersion
        d[ix] = np.log(1 + dispersion * mu[ix]) / dispersion
        yb, mb = y[~ix], mu[~ix]
        u = (yb + v) / (mb + v)
        d[~ix] = (yb*np.log(yb / mb) - (yb + v) * np.log(u))
        d *= 2*w
        return d

    def dtau(self, tau, y, mu, reduce=True, weights=None):
        y, mu = self.cshape(y, mu)
        w = self.weights if weights is None else weights
        dispersion = np.exp(tau)
        T0 = dispersion * (y - mu) / (1 + dispersion * mu)
        T1 = np.log(1 + dispersion * mu)
        T2 = -sp.special.digamma(y + 1 / dispersion)
        T3 = sp.special.digamma(1 / dispersion)
        g = -(w / dispersion) * (T0 + T1 + T2 + T3)
        if reduce:
            g = np.sum(g)
        return g

    def d2tau(self, tau, y, mu, reduce=True, weights=None):
        y, mu = self.cshape(y, mu)
        w = self.weights if weights is None else weights
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
        #a = 1/phi
        #T0 = (2 * phi * mu**2 + mu - y * phi * mu) / (1 + phi * mu)**2
        #T1 = a * np.log(1 + phi * mu)
        #T2 = a * (sp.special.digamma(y + a) - sp.special.digamma(a))
        #T3 = a * (sp.special.polygamma(1, y + a) - sp.special.polygamma(1, a))
        #dtt = (T0 - T1 + T2 + T3)
        h = -w * dtt
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

    def _loglike(self, y, eta=None, mu=None, T=None, phi=1.0, dispersion=1.0, weights=None):
        mu = self._to_mean(eta=eta, T=T) if mu is None else mu
        weights = self.weights if weights is None else weights
        y, mu = self.cshape(y, mu)
        w = weights / phi
        ll = -w * (y * np.log(mu) - mu)
        return ll

    def _full_loglike(self, y, eta=None, mu=None, T=None, phi=1.0, dispersion=1.0, weights=None):
        weights = self.weights if weights is None else weights
        ll = self._loglike(y, eta, mu, T, phi, dispersion, weights)
        llf = ll + weights / phi * sp.special.gammaln(y+1)
        return llf

    def canonical_parameter(self, mu, dispersion=1.0):
        T = np.log(mu)
        return T

    def cumulant(self, T, dispersion=1.0):
        b = np.exp(T)
        return b

    def mean_func(self, T, dispersion=1.0):
        mu = np.exp(T)
        return mu

    def var_func(self, T=None, mu=None, eta=None, dispersion=1.0):
        mu = self._to_mean(eta=eta, T=T) if mu is None else mu
        V = mu
        return V

    def d2canonical(self, mu, dispersion=1.0):
        res = -1 / (mu**2)
        return res

    def deviance(self, y, weights=None, T=None, mu=None, eta=None, phi=1.0, dispersion=1.0):
        mu = self._to_mean(eta=eta, T=T) if mu is None else mu
        y, mu = self.cshape(y, mu)
        w = self.weights if weights is None else weights
        d = np.zeros(y.shape[0])
        ixa = y == 0
        ixb = ~ixa
        d[ixa] = mu[ixa]
        d[ixb] = (y[ixb]*np.log(y[ixb]/mu[ixb]) - (y[ixb] - mu[ixb]))
        d *= 2.0 * w
        return d

    def dvar_dmu(self, mu, dispersion=1.0):
        return np.ones_like(mu)

    def d2var_dmu2(self, mu, dispersion=1.0):
        return np.zeros_like(mu)

    def d3var_dmu3(self, mu, dispersion=1.0):
        return np.zeros_like(mu)

    def d4var_dmu4(self, mu, dispersion=1.0):
        return np.zeros_like(mu)

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

    def _loglike(self, y, eta=None, mu=None, T=None, phi=1.0, dispersion=1.0, weights=None):
        mu = self._to_mean(eta=eta, T=T) if mu is None else mu
        weights = self.weights if weights is None else weights
        y, mu = self.cshape(y, mu)
        w = weights / phi
        ll = -w * (y * np.log(mu) + (1 - y) * np.log(1 - mu))
        return ll

    def _full_loglike(self, y, eta=None, mu=None, T=None, phi=1.0, dispersion=1.0, weights=None):
        weights = self.weights if weights is None else weights
        ll = self._loglike(y, eta, mu, T, phi, dispersion, weights)
        r = weights * _check_shape(_check_np(y), 1)
        llf = ll - logbinom(weights, r)
        return llf

    def canonical_parameter(self, mu, dispersion=1.0):
        u = mu / (1 - mu)
        T = np.log(u)
        return T

    def cumulant(self, T, dispersion=1.0):
        u = 1 + np.exp(T)
        b = np.log(u)
        return b

    def mean_func(self, T, dispersion=1.0):
        u = np.exp(T)
        mu = u / (1 + u)
        return mu

    def var_func(self, T=None, mu=None, eta=None, dispersion=1.0):
        mu = self._to_mean(eta=eta, T=T) if mu is None else mu
        V = mu * (1 - mu)
        return V

    def d2canonical(self, mu, dispersion=1.0):
        res = 1.0/((1 - mu)**2)-1.0/(mu**2)
        return res

    def deviance(self, y, weights=None, T=None, mu=None, eta=None, phi=1.0, dispersion=1.0):
        mu = self._to_mean(eta=eta, T=T) if mu is None else mu
        y, mu = self.cshape(y, mu)
        w = self.weights if weights is None else weights
        ixa = y == 0
        ixb = (y != 0) & (y != 1)
        ixc = y == 1
        d = np.zeros(y.shape[0])
        u = (1 - y)[ixb]
        v = (1 - mu)[ixb]
        d[ixa] = -np.log(1-mu[ixa])
        d[ixc] = -np.log(mu[ixc])
        d[ixb] = y[ixb]*np.log(y[ixb]/mu[ixb]) + u*np.log(u/v)
        return 2*w*d

    def dvar_dmu(self, mu, dispersion=1.0):
        return 1.0 - 2.0 * mu

    def d2var_dmu2(self, mu, dispersion=1.0):
        return np.ones_like(mu) * -2.0

    def d3var_dmu3(self, mu, dispersion=1.0):
        return np.zeros_like(mu)

    def d4var_dmu4(self, mu, dispersion=1.0):
        return np.zeros_like(mu)

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
