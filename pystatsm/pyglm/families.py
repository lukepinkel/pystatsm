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

    def __init__(self, link=IdentityLink, weights=1.0, scale=1.0):

        if not isinstance(link, Link):
            link = link()

        self._link = link
        self.weights = weights
        self.scale = scale

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

    def loglike(self, y, eta=None, mu=None, T=None, scale=1.0):
        return np.sum(self._loglike(y, eta, mu, T, scale))

    def full_loglike(self, y, eta=None, mu=None, T=None, scale=1.0):
        return np.sum(self._full_loglike(y, eta, mu, T, scale))

    def pearson_resid(self, y, eta=None, mu=None, T=None, scale=1.0):
        if mu is None:
            mu = self._to_mean(eta=eta, T=T)
        y, mu = self.cshape(y, mu)
        V = self.var_func(mu=mu)
        w = np.sqrt(self.weights / V)
        r_p = w * (y - mu)
        return r_p

    def pearson_chi2(self, y, eta=None, mu=None, T=None, scale=1.0):
        if mu is None:
            mu = self._to_mean(eta=eta, T=T)
        y, mu = self.cshape(y, mu)
        V = self.var_func(mu=mu)
        w = self.weights / V
        chi2 = np.sum(w * (y - mu)**2)
        return chi2

    def signed_resid(self, y, eta=None, mu=None, T=None, scale=1.0):
        if mu is None:
            mu = self._to_mean(eta=eta, T=T)
        y, mu = self.cshape(y, mu)
        d = self.deviance(y, mu=mu)
        r_s = np.sign(y - mu) * np.sqrt(d)
        return r_s

    def deviance_resid(self, y, eta=None, mu=None, T=None, scale=1.0):
        mu = self._to_mean(eta=eta, T=T) if mu is None else mu
        d = self.deviance(y, mu=mu, scale=scale)
        r = np.sign(y - mu) * np.sqrt(d)
        return r

    def gw(self, y, mu, phi=1.0):
        y, mu = self.cshape(y, mu)
        num = self.weights * (y - mu)
        if self.name == "NegativeBinomial":
            den = self.var_func(mu=mu, scale=phi) * self.dlink(mu)
        else:
            den = self.var_func(mu=mu, scale=phi) * self.dlink(mu) * phi
        res = num / den
        return -res

    def hw(self, y, mu, phi=1.0):
        y, mu = self.cshape(y, mu)
        eta = self.link(mu)
        Vinv = 1.0 / (self.var_func(mu=mu, scale=phi))
        W0 = self.dinv_link(eta)**2
        W1 = self.d2inv_link(eta)
        W2 = self.d2canonical(mu)

        Psc = (y-mu) * (W2*W0+W1*Vinv)
        Psb = Vinv*W0
        res = (Psc - Psb)*self.weights
        return -res/phi

    def get_ghw(self, y, mu, phi=1.0):
        y, mu = self.cshape(y, mu)
        V0, V1 = self.var_func(mu=mu, scale=phi), self.dvar_dmu(mu, scale=phi)
        g1, g2 = self.dlink(mu), self.d2link(mu)
        if self.name == "NegativeBinomial":
            g1V0 = g1 * V0
        else:
            g1V0 = g1 * V0 * phi
        r = y - mu
        a = 1.0 + r * (V1 / V0 + g2 / g1)
        rw = self.weights * r
        gw = -rw / g1V0
        hw = self.weights * a / (g1 * g1V0)
        return gw, hw

    def get_a(self, y, mu, phi=1.0):
        y, mu = self.cshape(y, mu)
        V0, V1 = self.var_func(mu=mu, scale=phi), self.dvar_dmu(mu, scale=phi)
        g1, g2 = self.dlink(mu), self.d2link(mu)
        a = 1.0 + (y - mu) * (V1 / V0 + g2 / g1)
        return a

    def get_g(self, y, mu, phi=1.0):
        y, mu = self.cshape(y, mu)
        res = self.dlink(mu) / self.get_a(y, mu, phi)
        return res

    def get_w(self, y, mu, phi=1.0):
        y, mu = self.cshape(y, mu)
        res = self.get_a(y, mu, phi) / (self.dlink(mu) **
                                        2 * self.var_func(mu=mu, scale=phi))
        return res

    def da_dmu(self, y, mu, phi=1.0):
        y, mu = self.cshape(y, mu)
        V0, V1, V2 = self.var_func(mu=mu, scale=phi), self.dvar_dmu(
            mu, scale=phi), self.d2var_dmu2(mu, scale=phi)
        g1, g2, g3 = self.dlink(mu), self.d2link(mu), self.d3link(mu)
        u = (V1 / V0 + g2 / g1)
        v = (V2 / V0 - (V1 / V0)**2 + g3 / g1 - (g2 / g1)**2)
        res = (y - mu) * v - u
        return res

    def d2a_dmu2(self, y, mu, phi=1.0):
        y, mu = self.cshape(y, mu)
        V0 = self.var_func(mu=mu, scale=phi)
        V1 = self.dvar_dmu(mu, scale=phi)
        V2 = self.d2var_dmu2(mu, scale=phi)
        V3 = self.d3var_dmu3(mu, scale=phi)

        g1 = self.dlink(mu)
        g2 = self.d2link(mu)
        g3 = self.d3link(mu)
        g4 = self.d4link(mu)

        u = V2 / V0 - (V1 / V0)**2 + g3 / g1 - (g2 / g1)**2
        v1 = V3 / V0 - 3.0 * (V1 * V2) / V0**2 + 2.0 * (V1 / V0)**3
        v2 = g4 / g1 - 3.0 * (g3 * g2) / g1**2 + 2.0 * (g2 / g1)**3
        res = (y - mu) * (v1 + v2) - 2.0 * u
        return res

    def dw_deta(self, y, mu, phi=1.0):
        y, mu = self.cshape(y, mu)
        a0, a1 = self.get_a(y, mu, phi), self.da_dmu(y, mu, phi)
        w = self.get_w(y, mu, phi)
        V0, V1 = self.var_func(mu=mu, scale=phi), self.dvar_dmu(mu, scale=phi)
        g1, g2 = self.dlink(mu), self.d2link(mu)
        res = (w / g1) * (a1 / a0 - V1 / V0 - 2.0 * g2 / g1)
        return res

    def d2w_deta2(self, y, mu, phi=1.0):
        y, mu = self.cshape(y, mu)
        w0, w1 = self.get_w(y, mu, phi), self.dw_deta(y, mu, phi)
        a0, a1 = self.get_a(y, mu, phi), self.da_dmu(y, mu, phi)
        a2 = self.d2a_dmu2(y, mu, phi)
        g1, g2, g3 = self.dlink(mu), self.d2link(mu), self.d3link(mu)
        V0, V1, V2 = self.var_func(mu=mu, scale=phi), self.dvar_dmu(
            mu, scale=phi), self.d2var_dmu2(mu, scale=phi)
        t1 = w1**2 / w0
        t2 = w1 * g2 / g1**2

        t3 = a2 / a0 - (a1 / a0)**2 - V2 / V0 + (V1 / V0)**2 \
            - 2.0 * (g3 / g1) + 2.0 * (g2 / g1)**2
        t3 *= w0 / (g1**2)
        res = t1 - t2 + t3
        return res


class Gaussian(ExponentialFamily):

    def __init__(self, link=IdentityLink, weights=1.0, scale=1.0):
        self.name = "Gaussian"
        super().__init__(link, weights, scale)

    def _loglike(self, y, eta=None, mu=None, T=None, scale=1.0):
        if mu is None:
            mu = self._to_mean(eta=eta, T=T)
        y, mu = self.cshape(y, mu)
        w = self.weights / scale
        ll = w * np.power((y - mu), 2) + np.log(scale/self.weights)
        ll = ll / 2.0
        return ll

    def _full_loglike(self, y, eta=None, mu=None, T=None, scale=1.0):
        ll = self._loglike(y, eta, mu, T, scale)
        llf = ll + LN2PI / 2.0
        return llf

    def canonical_parameter(self, mu):
        T = mu
        return T

    def cumulant(self, T):
        b = T**2 / 2.0
        return b

    def mean_func(self, T):
        mu = T
        return mu

    def var_func(self, T=None, mu=None, eta=None, scale=1.0):
        if mu is None:
            mu = self._to_mean(eta=eta, T=T)
        V = np.ones_like(mu)
        return V

    def d2canonical(self, mu):
        res = 0.0*mu+1.0
        return res

    def deviance(self, y, weights=None, T=None, mu=None, eta=None, scale=1.0):
        if mu is None:
            mu = self._to_mean(eta=eta, T=T)
        y, mu = self.cshape(y, mu)
        w = self.weights if weights is None else weights
        d = w * np.power((y - mu), 2.0)
        return d

    def dtau(self, tau, y, mu):
        y, mu = self.cshape(y, mu)
        w = self.weights
        phi = np.exp(tau)
        g = -np.sum(w * np.power((y - mu), 2) / phi - 1) / 2
        return g

    def d2tau(self, tau, y, mu):
        y, mu = self.cshape(y, mu)
        w = self.weights
        phi = np.exp(tau)
        g = np.sum(w * np.power((y - mu), 2) / (2 * phi)) / 2
        return g

    def dvar_dmu(self, mu, scale=1.0):
        return np.zeros_like(mu)

    def d2var_dmu2(self, mu, scale=1.0):
        return np.zeros_like(mu)

    def d3var_dmu3(self, mu, scale=1.0):
        return np.zeros_like(mu)

    def d4var_dmu4(self, mu, scale=1.0):
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

    def rvs(self, mu, scale=1.0, rng=None, seed=None):
        rng = np.random.default_rng(seed) if rng is None else rng
        return rng.normal(loc=mu, scale=np.sqrt(scale))

    def ppf(self, q, mu, scale):
        return sp.stats.norm(loc=mu, scale=np.sqrt(scale)).ppf(q)


class InverseGaussian(ExponentialFamily):

    def __init__(self, link=PowerLink(-2), weights=1.0, scale=1.0):
        self.name = "InverseGaussian"
        super().__init__(link, weights, scale)

    def _loglike(self, y, eta=None, mu=None, T=None, scale=1.0):
        if mu is None:
            mu = self._to_mean(eta=eta, T=T)

        y, mu = self.cshape(y, mu)
        w = self.weights

        num = (y - mu)**2
        den = (y * mu**2 * scale)
        ll = w * num / den + np.log((scale * y**3) / w)
        ll = ll / 2.0
        return ll

    def _full_loglike(self, y, eta=None, mu=None, T=None, scale=1.0):
        ll = self._loglike(y, eta, mu, T, scale)
        llf = ll + LN2PI / 2.0
        return llf

    def canonical_parameter(self, mu):
        T = 1.0 / (np.power(mu, 2.0))
        return T

    def cumulant(self, T):
        b = -np.sqrt(-2.0*T)
        return b

    def mean_func(self, T):
        mu = 1.0 / np.sqrt(-2.0*T)
        return mu

    def var_func(self, T=None, mu=None, eta=None, scale=1.0):
        if mu is None:
            mu = self._to_mean(eta=eta, T=T)

        V = np.power(_check_shape(mu, 1), 3.0)
        return V

    def d2canonical(self, mu):
        res = 3.0 / (np.power(mu, 4))
        return res

    def deviance(self, y, weights=None, T=None, mu=None, eta=None, scale=1.0):
        if mu is None:
            mu = self._to_mean(eta=eta, T=T)

        y, mu = self.cshape(y, mu)
        w = self.weights if weights is None else weights
        d = w * np.power((y - mu), 2.0) / (y * np.power(mu, 2))
        return d

    def dtau(self, tau, y, mu):
        y, mu = self.cshape(y, mu)
        w = self.weights
        phi = np.exp(tau)
        num = w * np.power((y - mu), 2)
        den = (phi * y * np.power(mu, 2))
        g = -np.sum(num / den - 1) / 2
        return g

    def d2tau(self, tau, y, mu):
        y, mu = self.cshape(y, mu)
        w = self.weights
        phi = np.exp(tau)
        g = np.sum(w * np.power((y - mu), 2) / (2 * phi * y * mu**2)) / 2
        return g

    def dvar_dmu(self, mu, scale=1.0):
        return 3.0 * mu**2

    def d2var_dmu2(self, mu, scale=1.0):
        return 6.0 * mu

    def d3var_dmu3(self, mu, scale=1.0):
        return np.ones_like(mu) * 6.0

    def d4var_dmu4(self, mu, scale=1.0):
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

    def rvs(self, mu, scale=1.0, rng=None, seed=None):
        rng = np.random.default_rng(seed) if rng is None else rng
        return rng.wald(mean=mu, scale=np.sqrt(scale))

    def ppf(self, q, mu, scale=1.0):
        return sp.stats.wald(loc=mu, scale=np.sqrt(scale)).ppf(q)


class Gamma(ExponentialFamily):

    def __init__(self, link=ReciprocalLink, weights=1.0, scale=1.0):
        self.name = "Gamma"
        super().__init__(link, weights, scale)

    def _loglike(self, y, eta=None, mu=None, T=None, scale=1.0):
        if mu is None:
            mu = self._to_mean(eta=eta, T=T)

        y, mu = self.cshape(y, mu)
        w = self.weights / scale
        z = w * y / mu
        ll = z - w * np.log(z) + sp.special.gammaln(self.weights/scale)
        return ll

    def _full_loglike(self, y, eta=None, mu=None, T=None, scale=1.0):
        ll = self._loglike(y, eta, mu, T, scale)
        llf = ll + np.log(y)
        return llf

    def canonical_parameter(self, mu):
        T = -1.0 / mu
        return T

    def cumulant(self, T):
        b = -np.log(-T)
        return b

    def mean_func(self, T):
        mu = -1 / T
        return mu

    def var_func(self, T=None, mu=None, eta=None, scale=1.0):
        if mu is None:
            mu = self._to_mean(eta=eta, T=T)

        V = _check_shape(mu, 1)**2
        return V

    def d2canonical(self, mu):
        res = -2 / (mu**3)
        return res

    def deviance(self, y, weights=None, T=None, mu=None, eta=None, scale=1.0):
        if mu is None:
            mu = self._to_mean(eta=eta, T=T)

        y, mu = self.cshape(y, mu)
        w = self.weights if weights is None else weights
        d = 2 * w * ((y - mu) / mu - np.log(y / mu))
        return d

    def dtau(self, tau, y, mu):
        y, mu = self.cshape(y, mu)
        w = self.weights
        phi = np.exp(tau)
        T0 = np.log(w * y / (phi * mu))
        T1 = (1 - y / mu)
        T2 = -sp.special.digamma(w / phi)
        g = (w / phi) * (T0 + T1 + T2)
        return g

    def d2tau(self, tau, y, mu):
        y, mu = self.cshape(y, mu)
        w = self.weights
        phi = np.exp(tau)
        T0 = np.log(w * y / (phi * mu))
        T1 = (2 - y / mu)
        T2 = sp.special.digamma(w / phi)
        T3 = w / phi * sp.special.polygamma(1, w / phi)
        g = np.sum(w / phi * (T3+T2-T1-T0))
        return g

    def dvar_dmu(self, mu, scale=1.0):
        return 2.0 * mu

    def d2var_dmu2(self, mu, scale=1.0):
        return np.ones_like(mu) * 2.0

    def d3var_dmu3(self, mu, scale=1.0):
        return np.zeros_like(mu)

    def d4var_dmu4(self, mu, scale=1.0):
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

    def rvs(self, mu, scale=1.0, rng=None, seed=None):
        rng = np.random.default_rng(seed) if rng is None else rng
        return rng.gamma(shape=1/scale, scale=mu*scale)

    def ppf(self, q, mu, scale=1.0):
        a = 1 / scale
        b = 1 / (mu * scale)
        return sp.stats.gamma(a=a, scale=1/b).ppf(q)


class NegativeBinomial(ExponentialFamily):

    def __init__(self, link=LogLink, weights=1.0, scale=1.0):
        self.name = "NegativeBinomial"
        super().__init__(link, weights, scale)

    def _loglike(self, y, eta=None, mu=None, T=None, scale=1.0):
        if mu is None:
            mu = self._to_mean(eta=eta, T=T)
        phi = scale
        y, mu = self.cshape(y, mu)
        w = self.weights / 1.0
        a = 1 / phi

        kmu = phi * mu
        ypa = y + a
        llk = -w * (y * np.log(kmu) - ypa * np.log(1 + kmu) +
                    loggamma(ypa) - loggamma(a))
        return llk

    def _full_loglike(self, y, eta=None, mu=None, T=None, scale=1.0):
        ll = self._loglike(y, eta, mu, T, scale)
        llf = ll + self.weights / 1.0 * loggamma(y + 1.0)
        return llf

    def canonical_parameter(self, mu, scale=1.0):
        u = mu * scale
        T = np.log(u / (1.0 + u))
        return T

    def cumulant(self, T, scale=1.0):
        b = (-1.0 / scale) * np.log(1 - scale * np.exp(T))
        return b

    def mean_func(self, T, scale=1.0):
        u = np.exp(T)
        mu = -1.0 / scale * (u / (1 - u))
        return mu

    def var_func(self, T=None, mu=None, eta=None, scale=1.0):
        if mu is None:
            mu = self._to_mean(eta=eta, T=T)
        V = mu + np.power(mu, 2) * scale
        return V

    def d2canonical(self, mu, scale=1.0):
        res = -2 * scale * mu - 1
        res /= (np.power(mu, 2) * np.power((mu*scale+1.0), 2))
        return res

    def deviance(self, y, weights=None, T=None, mu=None, eta=None, scale=1.0):
        if mu is None:
            mu = self._to_mean(eta=eta, T=T)

        y, mu = self.cshape(y, mu)
        w = self.weights if weights is None else weights
        d = np.zeros(y.shape[0])
        ix = (y == 0)
        v = 1.0 / scale
        d[ix] = np.log(1 + scale * mu[ix]) / scale
        yb, mb = y[~ix], mu[~ix]
        u = (yb + v) / (mb + v)
        d[~ix] = (yb*np.log(yb / mb) - (yb + v) * np.log(u))
        d *= 2*w
        return d

    def dtau(self, tau, y, mu):
        y, mu = self.cshape(y, mu)
        w = self.weights
        phi = np.exp(tau)
        T0 = phi * (y - mu) / (1 + phi * mu)
        T1 = np.log(1 + phi * mu)
        T2 = -sp.special.digamma(y + 1 / phi)
        T3 = sp.special.digamma(1 / phi)
        dt = (w / phi) * (T0 + T1 + T2 + T3)
        return -dt

    def d2tau(self, tau, y, mu):
        y, mu = self.cshape(y, mu)
        w = self.weights
        phi = np.exp(tau)
        a = 1.0 / phi
        ypa = y + a
        u = 1 + phi * mu
        denom = -y * phi * mu + mu + 2 * phi * mu**2
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
        dtt = -np.sum(w * dtt)
        return dtt

    def dvar_dmu(self, mu, scale=1.0):
        return 2.0 * scale * mu + 1.0

    def d2var_dmu2(self, mu, scale=1.0):
        return 2.0 * scale

    def d3var_dmu3(self, mu, scale=1.0):
        return np.zeros_like(mu)

    def d4var_dmu4(self, mu, scale=1.0):
        return np.zeros_like(mu)

    def llscale(self, phi, y):
        return None

    def dllscale(self, phi, y):
        return None

    def d2llscale(self, phi, y):
        return None

    def rvs(self, mu, scale=1.0, rng=None, seed=None):
        rng = np.random.default_rng(seed) if rng is None else rng
        var = mu + scale * mu**2
        n = - mu**2 / (mu - var)
        p = mu / var
        y = rng.negative_binomial(n=n, p=p)
        return y


class Poisson(ExponentialFamily):

    def __init__(self, link=LogLink, weights=1.0, scale=1.0):
        self.name = "Poisson"
        super().__init__(link, weights, scale)

    def _loglike(self, y, eta=None, mu=None, T=None, scale=1.0):
        if mu is None:
            mu = self._to_mean(eta=eta, T=T)

        y, mu = self.cshape(y, mu)
        w = self.weights / scale

        ll = -w * (y * np.log(mu) - mu)
        return ll

    def _full_loglike(self, y, eta=None, mu=None, T=None, scale=1.0):
        ll = self._loglike(y, eta, mu, T, scale)
        llf = ll + self.weights / scale * sp.special.gammaln(y+1)
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

    def var_func(self, T=None, mu=None, eta=None, scale=1.0):
        if mu is None:
            mu = self._to_mean(eta=eta, T=T)

        V = mu
        return V

    def d2canonical(self, mu, dispersion=1.0):
        res = -1 / (mu**2)
        return res

    def deviance(self, y, weights=None, T=None, mu=None, eta=None, scale=1.0):
        if mu is None:
            mu = self._to_mean(eta=eta, T=T)

        y, mu = self.cshape(y, mu)
        w = self.weights if weights is None else weights
        d = np.zeros(y.shape[0])
        ixa = y == 0
        ixb = ~ixa
        d[ixa] = mu[ixa]
        d[ixb] = (y[ixb]*np.log(y[ixb]/mu[ixb]) - (y[ixb] - mu[ixb]))
        d *= 2.0 * w
        return d

    def dvar_dmu(self, mu, scale=1.0):
        return np.ones_like(mu)

    def d2var_dmu2(self, mu, scale=1.0):
        return np.zeros_like(mu)

    def d3var_dmu3(self, mu, scale=1.0):
        return np.zeros_like(mu)

    def d4var_dmu4(self, mu, scale=1.0):
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

    def __init__(self, link=LogitLink, weights=1.0, scale=1.0):
        self.name = "Binomial"
        super().__init__(link, weights, scale)

    def _loglike(self, y, eta=None, mu=None, T=None, scale=1.0):
        if mu is None:
            mu = self._to_mean(eta=eta, T=T)

        y, mu = self.cshape(y, mu)
        w = self.weights / scale

        ll = -w * (y * np.log(mu) + (1 - y) * np.log(1 - mu))
        return ll

    def _full_loglike(self, y, eta=None, mu=None, T=None, scale=1.0):
        ll = self._loglike(y, eta, mu, T, scale)
        r = self.weights * _check_shape(_check_np(y), 1)
        llf = ll - logbinom(self.weights, r)
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

    def var_func(self, T=None, mu=None, eta=None, scale=1.0):
        if mu is None:
            mu = self._to_mean(eta=eta, T=T)

        V = mu * (1 - mu)
        return V

    def d2canonical(self, mu, dispersion=1.0):
        res = 1.0/((1 - mu)**2)-1.0/(mu**2)
        return res

    def deviance(self, y, weights=None, T=None, mu=None, eta=None, scale=1.0):
        if mu is None:
            mu = self._to_mean(eta=eta, T=T)

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

    def dvar_dmu(self, mu, scale=1.0):
        return 1.0 - 2.0 * mu

    def d2var_dmu2(self, mu):
        return np.ones_like(mu) * -2.0

    def d3var_dmu3(self, mu, scale=1.0):
        return np.zeros_like(mu)

    def d4var_dmu4(self, mu, scale=1.0):
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
