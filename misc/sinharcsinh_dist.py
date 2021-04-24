# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 22:01:33 2021

@author: lukepinkel
"""

import numpy as np
import scipy as sp
import scipy.stats

TWOPI = 2.0 * np.pi
LN2PI = np.log(TWOPI)



def _logpdf(x, m=0.0, s=1.0, v=0.0, t=1.0):
    z = (x - m) / s
    y = np.arcsinh(z) * t - v
    r, c, w = np.sinh(y), np.cosh(y), np.sqrt(1.0 + z**2)
    logp = np.log(t) + np.log(c) - np.log(s) - np.log(w) - r**2 / 2.0 - LN2PI / 2.0
    return logp

def _pdf(x, m=0.0, s=1.0, v=0.0, t=1.0):
    return np.exp(_logpdf(x, m=m, s=s, v=v, t=t))


def _cdf(x, m=0.0, s=1.0, v=0.0, t=1.0):
    z = (x - m) / s
    r = np.sinh(t * np.arcsinh(z) - v)
    return sp.special.ndtr(r)


def _qtf(q, m=0.0, s=1.0, v=0.0, t=1.0):
    y = sp.special.ndtri(q)
    z = np.sinh((np.arcsinh(y) + v) / t)
    x = s * z + m
    return x

def _rvs(random_state, m=0.0, s=1.0, v=0.0, t=1.0, size=None):
    u = random_state.uniform(low=0.0, high=1.0, size=size)
    y = _qtf(u, m=m, s=s, v=v, t=t)
    return y


def _dL_dmu(x, m, s, v, t):
    z = (x - m) / s
    y = np.arcsinh(z) * t - v
    r, c, w = np.sinh(y), np.cosh(y), np.sqrt(1.0 + z**2)
    rt, sw = r * t, s * w
    g = (rt * c - rt / c) / sw + z / (sw * w)
    return g

def _d2L_dmu2(x, m, s, v, t):
    z = (x - m) / s
    y = np.arcsinh(z) * t - v
    r, c, w = np.sinh(y), np.cosh(y), np.sqrt(1.0 + z**2)
    t2, s2 = t**2, s**2
    w2 = w**2
    w3 = w2 * w
    w4 = w3 * w
    c2, r2 = c**2, r**2
    f1 = (t * r * z * (c2 - 1)) / (s2 * w3 * c)
    f2 =  (2.0 * z**2) / (s2 * w4)
    f3 = (t2 - t2 * (r2 + c2)-1.0) / (s2 * w2)
    f4 = -(t2 * r2) / (s2 * w2 * c2)
    H = f1 + f2 + f3 + f4
    return H

def _d2L_dmu_dsigma(x, m, s, v, t):
    z = (x - m) / s
    y = np.arcsinh(z) * t - v
    r, c, w = np.sinh(y), np.cosh(y), np.sqrt(1.0 + z**2)
    
    f11 =-(t * r * c) / (s**2 * w)
    f12 =-(t**2 * r**2 * z) / (s**2 * w**2)
    f13 =-(t**2 * c**2 * z) / (s**2 * w**2)
    f14 = (t * r * c * z**2) / (s**2 * w**3)
    f1 = f11 + f12 + f13 + f14
    
    f21 = (t * r) / (c * w *s**2)
    f22 =-(t**2 * r**2 * z) / (c**2 * w**2 * s**2)
    f23 = (t**2 * z) / (w**2 * s**2)
    f24 =-(t * z**2 * r) / (c * w**3 * s**2)
    f2 = f21 + f22 + f23 + f24
    
    f31 = (2 * z**3) / (w**4 * s**2)
    f32 =-(2 * z) / (w**2 * s**2)
    f3 = f31 + f32
    
    f = f1 + f2 + f3
    return f

def _d2L_dmu_dnu(x, m, s, v, t):
    z = (x - m) / s
    y = np.arcsinh(z) * t - v
    r, c, w = np.sinh(y), np.cosh(y), np.sqrt(1.0 + z**2)
    
    f11 =-(t * r**2) / (s * w)
    f12 =-(t * c**2) / (s * w)
    f1 = f11 + f12
    
    f21 = -(t * r**2) / (c**2 * s * w)
    f22 = (t) / (s * w)
    f2 = f21 + f22
    
    #f3 = 0
    
    f = f1 + f2
    return f

def _d2L_dmu_dtau(x, m, s, v, t):
    z = (x - m) / s
    u = np.arcsinh(z)
    y = u * t - v
    r, c, w = np.sinh(y), np.cosh(y), np.sqrt(1.0 + z**2)
    
    f11 = (u * t * r**2) / (s * w)
    f12 = (u * t * c**2) / (s * w)
    f13 = (c * r) / (s * w)
    f1 = f11 + f12 + f13
    
    f21 = (u * t * r**2) / (s * w * c**2)
    f22 =-(r) / (s * w * c)
    f23 =-(u * t) / (s * w)
    f2 = f21 + f22 + f23
    
    #f3 = 0
    
    f = f1 + f2
    return f

def _dL_dsigma(x, m, s, v, t):
    z = (x - m) / s
    y = np.arcsinh(z) * t - v
    r, c, w = np.sinh(y), np.cosh(y), np.sqrt(1.0 + z**2)
    f1 = (z * r * t * c) / (w * s)
    f2 =-(z * r * t) / (w * s * c)
    f3 = (z**2) / (w**2 * s)
    f4 =-(1.0  / s)
    g = f1 + f2 + f3 + f4
    return g

def _d2L_dsigma2(x, m, s, v, t):
    z = (x - m) / s
    y = np.arcsinh(z) * t - v
    r, c, w = np.sinh(y), np.cosh(y), np.sqrt(1.0 + z**2)
    
    f11 = (-2.0 * t * z * c * r) / (w * s**2)
    f12 =-((t*z*r)**2) / (w**2 * s**2)
    f13 =-((t*z*c)**2) / (w**2 * s**2)
    f14 = (t * c * r * z**3) / (w**3 * s**2)
    f1 = f11 + f12 + f13 + f14
    
    f21 = (2 * t * z * r) / (w * c * s**2)
    f22 =-((t * z * r)**2) / ((w * c * s)**2)
    f23 = ((t * z) / (w * s))**2
    f24 =-(t * r * z**3) / (w**3 * c * s**2)
    f2 = f21 + f22 + f23 + f24
    
    f31 = (2 * z**4) / (w**4 * s**2)
    f32 =-(3 * z**2) / (w**2 * s**2)
    f3 = f31 + f32
    
    f4 = 1 / s**2
    
    f = f1 + f2 + f3 + f4
    return f

def _d2L_dsigma_dnu(x, m, s, v, t):
    z = (x - m) / s
    y = np.arcsinh(z) * t - v
    r, c, w = np.sinh(y), np.cosh(y), np.sqrt(1.0 + z**2)
    
    f11 =-(t * z * r**2) / (s * w)
    f12 =-(t * z * c**2) / (s * w)
    f1 = f11 + f12
    
    f21 = (t * z) / (s * w)
    f22 =-(t * r**2 * z) / (s * w * c**2)
    f2 = f21 + f22
    
    #f3 = 0
    #f4 = 0
    f = f1 + f2
    return f

def _d2L_dsigma_dtau(x, m, s, v, t):
    z = (x - m) / s
    u = np.arcsinh(z)
    y = u * t - v
    r, c, w = np.sinh(y), np.cosh(y), np.sqrt(1.0 + z**2)
    
    f11 = (z * u * t * r**2) / (s * w)
    f12 = (z * u * t * c**2) / (s * w)
    f13 = (z * c * r) / (s * w)
    f1 = f11 + f12 + f13
    
    f21 = (z * u * t * r**2) / (s * w * c**2)
    f22 =-(z * r) / (s * w * c)
    f23 =-(z * u * t) / (s * w)
    f2 = f21 + f22 + f23
    
    f = f1 + f2
    return f
    

def _dL_dnu(x, m, s, v, t):
    z = (x - m) / s
    y = np.arcsinh(z) * t - v
    r, c = np.sinh(y), np.cosh(y)
    g = r * c - r / c
    return g

def _d2L_dnu2(x, m, s, v, t):
    z = (x - m) / s
    y = np.arcsinh(z) * t - v
    r, c = np.sinh(y), np.cosh(y)
    g = 1 - r**2 / c**2 - r**2 - c**2
    return g

def _d2L_dnu_dtau(x, m, s, v, t):
    z = (x - m) / s
    u = np.arcsinh(z)
    y = t * u - v
    r, c = np.sinh(y), np.cosh(y)
    
    f1 = (u * r**2) / (c**2)
    f2 = u * r**2
    f3 = u * c**2
    f4 =-u
    
    f = f1 + f2 + f3 + f4
    return f

def _dL_dtau(x, m, s, v, t):
    z = (x - m) / s
    u = np.arcsinh(z)
    y = t * u - v
    r, c = np.sinh(y), np.cosh(y)
    g = ((r / c) - (r * c)) * u + 1.0 / t
    return g

def _d2L_dtau2(x, m, s, v, t):
    z = (x - m) / s
    u = np.arcsinh(z)
    y = t * u - v
    r, c = np.sinh(y), np.cosh(y)
    
    f1 = -(u**2 * r**2) / c**2
    f2 = -(u**2 * r**2)
    f3 = -(u**2 * c**2)
    f4 = u**2 - 1 / t**2 
    f = f1 + f2 + f3 + f4
    return f


def _grad_elementwise(theta, x):
    m, s, v, t = theta[0], theta[1], theta[2], theta[3]
    m1 = np.expand_dims(np.asarray(_dL_dmu(x, m, s, v, t)), -1)
    s1 = np.expand_dims(np.asarray(_dL_dsigma(x, m, s, v, t)), -1)
    v1 = np.expand_dims(np.asarray(_dL_dnu(x, m, s, v, t)), -1)
    t1 = np.expand_dims(np.asarray(_dL_dtau(x, m, s, v, t)), -1)
    g = np.concatenate([m1, s1, v1, t1], axis=-1)
    return g


def _hess_elementwise(theta, x):
    m, s, v, t = theta[0], theta[1], theta[2], theta[3]
    m2 = np.expand_dims(np.asarray(_d2L_dmu2(x, m, s, v, t)), -1)
    ms = np.expand_dims(np.asarray(_d2L_dmu_dsigma(x, m, s, v, t)), -1)
    mv = np.expand_dims(np.asarray(_d2L_dmu_dnu(x, m, s, v, t)), -1)
    mt = np.expand_dims(np.asarray(_d2L_dmu_dtau(x, m, s, v, t)), -1)
    s2 = np.expand_dims(np.asarray(_d2L_dsigma2(x, m, s, v, t)), -1)
    sv = np.expand_dims(np.asarray(_d2L_dsigma_dnu(x, m, s, v, t)), -1)
    st = np.expand_dims(np.asarray(_d2L_dsigma_dtau(x, m, s, v, t)), -1)
    v2 = np.expand_dims(np.asarray(_d2L_dnu2(x, m, s, v, t)), -1)
    vt = np.expand_dims(np.asarray(_d2L_dnu_dtau(x, m, s, v, t)), -1)
    t2 = np.expand_dims(np.asarray(_d2L_dtau2(x, m, s, v, t)), -1)

    r1 = np.expand_dims(np.concatenate([m2, ms, mv, mt], axis=-1), -1)
    r2 = np.expand_dims(np.concatenate([ms, s2, sv, st], axis=-1), -1)
    r3 = np.expand_dims(np.concatenate([mv, sv, v2, vt], axis=-1), -1)
    r4 = np.expand_dims(np.concatenate([mt, st, vt, t2], axis=-1), -1)
    H = np.concatenate([r1, r2, r3, r4], axis=-1)
    return H


def _grad(theta, x):
    m, s, v, t = theta[0], theta[1], theta[2], theta[3]
    m1 = np.asarray(_dL_dmu(x, m, s, v, t)).sum()
    s1 = np.asarray(_dL_dsigma(x, m, s, v, t)).sum()
    v1 = np.asarray(_dL_dnu(x, m, s, v, t)).sum()
    t1 = np.asarray(_dL_dtau(x, m, s, v, t)).sum()
    g = np.array([m1, s1, v1, t1])
    return g

def _hess(theta, x):
    m, s, v, t = theta[0], theta[1], theta[2], theta[3]
    m2 = np.sum(np.asarray(_d2L_dmu2(x, m, s, v, t)))
    ms = np.sum(np.asarray(_d2L_dmu_dsigma(x, m, s, v, t)))
    mv = np.sum(np.asarray(_d2L_dmu_dnu(x, m, s, v, t)))
    mt = np.sum(np.asarray(_d2L_dmu_dtau(x, m, s, v, t)))
    s2 = np.sum(np.asarray(_d2L_dsigma2(x, m, s, v, t)))
    sv = np.sum(np.asarray(_d2L_dsigma_dnu(x, m, s, v, t)))
    st = np.sum(np.asarray(_d2L_dsigma_dtau(x, m, s, v, t)))
    v2 = np.sum(np.asarray(_d2L_dnu2(x, m, s, v, t)))
    vt = np.sum(np.asarray(_d2L_dnu_dtau(x, m, s, v, t)))
    t2 = np.sum(np.asarray(_d2L_dtau2(x, m, s, v, t)))

    H = np.array([[m2, ms, mv, mt],
                  [ms, s2, sv, st],
                  [mv, sv, v2, vt],
                  [mt, st, vt, t2]])
    return H

def _pq(q):
    v1, v2 = (q + 1.0) / 2.0, (q - 1.0) / 2.0
    k = np.exp(0.25) / np.sqrt(8.0 * np.pi)
    return k * (sp.special.kv(v1, 0.25) + sp.special.kv(v2, 0.25))

def _mean(m, s, v, t):
    dist_mean = m + s * np.sinh(v / t) * _pq(1.0 / t)
    return dist_mean

def _variance(m, s, v, t):
    s2 = s**2
    f1 = (np.cosh(2.0 * v / t) * _pq(2.0 / t) - 1.0) * s2 / 2.0
    f2 =-s2 * (np.sinh(v / t) * _pq(1.0 / t))**2
    dist_var = f1 + f2
    return dist_var
    

###############################################################
import seaborn as sns
import matplotlib.pyplot as plt
from pystats.utilities import numerical_derivs

mu, sigma, nu, tau = 10.0, 2.5, 3.0, 1.5
rng = np.random.default_rng(1234)

y = _rvs(rng, mu, sigma, nu, tau, (10_000_000, ))

x = np.linspace(y.min(), y.max(), 2000)
p = np.exp(_logpdf(x, mu, sigma, nu, tau))

fig, ax = plt.subplots()
sns.histplot(y, stat="density", edgecolor="white", ax=ax, alpha=0.5)
sns.kdeplot(y, ax=ax, ls='--')
ax.plot(x, p)


p_approx = numerical_derivs.finite_diff(lambda x: _cdf(x, mu, sigma, nu, tau), x)
np.allclose(p_approx, p)
_qtf(_cdf(mu+1, mu, sigma, nu, tau), mu, sigma, nu, tau)
_qtf(_cdf(mu-1, mu, sigma, nu, tau), mu, sigma, nu, tau)

p_ext = (_cdf(mu+2*sigma, mu, sigma, nu, tau) - _cdf(mu-2*sigma, mu, sigma, nu, tau))
p_sim = np.mean((y>(mu - 2*sigma)) * (y < (mu + 2*sigma)))



np.allclose(numerical_derivs.finite_diff(lambda x: _logpdf(y[0], x, sigma, nu, tau), mu),
            _dL_dmu(y[0], mu, sigma, nu, tau))

np.allclose(numerical_derivs.finite_diff(lambda x: _logpdf(y[0], mu, x, nu, tau), sigma),
            _dL_dsigma(y[0], mu, sigma, nu, tau))

np.allclose(numerical_derivs.finite_diff(lambda x: _logpdf(y[0], mu, sigma, x, tau), nu),
            _dL_dnu(y[0], mu, sigma, nu, tau))

np.allclose(numerical_derivs.finite_diff(lambda x: _logpdf(y[0], mu, sigma, nu, x), tau),
            _dL_dtau(y[0], mu, sigma, nu, tau))


numerical_derivs.finite_diff(lambda x: _logpdf(y[0], x, sigma, nu, tau), mu, order=2)
numerical_derivs.finite_diff(lambda x: _dL_dmu(y[0], x, sigma, nu, tau), mu, order=1)
_d2L_dmu2(y[0], mu, sigma, nu, tau)


numerical_derivs.finite_diff(lambda x:    _logpdf(y[0], mu, x, nu, tau), sigma, order=2)
numerical_derivs.finite_diff(lambda x: _dL_dsigma(y[0], mu, x, nu, tau), sigma, order=1)
_d2L_dsigma2(y[0], mu, sigma, nu, tau)


numerical_derivs.finite_diff(lambda x: _logpdf(y[0], mu, sigma, x, tau), nu, order=2)
numerical_derivs.finite_diff(lambda x: _dL_dnu(y[0], mu, sigma, x, tau), nu, order=1)
_d2L_dnu2(y[0], mu, sigma, nu, tau)


numerical_derivs.finite_diff(lambda x:  _logpdf(y[0], mu, sigma, nu, x), tau, order=2)
numerical_derivs.finite_diff(lambda x: _dL_dtau(y[0], mu, sigma, nu, x), tau, order=1)
_d2L_dtau2(y[0], mu, sigma, nu, tau)


numerical_derivs.finite_diff(lambda x: _dL_dmu(y[0], mu, x, nu, tau), sigma, order=1)
_d2L_dmu_dsigma(y[0], mu, sigma, nu, tau)


numerical_derivs.finite_diff(lambda x: _dL_dmu(y[0], mu, sigma, x, tau), nu, order=1)
_d2L_dmu_dnu(y[0], mu, sigma, nu, tau)


numerical_derivs.finite_diff(lambda x: _dL_dmu(y[0], mu, sigma, nu, x), tau, order=1)
_d2L_dmu_dtau(y[0], mu, sigma, nu, tau)


numerical_derivs.finite_diff(lambda x: _dL_dsigma(y[0], mu, sigma, x, tau), nu, order=1)
_d2L_dsigma_dnu(y[0], mu, sigma, nu, tau)


numerical_derivs.finite_diff(lambda x: _dL_dsigma(y[0], mu, sigma, nu, x), tau, order=1)
_d2L_dsigma_dtau(y[0], mu, sigma, nu, tau)

numerical_derivs.finite_diff(lambda x: _dL_dnu(y[0], mu, sigma, nu, x), tau, order=1)
_d2L_dnu_dtau(y[0], mu, sigma, nu, tau)


theta = np.asarray([mu, sigma, nu, tau])
y = _rvs(rng, mu, sigma, nu, tau, (1_000, ))

g_elem = _grad_elementwise(theta, y)
H_elem = _hess_elementwise(theta, y)

g = _grad(theta, y)
H = _hess(theta, y)

numerical_derivs.so_gc_cd(lambda x: _grad(x, y), theta)
numerical_derivs.fo_fc_cd(lambda x: np.sum(_logpdf(y, x[0], x[1], x[2], x[3])), theta)

np.allclose(numerical_derivs.fo_fc_cd(lambda x: np.sum(_logpdf(y, x[0], x[1], x[2], x[3])), theta),
            g)

np.allclose(numerical_derivs.so_gc_cd(lambda x: _grad(x, y), theta), H)








