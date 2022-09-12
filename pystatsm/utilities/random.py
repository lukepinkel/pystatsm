# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 17:08:22 2021

@author: lukepinkel
"""
import numba
import numpy as np
from math import erf
from .data_utils import corr_nb as _corr, csd as _csd
from .tf_utils import spherical_uniform, clockwise_spiral_fill_triangular
SQRT2 = np.sqrt(2.0)

@numba.jit(nopython=True)
def vine_corr(d, eta=1, beta=None, seed=None, min_eig=False):
    if beta is None:
        beta = eta + (d - 1) / 2.0
    if seed is not None:
        np.random.seed(seed)
    P = np.zeros((d, d))
    S = np.eye(d)
    for k in range(d-1):
        beta -= 0.5
        for i in range(k+1, d):
            P[k, i] = np.random.beta(beta, beta)
            P[k, i] = (P[k, i] - 0.5)*2.0
            p = P[k, i]
            for l in range(k-1, 1, -1):
                p = p * np.sqrt((1 - P[l, i]**2)*(1 - P[l, k]**2)) + P[l, i]*P[l, k]
            S[k, i] = p
            S[i, k] = p
    if min_eig:
        u, V = np.linalg.eigh(S)
        umin = np.min(u[u>0])
        u[u<0] = [umin*0.5**(float(i+1)/len(u[u<0])) for i in range(len(u[u<0]))]
        V = np.ascontiguousarray(V)
        S = V.dot(np.diag(u)).dot(np.ascontiguousarray(V.T))
        v = np.diag(S)
        v = np.diag(1/np.sqrt(v))
        S = v.dot(S).dot(v)
    return S


@numba.jit(nopython=True)
def onion_corr(d, eta=1, beta=None):
    if beta is None:
        beta = eta + (d - 2) / 2.0
    u = np.random.beta(beta, beta)
    r12 = 2 * u  - 1
    S = np.array([[1, r12], [r12, 1]])
    I = np.array([[1.0]])
    for i in range(3, d+1):
        beta -= 0.5
        r = np.sqrt(np.random.beta((i - 1) / 2, beta))
        theta = np.random.normal(0, 1, size=(i-1, 1))
        theta/= np.linalg.norm(theta)
        w = r * theta
        c, V = np.linalg.eig(S)
        R = (V * np.sqrt(c)).dot(V.T)
        q = R.dot(w)
        S = np.concatenate((np.concatenate((S, q), axis=1),
                            np.concatenate((q.T, I), axis=1)), axis=0)
    return S


@numba.jit(nopython=True)
def exact_rmvnorm(S, n=1000, mu=None, seed=None):
    if seed is not None:
        np.random.seed(seed)
    p = S.shape[0]
    U, d, _ = np.linalg.svd(S)
    d = d.reshape(1, -1)
    L = U * d**0.5
    X = _csd(np.random.normal(0.0, 1.0, size=(n, p)))
    R = _corr(X)
    L = L.dot(np.linalg.inv(np.linalg.cholesky(R)))
    X = X.dot(L.T)
    if mu is not None:
        X = X + mu
    return X

@numba.jit(nopython=True)
def wishart(df, V):
    n = V.shape[0]
    T = np.zeros((n, n))
    for i in range(n):
        T[i, i] = np.sqrt(np.random.chisquare(df-(i+1)+1))
        for j in range(i):
            T[i, j] = np.random.normal(0.0, 1.0)
    L = np.linalg.cholesky(V)
    A = L.dot(T)
    W = A.dot(A.T)
    return W

@numba.jit(nopython=True)
def invwishart(df, V):
    n = V.shape[0]
    T = np.zeros((n, n))
    for i in range(n):
        T[i, i] = np.sqrt(np.random.chisquare(df-(i+1)+1))
        for j in range(i):
            T[i, j] = np.random.normal(0.0, 1.0)
    L = np.linalg.cholesky(V)
    A = L.dot(T)
    W = A.dot(A.T)
    IW = np.linalg.inv(W)
    return IW

def r_invwishart(df, V):
    Vinv = np.linalg.inv(V)
    return invwishart(df, Vinv)

def r_invgamma(df, scale):
    return 1.0/np.random.gamma(df, scale=1/scale)
    



@numba.jit(nopython=True)
def norm_cdf(x):
    y = (1.0 + erf(x / SQRT2)) / 2.0
    return y
    

#Adapted from MCMCglmm
@numba.jit(nopython=True)
def scalar_truncnorm(mu, sd, lb, ub):
    sample = 1
    if (lb < -1e16) or (ub>1e16):
        if (lb < -1e16) and (ub>1e16):
            z = np.random.normal(mu, sd)
        else:
            if ub > 1e16:
                tr = (lb - mu) / sd
            else:
                tr = (mu - ub) / sd
            if tr < 0:
                while sample==1:
                    z = np.random.normal(0.0, 1.0)
                    if z>tr:
                        sample = 0
            else:
                alpha = (tr + np.sqrt((tr * tr) + 4.0)) / 2.0
                while sample==1:
                    z = np.random.exponential(scale=1/alpha) + tr
                    pz = -((alpha - z) * (alpha - z) / 2.0)
                    u = -np.random.exponential(scale=1.0)
                    if (u<=pz):
                        sample = 0
    else:
        sl = (lb - mu) / sd
        su = (ub - mu) / sd
    
        tr = norm_cdf(su) - norm_cdf(sl)
        
        if tr>0.5:
            while sample==1:
                z = np.random.normal(0.0, 1.0)
                if (z>sl) and (z<su):
                    sample = 0
        else:
            while sample==1:
                z = np.random.uniform(sl, su)
                if (sl<=0.0) and (0.0<=su):
                    pz = -z*z / 2.0
                else:
                    if su<0.0:
                        pz = (su * su - z * z) / 2.0
                    else:
                        pz = (sl * sl - z * z) / 2.0
                u = -np.random.exponential(scale=1.0)
                if u<pz:
                    sample = 0
    if lb<-1e16:
        return mu-z*sd
    else:
        return z*sd+mu
            
@numba.jit(nopython=True)      
def trnorm(mu, sd, lb, ub):
    n = len(mu)
    z = np.zeros((n, ), dtype=numba.float32)
    for i in range(n):
        z[i] = scalar_truncnorm(mu[i], sd[i], lb[i], ub[i])
    return z

def students_t(loc, scale, nu=1, size=None, rng=None):
     rng = np.random.default_rng() if rng is None else rng
     x = loc + rng.standard_t(df=nu, size=size) * scale
     return x

def cauchy(loc, scale, size=None, rng=None):
     rng = np.random.default_rng() if rng is None else rng
     x = loc + rng.standard_cauchy(size=size) * scale
     return x
    
def multivariate_t(mean, cov, nu=1, size=None, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    u = np.sqrt(nu / rng.chisquare(df=nu, size=size))[:, np.newaxis]
    Y = rng.multivariate_normal(mean=mean, cov=cov, size=size)
    X = mean + u * Y
    return X
    
def r_lkj_cholesky(eta=1.0, n=1, dim=1, rng=None, seed=None):
    rng = np.random.default_rng(seed) if rng is None else rng
    
    eta = np.atleast_1d(eta)
    batch_shape = np.concatenate([[n], np.shape(eta)], axis=0).astype(np.int32)
    beta = eta + (dim - 2.) / 2.
    dimension_range = np.arange(1., dim)
    a = dimension_range / 2.
    b = beta[..., np.newaxis] - (dimension_range - 1) / 2.
    norm = rng.beta(a=a, b=b, size=[n] + list(b.shape))
    distance = np.sqrt(norm)[..., np.newaxis]
    d = dim - 1
    rows = []
    paddings_prepend = [[0, 0]] * len(batch_shape)
    for n in range(1, min(d, 2) + 1):
        u = spherical_uniform(shape=batch_shape, dim=n, rng=rng)
        row = np.pad(u, paddings_prepend + [[0, d - n]], constant_values=0.)
        rows.append(row)
    samples = np.stack(rows, axis=-2)
    if d>2:
        normal_shape = np.concatenate([batch_shape, [d * (d + 1) // 2 - 3]], axis=0).astype(np.int32)
        j1 = np.ones(np.concatenate([batch_shape, [1]], axis=0).astype(np.int32))
        j2 = np.ones(np.concatenate([batch_shape, [2]], axis=0).astype(np.int32))
        u = rng.normal(size=normal_shape)
        normal_samples = np.concatenate([u[..., :d], j1,  u[..., d:(2 * d - 1)],
                                         j2, u[..., (2 * d - 1):]], axis=-1)
        mat_samples = clockwise_spiral_fill_triangular(normal_samples, upper=False)[..., 2:, :]
        remaining_rows = mat_samples / np.linalg.norm(mat_samples, ord=2, axis=-1, keepdims=True)
        samples = np.concatenate([samples, remaining_rows], axis=-2)
    
    direction = samples
    raw_correlation = distance * direction
    
    paddings_prepend = [[0, 0]] * len(batch_shape)
    diag = np.pad(np.sqrt(1. - norm), paddings_prepend + [[1, 0]], constant_values=1.)
    chol_result = np.pad(raw_correlation, paddings_prepend + [[1, 0], [0, 1]], constant_values=0.)
    ix = np.arange(dim)
    chol_result[...,  ix, ix] = diag
    return chol_result

def r_lkj(eta=1.0, n=1, dim=1, rng=None, seed=None):
    L = r_lkj_cholesky(eta, n, dim, rng, seed)
    axes = np.arange(len(L.shape), dtype=np.int32)
    axes[2:] = 3, 2
    R = np.matmul(L, np.transpose(L, axes))
    ix = np.arange(dim)
    R[...,  ix, ix] = 1.0
    return R



def _special_ortho(dim, rng=None, seed=None):
    rng = np.random.default_rng(seed) if rng is None else rng
    H = np.eye(dim)
    D = np.ones((dim,))
    for n in range(1, dim):
        x = rng.normal(size=(dim-n+1,))
        D[n-1] = np.sign(x[0])
        x[0] -= D[n-1] * np.sqrt((x * x).sum())
        Hx = (np.eye(dim - n + 1) - 2.*np.outer(x, x) / (x * x).sum())
        mat = np.eye(dim)
        mat[n-1:, n-1:] = Hx
        H = np.dot(H, mat)
    D[-1] = (-1)**(1 - (dim % 2)) * D.prod()
    H = (D * H.T).T
    return H

def _ortho_norm(n, p=None, rng=None, seed=None):
    p = n if p is None else p
    rng = np.random.default_rng(seed) if rng is None else rng
    H = rng.normal(0, 1, size=(n, p))
    Q, _ = np.linalg.qr(H, mode="reduced")
    return Q



