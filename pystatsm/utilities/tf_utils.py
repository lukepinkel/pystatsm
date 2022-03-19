# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 12:39:23 2022

@author: lukepinkel
"""

import numpy as np


def spherical_uniform(shape, dim, rng):
    size = np.concatenate([shape, [dim]], axis=0).astype(np.int32)
    if dim==1:
        x = rng.binomial(n=1, p=0.5, size=size) * 2.0 - 1.0
    elif dim==2:
        z = rng.uniform(low=0.0, high=2.0*np.pi, size=shape)
        x = np.stack([np.cos(z), np.sin(z)], axis=-1)
    else:
        z = rng.normal(size=size)
        x = z / np.linalg.norm(z, ord=2, axis=-1)[..., np.newaxis]
    return x

def clockwise_spiral_fill_triangular(x, upper=False):
    x = np.asarray(x)
    m = np.int32(x.shape[-1])
    n = np.sqrt(0.25 + 2. * m) - 0.5
    n = np.int32(n)
    x_tail = x[..., (m - (n * n - m)):]
    y = np.concatenate([x, x_tail[..., ::-1]] if upper else [x_tail, x[..., ::-1]],axis=-1)
    y = y.reshape(np.concatenate([
        np.int32(x.shape[:-1]),
        np.int32([n, n]),
    ], axis=0))
    return np.triu(y) if upper else np.tril(y)