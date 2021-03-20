#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 16 21:43:30 2020

@author: lukepinkel
"""
import jax
import jax.numpy as jnp


@jax.jit
def jax_vec(X):
    '''
    Takes an n \times p matrix and returns a 1 dimensional np vector
    '''
    return X.reshape(-1, order='F')

@jax.jit
def jax_invec(x, n_rows, n_cols):
    '''
    Takes an np 1 dimensional vector and returns an n \times p matrix
    '''
    return x.reshape(int(n_rows), int(n_cols), order='F')

@jax.jit
def jax_vech(X):
    '''
    Half vectorization operator; returns an \frac{(n+1)\times n}{2} vector of
    the stacked columns of unique items in a symmetric  n\times n matrix
    '''
    rix, cix = jnp.triu_indices(len(X))
    res = jnp.take(X.T, rix*len(X)+cix)
    return res

@jax.jit
def jax_invech(v):
    '''
    Inverse half vectorization operator
    '''
    rows = int(jnp.round(.5 * (-1 + jnp.sqrt(1 + 8 * len(v)))))
    res = jnp.zeros((rows, rows))
    res = jax.ops.index_update(res, jnp.triu_indices(rows), v)
    res = res + res.T - jnp.diag(jnp.diag(res))
    return res