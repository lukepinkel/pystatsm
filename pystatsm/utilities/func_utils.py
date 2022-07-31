#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 13:34:49 2020

@author: lukepinkel
"""
import numba
import numpy as np
import scipy as sp
import scipy.special
SQRT2 = np.sqrt(2)
ROOT2PI = np.sqrt(2.0 * np.pi)

def poisson_logp(x, mu, logp=True):
     p = sp.special.xlogy(x, mu) - sp.special.gammaln(x + 1) - mu
     if logp==False:
         p = np.exp(p)
     return p
 
    
def log1p(x):
    return np.log(1+x)


def norm_cdf(x, mean=0.0, sd=1.0):
    z = (x - mean) / sd
    p = (sp.special.erf(z/SQRT2) + 1.0) / 2.0
    return p

def norm_pdf(x, mean=0.0, sd=1.0):
    z = (x - mean) / sd
    p = np.exp(-z**2 / 2.0) / (ROOT2PI * sd)
    return p



def get_part(arr, sol, size, step, maximum, res):
    if step==size:
        res.append(sol.copy())
    else:
        sol[step] = 1
        while sol[step]<=maximum:
            get_part(arr, sol, size, step+1, maximum, res)
            sol[step] += 1
        get_part(arr, sol, size, step+1, maximum+1, res)

def partition_set(n):    
    size = n
    arr = np.arange(1, size+1)-1
    sol = np.zeros(size, dtype=int)
    res = []
    get_part(arr, sol, size, 0, 0, res)
    return res

@numba.jit(nopython=True)
def soft_threshold(x, t):
    y = np.maximum(np.abs(x) - t, 0) * np.sign(x)
    return y

@numba.jit(nopython=True)
def expit(x):
    u = np.exp(x)
    y = u / (1.0 + u)
    return y


def sum_preserving_round(arr):
    arr_floor = np.floor(arr)
    arr_fract = arr - arr_floor
    arr_fract_sort = np.argsort(arr_fract)
    sum_diff = int(np.round(np.sum(arr) -  np.sum(arr_floor)))
    ind = arr_fract_sort[-sum_diff:]
    arr_floor[ind] = arr_floor[ind] + 1
    return arr_floor


def sum_preserving_min(arr, min_):
    arr_ind = arr < min_
    arr_diff= arr - min_
    n_lt = np.sum(arr_ind)
    if n_lt > 0:
        arr_sort = np.argsort(arr)[-n_lt:]
        arr[arr_ind] = arr[arr_ind] - arr_diff[arr_ind]
        arr[arr_sort] = arr[arr_sort] + arr_diff[arr_ind]
    return arr
    
def sizes_to_inds(sizes):
    return np.r_[0, np.cumsum(sizes)]
    
def sizes_to_slice_vals(sizes):
    inds = sizes_to_inds(sizes)
    return list(zip(inds[:-1], inds[1:]))


def allocate_from_proportions(n, proportions):
    if np.abs(1.0 - np.sum(proportions)) > 1e-12:
        raise ValueError("Proportions Don't Sum to One")
    k = proportions * n
    k = sum_preserving_min(k, 1)
    k = sum_preserving_round(k).astype(int)
    slice_vals = sizes_to_slice_vals(k)
    return k, slice_vals


def handle_default_kws(kws, default_kws):
    kws = {} if kws is None else kws
    kws = {**default_kws, **kws}
    return kws
    


