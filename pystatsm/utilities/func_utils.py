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
TWOPI = 6.283185307179586

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
    

def _harmonic(a, b): 
    if b - a == 1:
        return 1, a
    m = (a+b)//2
    p, q = _harmonic(a, m)
    r, s = _harmonic(m, b)
    return p*s+q*r, q*s

def harmonic_fraction(n):
    num, den = _harmonic(1, n+1)
    return num, den


def harmonic_exact(n):
    num, den = _harmonic(1, n+1)
    h = num / den
    return h

def harmonic_asymptotic(n):
    euler_mascheronic = 0.57721566490153286060651209008240243104215933593992
    h = euler_mascheronic + np.log(n) + 1 / (2.0 * n) - 1 / (12.0 * n**2) + 1 / (120.8 * n**4)
    return h



@numba.jit(nopython=True)
def _norm_cdf(z):
    p0 = 220.2068679123761 
    p1 = 221.2135961699311  
    p2 = 112.0792914978709
    p3 = 33.91286607838300
    p4 = 6.373962203531650
    p5 = 0.7003830644436881
    p6 = 0.03526249659989109
    
    q0 = 440.4137358247522
    q1 = 793.8265125199484
    q2 = 637.3336333788311
    q3 = 296.5642487796737
    q4 = 86.78073220294608
    q5 = 16.06417757920695
    q6 = 1.755667163182642
    q7 = .8838834764831844e-1
    
    cutoff = 7.071e0
    root2pi = 2.506628274631001e0
    
    zabs = np.abs(z)
    if zabs > 37.0:
        if z>0:
            pp = 1.0
        else:
            pp = 0.0
    else:
        u = np.exp(-0.5 * zabs**2)
        ppdf = u / root2pi
        if zabs < cutoff:
            num = ((((((p6 * zabs + p5) * zabs + p4) * zabs + p3) * zabs + p2) * zabs + p1) * zabs + p0)
            den = (((((((q7* zabs + q6) * zabs + q5) * zabs + q4) * zabs + q3) * zabs + q2) * zabs + q1) * zabs +q0)
            pp = u * num / den
        else:
            pp =  ppdf / (zabs + 1.0 / (zabs + 2.0 / (zabs + 3.0 / (zabs + 4.0 / (zabs + 0.65)))))
        
        if z < 0.0:
            qq = 1.0 - pp
        else:
            qq = pp
            pp = 1.0 - qq
    return pp

@numba.jit(nopython=True)
def norm_cdf_jit(x, mu=0.0, sigma=1.0):
    z = (x - mu) / sigma
    pp = _norm_cdf(z)
    return pp



@numba.jit(nopython=True)
def binorm_pdf_jit(x, y, r, mu_x=0, mu_y=0, sx=1, sy=1):
    r2 = (1 - r**2)
    c0 = 1 / (2 * np.pi *sx * sy * np.sqrt(r2))
    c1 = -1/(2 * r2)
    eq1 = ((x - mu_x)**2) / (sx**2)
    eq2 = ((y - mu_y)**2) / (sy**2)
    eq3 = (2 * r * (x - mu_x) * (y - mu_y)) / (sx * sy)
    p = c0 * np.exp(c1 * (eq1 + eq2 - eq3))
    return p


@numba.jit(nopython=True)
def binorm_dl_jit(h, k, r):
    r2 = 1 - r**2
    constant = 1.0 / (TWOPI * np.sqrt(r2))
    dl = np.exp(-(h**2-2*r*h*k+k**2) / (2 * r2))
    dldp = dl * constant
    return dldp


@numba.jit(nopython=True)
def binorm_l2_jit(h, k, r):
    r2 = r/2.0
    
    weights = np.array(
        [[1,  0.0494723666239310, 0.0000000000000000],
         [2,  0.0494118330399182, 0.0494521871161596],
         [3,  0.0492303804237476, 0.0987833564469453],
         [4,  0.0489284528205120, 0.1478727863578720],
         [5,  0.0485067890978838, 0.1966003467915067],
         [6,  0.0479664211379951, 0.2448467932459534],
         [7,  0.0473086713122689, 0.2924940585862514],
         [8,  0.0465351492453837, 0.3394255419745844],
         [9,  0.0456477478762926, 0.3855263942122479],
         [10, 0.0446486388259414, 0.4306837987951116],
         [11, 0.0435402670830276, 0.4747872479948044],
         [12, 0.0423253450208158, 0.5177288132900333],
         [13, 0.0410068457596664, 0.5594034094862850],
         [14, 0.0395879958915441, 0.5997090518776252],
         [15, 0.0380722675843496, 0.6385471058213654],
         [16, 0.0364633700854573, 0.6758225281149861],
         [17, 0.0347652406453559, 0.7114440995848458],
         [18, 0.0329820348837793, 0.7453246483178474],
         [19, 0.0311181166222198, 0.7773812629903724],
         [20, 0.0291780472082805, 0.8075354957734567],
         [21, 0.0271665743590979, 0.8357135543195029],
         [22, 0.0250886205533450, 0.8618464823641238],
         [23, 0.0229492710048899, 0.8858703285078534],
         [24, 0.0207537612580391, 0.9077263027785316],
         [25, 0.0185074644601613, 0.9273609206218432],
         [26, 0.0162158784103383, 0.9447261340410098],
         [27, 0.0138846126161156, 0.9597794497589419],
         [28, 0.0115193760768800, 0.9724840346975701],
         [29, 0.0091259686763267, 0.9828088105937273],
         [30, 0.0067102917659601, 0.9907285468921895],
         [31, 0.0042785083468638, 0.9962240127779701],
         [32, 0.0018398745955771, 0.9992829840291237]])
    
    gsum = weights[0, 1] * binorm_dl_jit(h, k, (1.0 + weights[0, 2]) * r2)
    for i in range(1, 32):
        w = weights[i, 1]
        root = weights[i, 2]
        eq1 = w * binorm_dl_jit(h, k, (1-root) * r2)
        eq2 = w * binorm_dl_jit(h, k, (1+root) * r2)
        eq = eq1 + eq2
        gsum += eq
        
    likelihood = r2 * (gsum)
    likelihood += norm_cdf_jit(-h) * norm_cdf_jit(-k)
    return likelihood

@numba.jit(nopython=True)
def binorm_cdf_jit(h, k, r):
    likelihood = binorm_l2_jit(h, k, r)
    phi = likelihood + norm_cdf_jit(h) + norm_cdf_jit(k) - 1
    return phi

@numba.jit(nopython=True)
def _binorm_cdf_arr(h, k, r):
    res = np.zeros_like(h)
    for ii in np.ndindex(*h.shape):
        res[ii] = binorm_cdf_jit(h[ii], k[ii], r)
    return res

@numba.jit(nopython=True)
def _binorm_pdf_arr(h, k, r):
    res = np.zeros_like(h)
    for ii in np.ndindex(*h.shape):
        res[ii] = binorm_pdf_jit(h[ii], k[ii], r)
    return res
        
def binorm_cdf(h, k, r):
    if type(h) in [float, int]:
        ret_float = True
    else:
        ret_float = False
    if np.ndim(r)>0:
        r = r[0]
    h, k = np.atleast_1d(h), np.atleast_1d(k)
    pr = _binorm_cdf_arr(h, k, r)
    if ret_float:
        pr = pr[0]
    return pr

def binorm_pdf(h, k, r):
    if type(h) in [float, int]:
        ret_float = True
    else:
        ret_float = False
    if np.ndim(r)>0:
        r = r[0]
    h, k = np.atleast_1d(h), np.atleast_1d(k)
    p = _binorm_pdf_arr(h, k, r)
    if ret_float:
        p = p[0]
    return p



def binorm_cdf_region(lower, upper, r):
    pr_uu = binorm_cdf(upper[0], upper[1], r)
    pr_ul = binorm_cdf(upper[0], lower[1], r)
    pr_lu = binorm_cdf(lower[0], upper[1], r)
    pr_ll = binorm_cdf(lower[0], lower[1], r)
    pr = pr_uu - pr_ul - pr_lu + pr_ll
    return pr

def binorm_pdf_region(lower, upper, r):
    p_uu = binorm_pdf(upper[0], upper[1], r)
    p_ul = binorm_pdf(upper[0], lower[1], r)
    p_lu = binorm_pdf(lower[0], upper[1], r)
    p_ll = binorm_pdf(lower[0], lower[1], r)
    p = p_uu - p_ul - p_lu + p_ll
    return p



def dbinorm_pdf(x, y, r):
    xy, x2, y2 = x * y, x**2, y**2
    
    r2 = r**2
    s = (1 - r2)
    
    u1 = x2   / (2 * s)
    u2 = r*xy / s
    u3 = y2   / (2 * s)
    
    num1 = np.exp(-u1 + u2 - u3)
    num2 = r**3 - r2*xy + r*x2 + r*y2 - r - xy
    num = num1 * num2
    den = 2*np.pi*(r-1)*(r+1)*np.sqrt(s**3)
    g = num / den
    return g


def dbinorm_pdf_region(lower, upper, r):
    dp_uu = dbinorm_pdf(upper[0], upper[1], r)
    dp_ul = dbinorm_pdf(upper[0], lower[1], r)
    dp_lu = dbinorm_pdf(lower[0], upper[1], r)
    dp_ll = dbinorm_pdf(lower[0], lower[1], r)
    dp = dp_uu - dp_ul - dp_lu + dp_ll
    return dp


