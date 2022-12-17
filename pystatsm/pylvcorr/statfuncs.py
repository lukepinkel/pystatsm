#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 21:03:51 2020

@author: lukepinkel
"""
import numba
import numpy as np # analysis:ignore
import scipy as sp # analysis:ignore
import pandas as pd # analysis:ignore
TWOPI = 2.0 * np.pi
ROOT2PI = np.sqrt(np.pi*2.0)
ROOT2 = np.sqrt(2.0)

@numba.jit(nopython=True)
def norm_pdf_jit(x, mu=0.0, sigma=1.0):
    z = (x - mu) / sigma
    p = np.exp(-z**2 / 2.0) / (sigma * ROOT2PI)
    return p

def norm_pdf(x, mu=0.0, sigma=1.0):
    z = (x - mu) / sigma
    p = np.exp(-z**2 / 2.0) / (sigma * ROOT2PI)
    return p

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

def norm_cdf(x, mu=0.0, sigma=1.0):
    z = (x - mu) / (sigma * ROOT2)
    p = (sp.special.erf(z) + 1.0) * 0.5
    return p    

def norm_qtf(x):
    prob = np.sqrt(2) * sp.special.erfinv(2.0 * x - 1.0)
    return prob


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


def binorm_pdf_old(x, y, r, mu_x=0, mu_y=0, sx=1, sy=1):
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

def binorm_dl(h, k, r):
    r2 = 1 - r**2
    constant = 1.0 / (TWOPI * np.sqrt(r2))
    dl = np.exp(-(h**2-2*r*h*k+k**2) / (2 * r2))
    dldp = dl * constant
    return dldp

def binorm_l(h, k, r):
    root1 = np.sqrt(5.0 - 2 * np.sqrt(10.0/7.0)) / 3
    root2 = np.sqrt(5.0 + 2 * np.sqrt(10.0/7.0)) / 3
    r2 = r/2.0
    w1 = 128.0/225.0
    w2 = (322.0+13.0*np.sqrt(70.0)) / 900.0
    w3 = (322.0-13.0*np.sqrt(70.0)) / 900.0
    
    eq1 = w1 * binorm_dl(h, k, r / 2)
    
    eq2 = w2 * binorm_dl(h, k, (1-root1) * r2)
    eq3 = w2 * binorm_dl(h, k, (1+root1) * r2)
    
    eq4 = w3 * binorm_dl(h, k, (1-root2) * r2)
    eq5 = w3 * binorm_dl(h, k, (1+root2) * r2)

    likelihood = r2 * (eq1 + eq2 + eq3 + eq4 + eq5) 
    likelihood += norm_cdf(-h) * norm_cdf(-k)
    return likelihood

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

def binorm_l2(h, k, r):
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
    
    gsum = weights[0, 1] * binorm_dl(h, k, (1.0 + weights[0, 2]) * r2)
    for i in range(1, 32):
        w = weights[i, 1]
        root = weights[i, 2]
        eq1 = w * binorm_dl(h, k, (1-root) * r2)
        eq2 = w * binorm_dl(h, k, (1+root) * r2)
        eq = eq1 + eq2
        gsum += eq
        
    likelihood = r2 * (gsum)
    likelihood += norm_cdf(-h) * norm_cdf(-k)
    return likelihood

@numba.jit(nopython=True)
def binorm_cdf_jit(h, k, r):
    likelihood = binorm_l2_jit(h, k, r)
    phi = likelihood + norm_cdf_jit(h) + norm_cdf_jit(k) - 1
    return phi

def binorm_cdf_old(h, k, r):
    likelihood = binorm_l2(h, k, r)
    phi = likelihood + norm_cdf(h) + norm_cdf(k) - 1
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



def binorm_cdf2(lower, upper, r):
    pr_uu = binorm_cdf(upper[0], upper[1], r)
    pr_ul = binorm_cdf(upper[0], lower[1], r)
    pr_lu = binorm_cdf(lower[0], upper[1], r)
    pr_ll = binorm_cdf(lower[0], lower[1], r)
    pr = pr_uu - pr_ul - pr_lu + pr_ll
    return pr

def binorm_pdf2(lower, upper, r):
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


def dbinorm_pdf2(lower, upper, r):
    dp_uu = dbinorm_pdf(upper[0], upper[1], r)
    dp_ul = dbinorm_pdf(upper[0], lower[1], r)
    dp_lu = dbinorm_pdf(lower[0], upper[1], r)
    dp_ll = dbinorm_pdf(lower[0], lower[1], r)
    dp = dp_uu - dp_ul - dp_lu + dp_ll
    return dp


def dbinorm_du(u, v, r):
    den = np.sqrt(1 - r**2)
    num = v - r * u
    dPhi = norm_cdf(num / den) * norm_pdf(u)
    return dPhi

def _dbinorm_du(u, v, r):
    den = np.sqrt(1 - r**2)
    num = v - r * u
    dPhi = norm_cdf(num / den)
    return dPhi




def polyex(x, tau, rho):
    return (tau - rho*x) / np.sqrt(1-rho**2)

def polyserial_ll(rho, x, y, tau, order):
    ll = []
    for xi, yi in list(zip(x, y)):
        k = order[yi]
        tau1, tau2 = polyex(xi, tau[k+1], rho), polyex(xi, tau[k], rho)
        ll.append(np.log(norm_cdf(tau1)-norm_cdf(tau2)))
    ll = -np.sum(np.array(ll), axis=0)
    return ll

def polychor_thresh(X):
    '''
    Maximum likelihood estimates for thresholds
    
    Parameters:
        X: crosstabulation table
    Returns:
        a: thresholds for axis 0
        b: thresholds for axis 1
    '''
    N = float(np.sum(X))
    a = norm_qtf(np.sum(X, axis=0).cumsum() / N)[:-1]
    b = norm_qtf(np.sum(X, axis=1).cumsum() / N)[:-1]
    a, b = np.concatenate([[-1e6], a, [1e6]]), np.concatenate([[-1e6], b, [1e6]])
    return a, b

def polychor_probs(a, b, r):
    '''
    Cumulative bivariate normal distribution.  Computes the probability
    that a value falls in category i,j
    
    Parameters:
        a: Thresholds along axis 0
        b: Thresholds along axis 1
        r: correlation coefficient
    
    Returns:
        pr: Matrix of probabilities
    '''
    pr = np.array([[binorm_cdf(x, y, r) for x in a] for y in b])
    return pr

def polychor_loglike(X, a, b, r):
    '''
    Log likelihood of a contingency table given thresholds and  the correlation
    coefficient
    
    Parameters:
        X: Contigency table
        a: Thresholds along axis 0
        b: Thresholds along axis 1
        r: correlation coefficient
    Returns:
        ll: Log likelihood
    '''
    pr = polychor_probs(a, b, r)
    if len(pr.shape)>=3:
        pr = pr[:, :, 0]
    n, k = pr.shape
    pr = np.array([[pr[i, j]+pr[i-1,j-1]-pr[i-1,j]-pr[i,j-1] 
                   for j in range(1,k)] for i in range(1,n)])
    pr = np.maximum(pr, 1e-16)
    ll = np.sum(X * np.log(pr))
    return ll

def normal_categorical(x, nx):
    '''
    Splits continuous variable into nx categories
    
    Parameters:
        x: continuous vaiable in an array
        nx: number of categories
    
    Returns:
        xcat: categorical x
    '''
    xcat = pd.qcut(x, nx, labels=[i for i in range(nx)]).astype(float)
    return xcat


def polychor_ll(params, X, k):
    rho = params[0]
    a, b = params[1:k+1], params[k+1:]
    return -polychor_loglike(X, a, b, rho)



def polychor_partial_ll(rho, X, k, params):
    a, b = params[:k], params[k:]
    return -polychor_loglike(X, a, b, rho)

  
