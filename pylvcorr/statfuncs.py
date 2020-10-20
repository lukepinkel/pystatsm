#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 21:03:51 2020

@author: lukepinkel
"""
import tqdm # analysis:ignore
import numpy as np # analysis:ignore
import scipy as sp # analysis:ignore
import pandas as pd # analysis:ignore
import seaborn as sns # analysis:ignore
import matplotlib.pyplot as plt # analysis:ignore
from pystats.utilities.random_corr import multi_rand  # analysis:ignore

ROOT2PI = np.sqrt(np.pi*2.0)
ROOT2 = np.sqrt(2.0)

def norm_pdf(x, mu=0.0, sigma=1.0):
    z = (x - mu) / sigma
    p = np.exp(-z**2 / 2.0) / (sigma * ROOT2PI)
    return p

def norm_cdf(x, mu=0.0, sigma=1.0):
    z = (x - mu) / (sigma * ROOT2)
    p = (sp.special.erf(z) + 1.0) * 0.5
    return p
    
    
def norm_qtf(x):
    prob = np.sqrt(2) * sp.special.erfinv(2.0 * x - 1.0)
    return prob

def binorm_pdf(x, y, r, mu_x=0, mu_y=0, sx=1, sy=1):
    r2 = (1 - r**2)
    c0 = 1 / (2 * np.pi *sx * sy * np.sqrt(r2))
    c1 = -1/(2 * r2)
    eq1 = ((x - mu_x)**2) / (sx**2)
    eq2 = ((y - mu_y)**2) / (sy**2)
    eq3 = (2 * r * (x - mu_x) * (y - mu_y)) / (sx * sy)
    p = c0 * np.exp(c1 * (eq1 + eq2 - eq3))
    return p

def binorm_dl(h, k, r):
    r2 = 1 - r**2
    constant = 1 / (2 * np.pi * np.sqrt(r2))
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


def binorm_cdf(h, k, r):
    likelihood = binorm_l2(h, k, r)
    phi = likelihood + norm_cdf(h) + norm_cdf(k) - 1
    return phi


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

  
