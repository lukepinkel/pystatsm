#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 22:00:15 2020

@author: lukepinkel
"""



import numpy as np
import scipy as sp
import scipy.stats
import pandas as pd
from pystatsm.pysem.sem import SEM
from pystatsm.utilities.random import exact_rmvnorm
from pystatsm.utilities.linalg_operations import invech
from pystatsm.utilities.linalg_operations import invech
from pystatsm.utilities.numerical_derivs import fo_fc_cd, so_gc_cd

rng = np.random.default_rng(123)

def test_sem_bollen():
    data_bollen = pd.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/sem/Bollen.csv", index_col=0)
    data_bollen = data_bollen[['x1', 'x2', 'x3', 'y1', 'y2', 'y3', 'y4', 'y5', 'y6', 'y7', 'y8', ]]
    
    L = np.array([[1., 0., 0.],
                  [1., 0., 0.],
                  [1., 0., 0.],
                  [0., 1., 0.],
                  [0., 1., 0.],
                  [0., 1., 0.],
                  [0., 1., 0.],
                  [0., 0., 1.],
                  [0., 0., 1.],
                  [0., 0., 1.],
                  [0., 0., 1.]])
    
    B = np.array([[0., 0., 0.],
                  [1., 0., 0.],
                  [1., 1., 0.]])
    
    Lambda1 = pd.DataFrame(L, index=data_bollen.columns, columns=['ind60', 'dem60', 'dem65'])
    Beta1 = pd.DataFrame(B, index=Lambda1.columns, columns=Lambda1.columns)
    Psi1 = pd.DataFrame(np.eye(Lambda1.shape[0]), index=Lambda1.index, columns=Lambda1.index)
    
    off_diag = [['y1', 'y5'], ['y2', 'y4'], ['y3', 'y7'], ['y4', 'y8'],
                ['y6', 'y8'], ['y2', 'y6']]
    for x, y in off_diag:
        Psi1.loc[x, y] = Psi1.loc[y, x] = 0.05
    Phi1 = pd.DataFrame(np.eye(Lambda1.shape[1]), index=Lambda1.columns,
                       columns=Lambda1.columns)
    
    model = SEM(Lambda1, Beta1, Phi1, Psi1, data=data_bollen)
    model.fit()
    
    theta = np.array([2.18036773, 1.81851130, 1.25674650, 1.05771677, 1.26478654, 
                      1.18569630, 1.27951218, 1.26594698, 1.48300054, 0.57233619,
                      0.83734481, 0.44843715, 3.95603311, 0.17248133, 0.08154935,
                      0.11980648, 0.46670258, 1.89139552, 0.62367107, 7.37286854, 
                      1.31311258, 2.15286127, 5.06746210, 0.79496028, 3.14790480,
                      0.34822604, 2.35097047, 4.95396775, 1.35616712, 3.43137392, 
                      3.25408501])
    assert(np.allclose(model.theta, theta))
    
    assert(model.opt.success)
    assert(np.abs(model.opt.grad).max()<1e-5)    
    assert(np.allclose(model.gradient(theta+0.1), fo_fc_cd(model.loglike, theta+0.1)))
    assert(np.allclose(model.hessian(theta), so_gc_cd(model.gradient, theta)))



################################################################

def test_sem_med():
    
    x = rng.normal(0, 1, size=1000)
    x = (x - x.mean()) / x.std()
    
    u = rng.normal(0, 1, size=1000)
    u = (u - u.mean()) / u.std()
    v = rng.normal(0, 1, size=1000)
    v = (v - v.mean()) / v.std()
    
    m = 0.5*x + u
    y = 0.7*m + v
    
    data_path = pd.DataFrame(np.vstack((x, m, y)).T, columns=['x', 'm', 'y'])
    data_path = data_path - data_path.mean(axis=0)
    Lambda = pd.DataFrame(np.eye(3), index=data_path.columns, columns=data_path.columns)
    Beta = np.array([[0.0, 0.0, 0.0],
                     [1.0, 0.0, 0.0],
                     [1.0, 1.0, 0.0]])
    Beta = pd.DataFrame(Beta, index=data_path.columns, columns=data_path.columns)
    
    Phi = np.array([[1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0]])
    Phi = pd.DataFrame(Phi, index=data_path.columns, columns=data_path.columns)
    
    Psi = np.array([[0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0]])
    Psi = pd.DataFrame(Psi, index=data_path.columns, columns=data_path.columns)
    
    
    model = SEM(Lambda, Beta, Phi, Psi, data=data_path)
    model.fit(opt_kws=dict(options=dict(gtol=1e-20, xtol=1e-100)))
    theta = np.array([0.52667623, 0.13011021, 0.40325746, 1.00000000, 0.99928838,
                      1.43672041])
    assert(np.allclose(model.theta, theta))
    assert(model.opt.success)
    assert(np.abs(model.opt.grad).max()<1e-5)    
    assert(np.allclose(model.gradient(theta+0.1), fo_fc_cd(model.loglike, theta+0.1)))
    assert(np.allclose(model.hessian(theta), so_gc_cd(model.gradient, theta)))



################################################################


def test_nlsy_model():
    vechS = [2.926, 1.390, 1.698, 1.628, 1.240, 0.592, 0.929,
             0.659, 4.257, 2.781, 2.437, 0.789, 1.890, 1.278, 0.949,
             4.536, 2.979, 0.903, 1.419, 1.900, 1.731, 5.605, 1.278, 1.004,
             1.000, 2.420, 3.208, 1.706, 1.567, 0.988, 3.994, 1.654, 1.170,
             3.583, 1.146, 3.649]
    
    S = pd.DataFrame(invech(np.array(vechS)), columns=['anti1', 'anti2',
                     'anti3', 'anti4', 'dep1', 'dep2', 'dep3', 'dep4'])
    S.index = S.columns
    
    X = pd.DataFrame(exact_rmvnorm(S.values, 180, seed=123), columns=S.columns)
    X += np.array([1.750, 1.928, 1.978, 2.322, 2.178, 2.489, 2.294, 2.222])
    
    data = pd.DataFrame(X, columns=S.columns)
    
    Lambda = pd.DataFrame(np.eye(8), index=data.columns, columns=data.columns)
    Lambda = Lambda.iloc[[1, 2, 3, 5, 6, 7, 0, 4], [1, 2, 3, 5, 6, 7, 0, 4]]
    
    Beta = pd.DataFrame([[0, 0, 0, 0, 0, 0, 1, 1],
                         [1, 0, 0, 1, 0, 0, 0, 0],
                         [0, 1, 0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 1, 1],
                         [1, 0, 0, 1, 0, 0, 0, 0],
                         [0, 1, 0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0],
                         ], index=Lambda.columns, columns=Lambda.columns)
    Phi   = pd.DataFrame([[1.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
                          [0.0, 0.0, 1.0, 0.0, 0.0, 0.1, 0.0, 0.0],
                          [0.1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                          [0.0, 0.1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                          [0.0, 0.0, 0.1, 0.0, 0.0, 1.0, 0.0, 0.0],
                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.1],
                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 1.0],
                          ], index=Lambda.columns, columns=Lambda.columns)
    
    Psi = Lambda.copy()*0.0
    data = data.loc[:, Lambda.index]
    model = SEM(Lambda, Beta, Phi, Psi, data=data)
    model.fit()
    theta = np.array([ 0.62734 ,  0.147299,  0.69399 ,  0.318353,  0.058419,  0.344418,
                      -0.088914,  0.151027,  0.443465, -0.027558,  0.074533,  0.542448,
                      3.581777,  1.500315,  2.70847 ,  1.001634,  3.626519,  1.3206  ,
                      3.084899,  2.825085,  2.924854,  2.926   ,  1.24    ,  3.208   ])
    
    assert(np.allclose(model.theta, theta))
    assert(model.opt.success)
    assert(np.abs(model.opt.grad).max()<1e-5)    
    assert(np.allclose(model.gradient(theta+0.1), fo_fc_cd(model.loglike, theta+0.1)))
    assert(np.allclose(model.hessian(theta), so_gc_cd(model.gradient, theta)))


def test_holzingercfa():
        
    data = pd.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/sem/HS.data.csv", index_col=0)
    df = data.loc[:, ["visual", "cubes", "flags", "paragrap", "sentence",
                       "wordm", "addition", "counting", "straight"]]
    df = df / np.array([6.,4.0, 8., 3., 4., 7., 23., 20., 36.])
    Lambda = np.zeros((9, 3))
    Lambda[0:3, 0] = 1.0
    Lambda[3:6, 1] = 1.0
    Lambda[6:9, 2] = 1.0
    
    Lambda = pd.DataFrame(Lambda, index=df.columns, columns=['visual', 'textual', 'speed'])
    Beta = pd.DataFrame(np.zeros((3, 3)), index=Lambda.columns, columns=Lambda.columns)
    Phi=Beta+np.eye(3)+0.05
    Psi = pd.DataFrame(np.eye(9), index=df.columns, columns=df.columns)
    model = SEM(Lambda, Beta, Phi, Psi, data=df)
    model.fit()
    theta = np.array([0.55372 , 0.729526, 1.113068, 0.926117, 1.180358, 1.083565,
                      0.809095, 0.408174, 0.261583, 0.979517, 0.173783, 0.383061,
                      0.549275, 1.133711, 0.844258, 0.371148, 0.446243, 0.356234,
                      0.796578, 0.488285, 0.567506])
    
    assert(np.allclose(model.theta, theta))
    assert(model.opt.success)
    assert(np.abs(model.opt.grad).max()<1e-5)    
    assert(np.allclose(model.gradient(theta+0.1), fo_fc_cd(model.loglike, theta+0.1)))
    assert(np.allclose(model.hessian(theta), so_gc_cd(model.gradient, theta)))
    

