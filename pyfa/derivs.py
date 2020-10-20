#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 10:09:26 2020

@author: lukepinkel
"""

from jax.config import config
config.update("jax_enable_x64", True)

import jax
import numpy as np
import scipy as sp
import pandas as pd
from pystats.utilities.special_mats import kronvec_mat, nmat
from pystats.utilities.linalg_operations import vec
from pystats.pyfa.factor_analysis2 import FactorAnalysis, multi_rand




Lambda = np.zeros((12, 3))
Lambda[0:4, 0] = 1
Lambda[4:8, 1] = 1
Lambda[8: , 2] = 1
Phi = np.eye(3)
Psi = np.diag(np.random.uniform(low=0.2, high=1.0, size=12))

S = Lambda.dot(Phi).dot(Lambda.T) + Psi
S = pd.DataFrame(S, index=['x%i'%i for i in range(1, 13)])
S.columns = S.index
X = multi_rand(S)
X = pd.DataFrame(X, columns=['x%i'%i for i in range(1, 13)])


model = FactorAnalysis(X, 3, loadings_free=Lambda/2)
model.fit()

LA, PH, PS = model.Lambda, model.Phi, model.Psi

def to_sigma(LA, PH, PS):
    LA = LA.reshape(12, 3, order='F')
    return (LA.dot(PH).dot(LA.T)+PS).reshape(-1, order='F')

d2sigma = jax.hessian(to_sigma, argnums=0)


Kv = kronvec_mat((12, 3), (12, 12))

N = sp.sparse.kron(sp.sparse.eye(36), nmat(12))
D = sp.sparse.kron(sp.sparse.csc_matrix(Phi), sp.sparse.eye(12))
D = sp.sparse.kron(D, sp.sparse.csc_matrix(vec(np.eye(12))).T)

J = N.dot(Kv).dot(D)



J2 = d2sigma(LA.reshape(-1, order='F'), PH, PS)

J2 = J2.reshape(144*36, 36, order='F')

J1 = J.A
J2 = np.asarray(J2)


model = FactorAnalysis(X, 3, loadings_free=Lambda/2)

history = []
def callback(x, args):
    history.append(np.concatenate([np.asarray(x),
                                   np.atleast_1d(np.asarray(model.loglike(x)))]))

res = sp.optimize.minimize(model.loglike, model.params, jac=model.gradient,
                           hess=model.hessian, method='trust-constr',
                           options=dict(verbose=3),
                           bounds=model.bounds, callback=callback)
print(model.LA)
model.fit()


history = pd.DataFrame(np.vstack(history))









