#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 21:04:17 2020

@author: lukepinkel
"""

import pandas as pd 
from pystats.pyscca.scca3 import SCCA, _corr, vech

X = pd.read_csv("/users/lukepinkel/Downloads/RGCCA_X.csv", index_col=0)
Y = pd.read_csv("/users/lukepinkel/Downloads/RGCCA_Y.csv", index_col=0)

model = SCCA(X, Y)
Wx, Wy, rhos, optinfo = model._fit_symdef_d(0.15, 0.15, n_comps=2)
Bx, By = model.orthogonalize_components(Wx, Wy)

Zx, Zy = X.dot(Wx), Y.dot(Wy)
Zv, Zu = X.dot(Bx), Y.dot(By)

_corr(Zx, Zy)
_corr(Zx, Zx)
_corr(Zy, Zy)

_corr(Zv, Zu)
_corr(Zv, Zv)
_corr(Zu, Zu)

