#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 14:55:12 2022

@author: lukepinkel
"""

import numpy as np


def get_loadings_permutation(L_target, L):
    B, _, _, _ = np.linalg.lstsq(L_target, L, None)
    j = np.argmax(np.abs(B), axis=1)
    s = np.sign(np.diag(B[:, j]))
    return j, s

def align_model_matrices(L_target, L, Phi, L_se, Phi_se, permutation=None):
    if permutation is None:
        j, s = get_loadings_permutation(L_target, L)
    else:
        j, s = permutation
    L_aligned = L[:, j] * s
    L_se_aligned = L_se[:, j]# * s
    Phi_aligned = s * Phi[:, j][j] * s[:,None]
    Phi_se_aligned = Phi_se[:, j][j]
    return L_aligned, L_se_aligned, Phi_aligned, Phi_se_aligned, j, s
        

    