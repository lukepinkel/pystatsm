# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 16:01:40 2021

@author: lukepinkel
"""

import numpy as np
import pystatsm.utilities.special_mats as special_mats
import pystatsm.utilities.linalg_operations as linalg


def test_special_mats():
    A = linalg.invech(np.arange(10))
    
    vh = linalg.vech(A)
    v = linalg.vec(A)
    
    Dp = special_mats.dmat(4).A
    Lp = special_mats.lmat(4).A
    Dpp = np.linalg.inv(np.dot(Dp.T, Dp)).dot(Dp.T)
    
    
    assert(np.allclose(Dp.dot(vh), v))
    assert(np.allclose(Lp.dot(v), vh))
    assert(np.allclose(Dpp.dot(v), vh))
    

    A = linalg.invec(np.arange(20), 5, 4)
    vecA = linalg.vec(A)
    vecAt = linalg.vec(A.T)
    Kpq = special_mats.kmat(5, 4).A
    
    assert(np.allclose(Kpq.dot(vecA), vecAt))
    
    
