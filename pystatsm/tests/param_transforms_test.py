# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 23:34:08 2023

@author: lukepinkel
"""
import numpy as np
import scipy as sp
from pystatsm.utilities.numerical_derivs import jac_approx
from pystatsm.utilities.param_transforms import OrderedTransform

x = sp.special.ndtri(np.linspace(0, 1, 6, endpoint=False)[1:])
y = OrderedTransform._fwd(x, 5)
x = OrderedTransform._rvs(y, 5)


dx_dy = OrderedTransform._jac_rvs(y, 5)
dy_dx = OrderedTransform._jac_fwd(x, 5)

d2x_dy2 = OrderedTransform._hess_rvs(y, 5)
d2y_dx2 = OrderedTransform._hess_fwd(x, 5)


dx_dy_nm = jac_approx(lambda y: OrderedTransform._rvs(y.copy(), 5), y)
dy_dx_nm = jac_approx(lambda x: OrderedTransform._fwd(x.copy(), 5), x)

assert(np.allclose(dx_dy_nm, dx_dy))
assert(np.allclose(dy_dx_nm, dy_dx))


d2x_dy2_nm = jac_approx(lambda y: OrderedTransform._jac_rvs(y.copy(), 5), y)
d2y_dx2_nm = jac_approx(lambda x: OrderedTransform._jac_fwd(x.copy(), 5), x)


assert(np.allclose(d2x_dy2_nm, d2x_dy2))
assert(np.allclose(d2y_dx2_nm , d2y_dx2))


