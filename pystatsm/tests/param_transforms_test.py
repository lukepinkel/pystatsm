import numpy as np
import scipy as sp
from pystatsm.utilities.random import r_lkj
from pystatsm.utilities.linalg_operations import _vech, _invech
from pystatsm.utilities.numerical_derivs import jac_approx
from pystatsm.utilities.param_transforms import (OrderedTransform,
                                                          CholeskyCov,
                                                          CombinedTransform,
                                                          LogTransform,
                                                          TanhTransform)


def test_choleskycov():
    rng = np.random.default_rng(123)
    mat_size = 5
    R = r_lkj(eta=1.0, n=1, dim=mat_size, rng=rng)[0, 0]
    V = np.diag(rng.uniform(low=2.0, high=6.0, size=mat_size)**0.5)
    S = V.dot(R).dot(V)
    x = _vech(S)

    t = CholeskyCov(mat_size)

    u = t.fwd(x)

    S1 = _invech(t.rvs(u))
    
    
    du_dx_nm = jac_approx(t.fwd, x)
    du_dx_an = t.jac_fwd(x)
    
    
    dx_du_nm = jac_approx(t.rvs, u)
    dx_du_an = t.jac_rvs(u)
    
    d2u_dx2_nm = jac_approx(t.jac_fwd, x)
    d2u_dx2_an = t.hess_fwd(x)
    
    d2x_du2_nm = jac_approx(t.jac_rvs, u)
    d2x_du2_an = t.hess_rvs(u)
    
    assert(np.allclose(S, S1))
    
    assert(np.allclose(t.rvs(t.fwd(x)), x))
    assert(np.allclose(t.fwd(t.rvs(u)), u))
    
    assert(np.allclose(du_dx_nm, du_dx_an, atol=1e-4))
    assert(np.allclose(dx_du_nm, dx_du_an, atol=1e-4))
    
    assert(np.allclose(d2u_dx2_nm, d2u_dx2_an, atol=1e-3))
    assert(np.allclose(d2x_du2_nm, d2x_du2_an, atol=1e-3))
   


def test_orderedtransform():
    x = sp.special.ndtri(np.linspace(0, 1, 6, endpoint=False)[1:])
    t = OrderedTransform()
    y = t.fwd(x)
    x = t.rvs(y)
    
    assert(np.allclose(t.rvs(t.fwd(x)), x))
    assert(np.allclose(t.fwd(t.rvs(y)), y))

    dx_dy = t.jac_rvs(y)
    dy_dx = t.jac_fwd(x)
    
    d2x_dy2 = t.hess_rvs(y)
    d2y_dx2 = t.hess_fwd(x)
    
    
    dx_dy_nm = jac_approx(t.rvs, y)
    dy_dx_nm = jac_approx(t.fwd, x)
        
    d2x_dy2_nm = jac_approx(t.jac_rvs, y)
    d2y_dx2_nm = jac_approx(t.jac_fwd, x)
    
    assert(np.allclose(dx_dy_nm, dx_dy))
    assert(np.allclose(dy_dx_nm, dy_dx))

    
    assert(np.allclose(d2x_dy2_nm, d2x_dy2))
    assert(np.allclose(d2y_dx2_nm, d2y_dx2))
    
    
    
def test_combinedtransform():
    t = CombinedTransform(transforms=[LogTransform, TanhTransform, OrderedTransform],
                                  index_objects = [5, 7, 5])
    rng = np.random.default_rng(123)
    x = rng.uniform(low=0, high=1.0, size=17)
    x[-5:] = OrderedTransform().rvs(x[-5:])
    
    
    y = t.fwd(x)
    
    assert(np.allclose(t.rvs(t.fwd(x)), x))
    assert(np.allclose(t.fwd(t.rvs(y)), y))
        
    dy_dx_an = t.jac_fwd(x)
    dx_dy_an = t.jac_rvs(y)
    
    dy_dx_nm = jac_approx(t.fwd, x)
    dx_dy_nm = jac_approx(t.rvs, y)

    d2y_dx2_an = t.hess_fwd(x)
    d2x_dy2_an = t.hess_rvs(y)
    
    d2y_dx2_nm = jac_approx(t.jac_fwd, x)
    d2x_dy2_nm = jac_approx(t.jac_rvs, y)
    
    
    assert(np.allclose(dx_dy_nm, dx_dy_an, rtol=1e-3, atol=1e-5))
    assert(np.allclose(dy_dx_nm, dy_dx_an, rtol=1e-3, atol=1e-5))

    assert(np.allclose(d2x_dy2_nm, d2x_dy2_an, rtol=1e-3, atol=1e-5))
    assert(np.allclose(d2y_dx2_nm, d2y_dx2_an, rtol=1e-3, atol=1e-5))
    
    
    x = rng.uniform(low=0, high=1.0, size=(5,3, 17))
    x[:, :,-5:] = OrderedTransform().rvs(x[:,:, -5:])

    y = t.fwd(x)
    
    assert(np.allclose(t.rvs(t.fwd(x)), x))
    assert(np.allclose(t.fwd(t.rvs(y)), y))
    
    dy_dx_an = t.jac_fwd(x)[0, 0]
    dx_dy_an = t.jac_rvs(y)[0, 0]
    
    d2y_dx2_an = t.hess_fwd(x)[0, 0]
    d2x_dy2_an = t.hess_rvs(y)[0, 0]
    
    dy_dx_nm = jac_approx(t.fwd, x[0, 0])
    dx_dy_nm = jac_approx(t.rvs, y[0, 0])
    
    d2y_dx2_nm = jac_approx(t.jac_fwd, x[0, 0])
    d2x_dy2_nm = jac_approx(t.jac_rvs, y[0, 0])
    
    assert(np.allclose(dx_dy_nm, dx_dy_an, rtol=1e-3, atol=1e-5))
    assert(np.allclose(dy_dx_nm, dy_dx_an, rtol=1e-3, atol=1e-5))

    assert(np.allclose(d2x_dy2_nm, d2x_dy2_an, rtol=1e-3, atol=1e-5))
    assert(np.allclose(d2y_dx2_nm, d2y_dx2_an, rtol=1e-3, atol=1e-5))
    
    
    
