 

import numpy as np
import pandas as pd
from pystatsm.utilities.numerical_derivs import jac_approx
from pystatsm.pyglm2.zinf import ZeroInflatedPoisson
from pystatsm.pyglm2.model_sim import LinearModelSim


def test_zeroinflatedpoisson():
    seed = 23644
    rng = np.random.default_rng(seed)
    n_obs = 1_000
    formula = "1+C(x0)+x1+x2+x3"
    zero_formula = "1+C(x4)+x5"
    
    
    msim = LinearModelSim.from_formula(formula, n_obs, 
                                       nlevels={"x0":5},
                                       coef_kws=dict(p_nnz=(1., 1.,)),
                                       corr_kws={"eig_kws":{"p_eff":0.5, "a":4}},
                                       rng=rng,
                                       seed=seed)
    
    zero_msim = LinearModelSim.from_formula(zero_formula, n_obs, 
                                       nlevels={"x4":5},
                                       coef_kws=dict(p_nnz=(1., 1.,)),
                                       corr_kws={"eig_kws":{"p_eff":0.5, "a":4}},
                                       rng=rng,
                                       seed=seed)
    
    data = pd.concat([msim.data, zero_msim.data],axis=1)
    data["linpred"] = msim.linpred#msim._simulate_y(msim.linpred, resid_var)
    data["zero_linpred"] = zero_msim.linpred#msim._simulate_y(msim.linpred, resid_var)
    
    
    data["zero_mu"] = np.exp(data["zero_linpred"]) / (1.0 + np.exp(data["zero_linpred"]))
    
    ybin = rng.binomial(1, p=data["zero_mu"])
    
    y = np.zeros(len(ybin), dtype=float)
    
    mu = np.exp(data["linpred"])
    q =  rng.poisson(mu[ybin==1])
    y[ybin==1] = q*1.0
    data["y"] = y
    
    model = ZeroInflatedPoisson("y~"+formula, "~"+zero_formula, data=data)
    model.fit()
    params = model.params.copy()*0.0
    
    
    model.loglike(params)
    model.loglike_i(params)
    
    assert(np.allclose(model.loglike(params), np.sum(model.loglike_i(params))))
    
    params = model.params.copy()*0.9
    
    gi_nm = jac_approx(model.loglike_i, params)
    g_nm = jac_approx(model.loglike, params)
    H_nm = jac_approx(model.gradient, params)
    
    
    gi_an = model.gradient_i(params)
    g_an = model.gradient(params)
    H_an = model.hessian(params)
    
    
    assert(np.allclose(gi_nm, gi_an, atol=1e-6))
    assert(np.allclose(g_nm, g_an))
    assert(np.allclose(H_nm, H_an, atol=1e-6))
    
    
    H_an = model.hessian(params*0.0)
    H_nm = jac_approx(model.gradient, params*0.0)
    
    assert(np.allclose(H_nm, H_an, atol=1e-6))
    
    
    H_an = model.hessian(params*2)
    H_nm = jac_approx(model.gradient,params*2)
    
    g_an = model.gradient(params*2)
    g_nm = jac_approx(model.loglike, params*2)
    
    
    assert(np.allclose(H_nm, H_an, atol=1e-6))
    
