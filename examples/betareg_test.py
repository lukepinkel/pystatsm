# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 20:42:02 2021

@author: lukepinkel
"""

import numpy as np
import pandas as pd
from pystats.utilities.random_corr import exact_rmvnorm
from pystats.pyglm.betareg import BetaReg, LogitLink, LogLink
from pystats.utilities.numerical_derivs import fo_fc_cd, so_gc_cd


seed = 1234
rng = np.random.default_rng(seed)
n_obs = 10_000
X = exact_rmvnorm(np.eye(4)/100, n=n_obs, seed=seed)
Z = exact_rmvnorm(np.eye(2)/100, n=n_obs, seed=seed)
betam = np.array([4.0, 1.0, -1.0, -2.0])
betas = np.array([2.0, -2.0])
etam, etas = X.dot(betam)+1.0, 2+Z.dot(betas)#np.tanh(Z.dot(betas))/2.0 + 2.4
mu, phi = LogitLink().inv_link(etam), LogLink().inv_link(etas)     
a = mu * phi
b = (1.0 - mu) * phi
y = rng.beta(a, b)
#sns.distplot(y)
#sns.jointplot(phi, mu)

xcols = [f"x{i}" for i in range(1, 4+1)]
zcols = [f"z{i}" for i in range(1, 2+1)]
data = pd.DataFrame(np.hstack((X, Z)), columns=xcols+zcols)
data["y"] = y

m_formula = "y~1+"+"+".join(xcols)
s_formula = "y~1+"+"+".join(zcols)

model = BetaReg(m_formula=m_formula, s_formula=s_formula, data=data)
model.fit()
theta = model.theta.copy()

g1 = fo_fc_cd(model.loglike, theta*0.95)
g2 = model.gradient(theta*0.95)
np.allclose(g1, g2)

H1 = so_gc_cd(model.gradient, theta)
H2 = model.hessian(theta)

np.allclose(H1, H2)
