# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 22:40:59 2021

@author: lukepinkel
"""


import numpy as np
import scipy as sp
import pandas as pd
from pystats.pygam.gauls import GauLS
from pystats.utilities.linalg_operations import dummy

rng = np.random.default_rng(123)
            
n_obs = 20000
df = pd.DataFrame(np.zeros((n_obs, 4)), columns=['x0', 'x1', 'x2', 'y'])
 
df['x0'] = rng.choice(np.arange(5), size=n_obs, p=np.ones(5)/5)
df['x1'] = rng.uniform(-1, 1, size=n_obs)
df['x2'] = rng.uniform(-1, 1, size=n_obs)
df['x3'] = rng.uniform(-1, 1, size=n_obs)

u0 =  dummy(df['x0']).dot(np.array([-0.2, 0.2, -0.2, 0.2, 0.0]))
f1 = (3.0 * df['x1']**3 - 2.43 * df['x1'])
f2 = -(3.0 * df['x2']**3 - 2.43 * df['x2']) 
f3 = (df['x3'] - 1.0) * (df['x3'] + 1.0)
eta =  u0 + f1 + f2
mu = eta.copy() 
tau = 1.0 / (np.exp(f3) + 0.01)
df['y'] = rng.normal(loc=mu, scale=tau)


mod = GauLS("y~C(x0)+s(x1, kind='cr')+s(x2, kind='cr')", "y~1+s(x3, kind='cr')", df)
mod.fit(opt_kws=dict(options=dict(verbose=3)))

mod.plot_smooth_comp(mod.beta, single_fig=False)
mod.plot_smooth_quantiles('x1', 'x3')
mod.plot_smooth_quantiles('x2', 'x3')

np.allclose(np.array([3.915687, 4.138739, 7.299964]), mod.theta)