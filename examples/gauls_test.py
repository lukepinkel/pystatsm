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
pd.set_option("display.max_columns", 20)
pd.set_option("display.max_rows", 200)
pd.set_option('display.width', 1000)
rng = np.random.default_rng(123)


n_obs = 20000
df = pd.DataFrame(np.zeros((n_obs, 4)), columns=['x0', 'x1', 'x2', 'y'])
 
df['x0'] = rng.choice(np.arange(5), size=n_obs, p=np.ones(5)/5)
df['x1'] = rng.uniform(-1, 1, size=n_obs)
df['x2'] = rng.uniform(-1, 1, size=n_obs)

u0 =  dummy(df['x0']).dot(np.array([-0.2, 0.2, -0.2, 0.2, 0.0]))
f1 = (3.0 * df['x1']**3 - 2.43 * df['x1'])
f2 = -(3.0 * df['x2']**3 - 2.43 * df['x2']) 
f3 = (df['x2'] - 1.0) * (df['x2'] + 1.0)
eta =  u0 + f1 + f2
mu = eta.copy() 
tau = 1.0 / (np.exp(f3) + 0.01)
df['y'] = rng.normal(loc=mu, scale=tau)


mod = GauLS("y~C(x0)+s(x1, kind='cr')+s(x2, kind='cr')", "y~1+s(x2, kind='cr')", df)
mod.fit(opt_kws=dict(options=dict(verbose=3)))

mod.plot_smooth_comp(mod.beta, single_fig=False)
mod.plot_smooth_quantiles('x1', 'x2')
mod.plot_smooth_quantiles('x2', 'x2')

np.allclose(np.array([4.235378, 4.165951, 7.181457]), mod.theta)

print(mod.res)
print(mod.res_smooths)

         
n_obs = 20_000
df = pd.DataFrame(np.zeros((n_obs, 4)), columns=['x0', 'x1', 'x2', 'y'])

df['x0'] = rng.choice(np.arange(5), size=n_obs, p=np.ones(5)/5)
df['x1'] = rng.uniform(-1, 1, size=n_obs)
df['x2'] = rng.uniform(-1, 1, size=n_obs)

u0 =  dummy(df['x0']).dot(np.array([-0.2, 0.2, -0.2, 0.2, 0.0]))
f1 = (4.0 * df['x1']**3 - 4.0 * df['x1'])
f2 = -(4.0 * df['x2']**3 - 4.0 * df['x2']) 
f3 = -(df['x2'] - 1.0) * (df['x2'] + 1.0) - 0.7
eta =  u0 + f1 + f2
mu = eta.copy() 
scale =  (np.exp(f3) + 0.01)
tau = 1.0 / scale
df['y'] = rng.normal(loc=mu, scale=scale)


mod = GauLS("y~C(x0)+s(x1, df=20, kind='cr')+s(x2, df=20, kind='cr')",
            "y~1+s(x2, df=20, kind='cr')", df)
mod.fit(opt_kws=dict(options=dict(verbose=3)))

mod.plot_smooth_comp(mod.beta, single_fig=False)
mod.plot_smooth_quantiles('x1', 'x2')
mod.plot_smooth_quantiles('x2', 'x2', quantiles=np.insert(np.arange(5, 50, 5), 0, 1),
                          b=mod.res.loc["s(Intercept)", "param"])

np.allclose(np.array([6.46996035292, 6.34966698854, 9.57254338164]), mod.theta)

print(mod.res)
print(mod.res_smooths)
print(mod.sumstats)
