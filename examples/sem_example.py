# -*- coding: utf-8 -*-
"""
Created on Mon May 17 07:31:12 2021

@author: lukepinkel
"""

import numpy as np
import scipy as sp
import scipy.stats
import pandas as pd
from pystats.pysem.sem import SEM, invech
from pystats.utilities.random_corr import exact_rmvnorm
data_bollen = pd.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/sem/Bollen.csv", index_col=0)
data_bollen = data_bollen[['x1', 'x2', 'x3', 'y1', 'y2', 'y3', 'y4', 'y5',
             'y6', 'y7', 'y8', ]]

L = np.array([[1., 0., 0.],
              [1., 0., 0.],
              [1., 0., 0.],
              [0., 1., 0.],
              [0., 1., 0.],
              [0., 1., 0.],
              [0., 1., 0.],
              [0., 0., 1.],
              [0., 0., 1.],
              [0., 0., 1.],
              [0., 0., 1.]])

B = np.array([[0., 0., 0.],
              [1., 0., 0.],
              [1., 1., 0.]])

Lambda1 = pd.DataFrame(L, index=data_bollen.columns, columns=['ind60', 'dem60', 'dem65'])
Beta1 = pd.DataFrame(B, index=Lambda1.columns, columns=Lambda1.columns)
S1 = data_bollen.cov()
Psi1 = pd.DataFrame(np.eye(Lambda1.shape[0]), index=Lambda1.index, columns=Lambda1.index)

off_diag = [['y1', 'y5'], ['y2', 'y4'], ['y3', 'y7'], ['y4', 'y8'],
            ['y6', 'y8'], ['y2', 'y6']]
for x, y in off_diag:
    Psi1.loc[x, y] = Psi1.loc[y, x] = 0.05
Phi1 = pd.DataFrame(np.eye(Lambda1.shape[1]), index=Lambda1.columns,
                   columns=Lambda1.columns)

model = SEM(Lambda1, Beta1, Phi1, Psi1, data=data_bollen)
model.fit()




################################################################


x = sp.stats.norm(0, 1).rvs(1000)
x = (x - x.mean()) / x.std()

u = sp.stats.norm(0, 1).rvs(1000)
u = (u - u.mean()) / u.std()
v = sp.stats.norm(0, 1).rvs(1000)
v = (v - v.mean()) / v.std()

m = 0.5*x + u
y = 0.7*m + v

data_path = pd.DataFrame(np.vstack((x, m, y)).T, columns=['x', 'm', 'y'])
data_path = data_path -data_path.mean(axis=0)
Lambda2 = pd.DataFrame(np.eye(3), index=data_path.columns, columns=data_path.columns)
Beta2 = np.array([[0.0, 0.0, 0.0],
                  [1.0, 0.0, 0.0],
                  [1.0, 1.0, 0.0]])
Beta2 = pd.DataFrame(Beta2, index=data_path.columns, columns=data_path.columns)

Phi2 = np.array([[1.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0]])
Phi2 = pd.DataFrame(Phi2, index=data_path.columns, columns=data_path.columns)

Psi2 = np.array([[0.0, 0.0, 0.0],
                 [0.0, 1.0, 0.0],
                 [0.0, 0.0, 1.0]])
Psi2 = pd.DataFrame(Psi2, index=data_path.columns, columns=data_path.columns)


model2 = SEM(Lambda2, Beta2, Phi2, Psi2, data=data_path)
model2.fit(opt_kws=dict(options=dict(gtol=1e-20, xtol=1e-100)))



################################################################


data_holzinger = pd.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/sem/HS.data.csv")
df_holzinger = data_holzinger.iloc[:, 7:]
Lambda3 = np.zeros((9, 3))
Lambda3[0:3, 0] = 1.0
Lambda3[3:6, 1] = 1.0
Lambda3[6:9, 2] = 1.0

Lambda3 = pd.DataFrame(Lambda3, index=df_holzinger.columns, columns=['visual', 'textual', 'speed'])
Beta3 = pd.DataFrame(np.zeros((3, 3)), index=Lambda3.columns, columns=Lambda3.columns)
Phi3 = Beta3 + np.eye(3) + 0.001
Psi3 = pd.DataFrame(np.eye(9), index=df_holzinger.columns, columns=df_holzinger.columns)
model3 = SEM(Lambda3, Beta3, Phi=Phi3, Psi=Psi3, data=df_holzinger)
model3.fit()


################################################################

pd.set_option('display.expand_frame_repr', False)
vechS = [2.926, 1.390, 1.698, 1.628, 1.240, 0.592, 0.929,
         0.659, 4.257, 2.781, 2.437, 0.789, 1.890, 1.278, 0.949,
         4.536, 2.979, 0.903, 1.419, 1.900, 1.731, 5.605, 1.278, 1.004,
         1.000, 2.420, 3.208, 1.706, 1.567, 0.988, 3.994, 1.654, 1.170,
         3.583, 1.146, 3.649]

S4 = pd.DataFrame(invech(np.array(vechS)), columns=['anti1', 'anti2',
                 'anti3', 'anti4', 'dep1', 'dep2', 'dep3', 'dep4'])
S4.index = S4.columns

data4 = exact_rmvnorm(S4.values, 180)
data4-= data4.mean(axis=0)
data4 += np.array([1.750, 1.928, 1.978, 2.322, 2.178, 2.489, 2.294, 2.222])

data4 = pd.DataFrame(data4, columns=S4.columns)

Lambda4 = pd.DataFrame(np.eye(8), index=data4.columns, columns=data4.columns)
Lambda4 = Lambda4.iloc[[1, 2, 3, 5, 6, 7, 0, 4], [1, 2, 3, 5, 6, 7, 0, 4]]

Beta4 = pd.DataFrame([[0, 0, 0, 0, 0, 0, 1, 1],
                      [1, 0, 0, 1, 0, 0, 0, 0],
                      [0, 1, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1, 1],
                      [1, 0, 0, 1, 0, 0, 0, 0],
                      [0, 1, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      ], index=Lambda4.columns, columns=Lambda4.columns)
Phi4   = pd.DataFrame([[1.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 1.0, 0.0, 0.0, 0.1, 0.0, 0.0],
                       [0.1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.1, 0.0, 0.0, 1.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.1],
                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 1.0],
                       ], index=Lambda4.columns, columns=Lambda4.columns)

Psi4 = Lambda4.copy()*0.0
data4 = data4.loc[:, Lambda4.index]
model4 = SEM(Lambda4, Beta4, Phi=Phi4, Psi=Psi4, data=data4)
model4.fit()

################################################################

Lambda5 = np.array([[ 1.0,  0.0,  0.0,  0.0],
                    [ 2.0,  0.0,  0.0,  0.0],
                    [-1.0,  0.0,  0.0,  0.0],
                    [ 0.0,  1.0,  0.0,  0.0],
                    [ 0.0,  2.0,  0.0,  0.0],
                    [ 0.0, -1.0,  0.0,  0.0],
                    [ 0.0,  1.0,  0.0,  0.0],
                    [ 0.0,  0.0,  1.0,  0.0],
                    [ 0.0,  0.0,  0.5,  0.0],
                    [ 0.0,  0.0,  0.2,  0.0],
                    [ 0.0,  0.0,  1.0,  0.0],
                    [ 0.0,  0.0,  0.0,  1.0]])

Beta5 = np.array([[ 0.0, 0.0, 0.0, 0.0],
                  [ 1.0, 0.0, 0.0, 0.0],
                  [-1.0, 1.0, 0.0, 0.0],
                  [ 0.0, 0.5, 1.0, 0.0]])

Phi5 = np.array([[1.0, 0.0, 0.0, 0.0],
                 [0.0, 1.0, 0.0, 0.0],
                 [0.0, 0.0, 1.0, 0.0],
                 [0.0, 0.0, 0.0, 1.0]])

Psi5a = np.eye(12)
IB = np.linalg.inv(np.eye(4) - Beta5)
S5 = Lambda5.dot(IB).dot(Phi5).dot(IB.T).dot(Lambda5.T) + Psi5a
data5 = pd.DataFrame(exact_rmvnorm(S5, 1000), columns=[f"x{i}" for i in range(1, 13)])
data5 = pd.DataFrame(np.random.multivariate_normal(np.zeros(12), cov=S5, size=(1000,)), 
                     columns=[f"x{i}" for i in range(1, 13)])
data5 = data5 - data5.mean(axis=0)

Lambda5 = pd.DataFrame(Lambda5, index=data5.columns, columns=[f"z{i}" for i in range(1, 5)])
Beta5 = pd.DataFrame(Beta5, index=Lambda5.columns, columns=Lambda5.columns)
Phi5 = pd.DataFrame(Phi5, index=Lambda5.columns, columns=Lambda5.columns)

model5a = SEM(Lambda5, Beta5, Phi5, Psi5a, data=data5)
model5a.fit()

Psi5b = np.eye(12)
Psi5b[-1, -1] = 0.0

model5b = SEM(Lambda5, Beta5, Phi5, Psi5b, data=data5)
model5b.fit()


