#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 18:21:36 2020

@author: lukepinkel
"""
import tqdm # analysis:ignore
import patsy # analysis:ignore
import numpy as np # analysis:ignore
import scipy as sp # analysis:ignore
import scipy.stats # analysis:ignore
import pandas as pd # analysis:ignore


class OLS:
    
    def __init__(self, X=None, y=None, formula=None, data=None):
        if formula is not None and data is not None:
            y, X = patsy.dmatrices(formula, data=data, return_type='dataframe')
            xcols, xinds = X.columns, X.index
            ycols, yinds = y.columns, y.index
            X, y = X.values, y.values[:, 0]
        elif X is not None and y is not None:
            if type(X) not in [pd.DataFrame, pd.Series]:
                xcols = [f'x{i}' for i in range(1, X.shape[1]+1)]
                xinds = np.arange(X.shape[0])
            else:
                xcols, xinds = X.columns, X.index
                X = X.values
            if type(y) not in [pd.DataFrame, pd.Series]:
                ycols = ['y']
                yinds = np.arange(y.shape[0])
            else:
                 ycols, yinds = y.columns, y.index
                 y = y.values
        
        if X.ndim==1:
            X = X[:, None]
        self.X, self.y = X, y
        self.xcols, self.xinds = xcols, xinds
        self.ycols, self.yinds = ycols, yinds
        self.G = X.T.dot(X)
        self.L = np.linalg.cholesky(self.G)
        self.Linv = np.linalg.inv(self.L)
        self.n, self.p = X.shape[0], X.shape[1]
        self.ymean = self.y.mean()
        
    
    def _fit_mats(self, X, y):
        n, p = X.shape
        G = X.T.dot(X)
        c = X.T.dot(y)
        L = np.linalg.cholesky(G)
        w = sp.linalg.solve_triangular(L, c, lower=True)
        s2 =  (y.T.dot(y) - w.T.dot(w)) / (n - p)
        beta = sp.linalg.solve_triangular(L.T, w, lower=False)
        Linv = np.linalg.inv(L)
        Ginv = np.diag(np.dot(Linv.T, Linv))
        beta_se = s2 * Ginv
        return beta, np.sqrt(beta_se)
    
    def _fit_y(self, L, Linv, X, y):
        n, p = X.shape
        c = X.T.dot(y)
        w = sp.linalg.solve_triangular(L, c, lower=True)
        s2 =  (y.T.dot(y) - w.T.dot(w)) / (n - p)
        beta = sp.linalg.solve_triangular(L.T, w, lower=False)
        Ginv = np.diag(np.dot(Linv.T, Linv))
        beta_se = s2 * Ginv
        return beta, np.sqrt(beta_se)
    
    def fit(self):
        beta, beta_se = self._fit_mats(self.X, self.y)
        res = pd.DataFrame(np.vstack((beta, beta_se)).T,
                                index=self.xcols, columns=['beta', 'SE'])
        res['t'] = res['beta'] / res['SE']
        res['p'] = sp.stats.t(self.n-self.p).sf(np.abs(res['t']))*2.0
        yhat = self.X.dot(beta)
        resids = self.y - yhat
        
        dfr = self.n - self.p - 1
        dfm = self.p
        dft = self.n - 1
        
        ssr = np.sum(resids**2)
        ssm = np.sum((yhat - self.ymean)**2)
        sst = np.sum((self.y - self.ymean)**2)
        
        msr = ssr / dfr
        msm = ssm / dfm
        mst = sst / dft
        
        rsquared = 1.0 - ssr / sst
        rsquared_adj = 1.0 - (msr / mst)
        
        fvalue = msm / msr
        fpval = sp.stats.f(dfm, dfr).sf(fvalue)
        
        self.sumstats = pd.DataFrame([[rsquared, '-'], 
                                      [rsquared_adj, '-'],
                                      [fvalue, fpval]])
        self.sumstats.index =['R2', 'R2 Adj', 'F test']
        self.sumstats.columns = ['Statistic', 'P']
        ssq_ind = ['Residuals','Model',' Total']
        ssq_col = ['Sum Square', 'Degrees of Freedom', 'Mean Square']
        self.ssq = pd.DataFrame([[ssr, dfr, msr], 
                                 [ssm, dfm, msm],
                                 [sst, dft, mst]], index=ssq_ind, columns=ssq_col)
        self.res = res
        self.beta, self.beta_se = beta, beta_se
        self.tvalues = self.beta / self.beta_se
    
    def _permutation_test_store(self,n_perms, L, Linv, X, y, verbose):
        pbar = tqdm.tqdm(total=n_perms) if verbose else None
        t_samples = np.zeros((n_perms, self.p))
        for i in range(n_perms):
            b, se = self._fit_y(L, Linv, X, y[np.random.permutation(self.n)])
            t_samples[i] = b / se
            if verbose:
                pbar.update(1)
        if verbose:
            pbar.close()
        return t_samples
        
    def _permutation_test(self, n_perms, L, Linv, X, y, verbose):
        pbar = tqdm.tqdm(total=n_perms) if verbose else None
        p_values = np.zeros((self.p))
        p_values_fwer = np.zeros((self.p))
        abst = np.abs(self.tvalues)
        for i in range(n_perms):
            b, se = self._fit_y(L, Linv, X, y[np.random.permutation(self.n)])
            abstp = np.abs(b / se)
            p_values_fwer += (abstp.max()>abst) / n_perms
            p_values +=  (abstp>abst) / n_perms
            if verbose:
                pbar.update(1)
        if verbose:
            pbar.close()
        return p_values_fwer, p_values
    
    def _freedman_lane(self, vars_of_interest, n_perms=5000, verbose=True):
        pbar = tqdm.tqdm(total=n_perms) if verbose else None
        p_values = np.zeros(len(vars_of_interest))
        p_values_fwer = np.zeros(len(vars_of_interest))
        abst = np.abs(self.tvalues[vars_of_interest])
        
        ixc = np.setdiff1d(np.arange(self.p), vars_of_interest)
        Xi, Xc, y = self.X[:, vars_of_interest], self.X[:, ixc], self.y
        L = np.linalg.cholesky(Xi.T.dot(Xi))
        Linv = np.linalg.inv(L)
        g, _ = self._fit_mats(Xc, y)
        u = Xc.dot(g)
        r = y - u
        for i in range(n_perms):
            b, se = self._fit_y(L, Linv, Xi, u + r[np.random.permutation(self.n)])
            abstp = np.abs(b / se)
            p_values_fwer += (abstp.max()>abst) / n_perms
            p_values +=  (abstp>abst) / n_perms
            if verbose:
                pbar.update(1)
        if verbose:
            pbar.close()
        return p_values_fwer, p_values
        
    def freedman_lane(self, vars_of_interest, n_perms=5000, verbose=True):
        if hasattr(self, 'res')==False:
            self.fit()
        pvals_fwer, pvals = self._freedman_lane(vars_of_interest, n_perms, verbose)
        rows = self.res.index[vars_of_interest]
        self.res['freedman_lane_p'] = '-'
        self.res['freedman_lane_p_fwer'] = '-'
        self.res.loc[rows, 'freedman_lane_p'] = pvals
        self.res.loc[rows, 'freedman_lane_p_fwer'] = pvals_fwer
        
    def permutation_test(self, n_perms=5_000, store_samples=False, verbose=True):
        if hasattr(self, 'res')==False:
            self.fit()
        L, Linv, X, y = self.L, self.Linv, self.X, self.y
        if store_samples:
            t_samples = self._permutation_test_store(n_perms, L, Linv, X, y, verbose)
            abst = t_samples
            p_values_fwer = (abst.max(axis=1)>np.abs(self.tvalues)).sum(axis=0)/n_perms
            p_values = (abst > np.abs(self.tvalues)).sum(axis=0) / n_perms
        else:
            p_values_fwer, p_values =  self._permutation_test(n_perms, L, Linv, X, y, verbose)
            t_samples = None
        self.permutation_t_samples = t_samples
        self.res['permutation_p'] = p_values
        self.res['permutation_p_fwer'] = p_values_fwer
        
    def _bootstrap(self, n_boot):
        pbar = tqdm.tqdm(total=n_boot)
        beta_samples = np.zeros((n_boot, self.p))
        beta_se_samples =  np.zeros((n_boot, self.p))
        for i in range(n_boot):
            ix = np.random.choice(self.n, self.n)
            Xb, Yb = self.X[ix],  self.y[ix]
            beta_samples[i], beta_se_samples[i] = self._fit_mats(Xb, Yb)
            pbar.update(1)
        pbar.close()
        return beta_samples, beta_se_samples
    
    def bootstrap(self, n_boot=5_000):
        if hasattr(self, 'res')==False:
            self.fit()
        self.beta_samples, self.beta_se_samples = self._bootstrap(n_boot)
        self.res.insert(2, "SE_boot", self.beta_samples.std(axis=0))
        self.res.insert(4, "t_boot", self.res['beta']/self.res['SE_boot'])
        self.res.insert(6, "p_boot",  
                        sp.stats.t(self.n-self.p).sf(np.abs(self.res['t_boot']))*2.0)
        

    def print_results(self):
        opt_cont = ('display.max_rows', None, 'display.max_columns', None,
                    'display.float_format', '{:.4f}'.format)
        with pd.option_context(*opt_cont):
            print(self.res)
        
