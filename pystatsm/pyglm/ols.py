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
from ..utilities.linalg_operations import wdiag_outer_prod, diag_outer_prod



class OLS:
    
    def __init__(self, formula=None, data=None, X=None, y=None):
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
        self.formula = formula
        self.data = data
        U, S, Vt= np.linalg.svd(X, full_matrices=False)
        self.U, self.S, self.V = U, S, Vt.T
        self.h = diag_outer_prod(U, U)
        self.W = (self.V / self.S).dot(self.U.T)
        self.vif = 1.0/np.diag((self.V * 1.0 / self.S**2).dot(self.V.T))
        self.condition_indices = np.max(self.S**2) / self.S**2
        self.vdp = np.sum(self.V**2, axis=0) / self.S**2
        self.cond = np.vstack((self.vif, self.condition_indices, self.vdp)).T
        self.cond = pd.DataFrame(self.cond, index=self.xcols,
                                 columns=["VIF", "Cond", "VDP"])
    
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
    
    def fit(self, var_beta=None):
        beta, beta_se = self._fit_mats(self.X, self.y)
        self.beta = beta
        yhat = self.X.dot(beta)
        resids = self.y - yhat
        if var_beta is not None:
            if type(var_beta) is str:
                var_beta = [var_beta]
            rse = {}
            for vb in var_beta:
                _, _, rse[vb] = self.robust_rcov(vb, resids=resids, W=self.W)
        else:
            rse = None
        
        res = pd.DataFrame(np.vstack((beta, beta_se)).T, index=self.xcols, columns=['beta', 'SE'])            
        if rse is not None:
            for key, val in rse.items():
                res["SE_"+key] = val
        res['t'] = res['beta'] / res['SE']
        if rse is not None:
            for key, val in rse.items():
                res["t_"+key] = res['beta'] / res["SE_"+key]
        res['p'] = sp.stats.t(self.n-self.p).sf(np.abs(res['t']))*2.0
        if rse is not None:
            for key, val in rse.items():
                res["p_"+key] = sp.stats.t(self.n-self.p).sf(np.abs(res['t_'+key]))*2.0
        dfr = self.n - self.p
        dfm = self.p - 1
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
        self.Ginv = np.dot(self.Linv.T, self.Linv)
        self.Vbeta = self.Ginv * msr
        self.s2 = msr
        self.resids = resids
    
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
        pbar = tqdm.tqdm(total=n_perms, smoothing=0.001) if verbose else None
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
        pbar = tqdm.tqdm(total=n_perms, smoothing=0.001) if verbose else None
        p_values = np.zeros(len(vars_of_interest))
        p_values_fwer = np.zeros(len(vars_of_interest))
        abst = np.abs(self.tvalues[vars_of_interest])
        
        ixc = np.setdiff1d(np.arange(self.p), vars_of_interest)
        Xc, y = self.X[:, ixc], self.y
        g, _ = self._fit_mats(Xc, y)
        u = Xc.dot(g)
        r = y - u
        for i in range(n_perms):
            b, se = self._fit_y(self.L, self.Linv, self.X, u + r[np.random.permutation(self.n)])
            abstp = np.abs(b / se)[vars_of_interest]
            p_values_fwer += (abstp.max()>abst) / n_perms
            p_values +=  (abstp>abst) / n_perms
            if verbose:
                pbar.update(1)
        if verbose:
            pbar.close()
        return p_values_fwer, p_values
        
    def freedman_lane(self, vars_of_interest=None, n_perms=5000, verbose=True):
        if hasattr(self, 'res')==False:
            self.fit()
        if vars_of_interest is None:
            vars_of_interest = np.arange(self.p)
            if "Intercept" in self.res.index:
                ii = np.ones(self.p).astype(bool)
                ii[self.res.index.get_loc("Intercept")] = False
                vars_of_interest = vars_of_interest[ii]
        vars_of_interest = np.arange(self.p) if vars_of_interest is None else vars_of_interest
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
        
    def _bootstrap(self, n_boot, ssq=False):
        pbar = tqdm.tqdm(total=n_boot, smoothing=0.001)
        beta_samples = np.zeros((n_boot, self.p))
        beta_se_samples =  np.zeros((n_boot, self.p))
        if ssq:
            ssq_samples = np.zeros((n_boot, 13))
        else:
            ssq_samples = None
        i = 0
        while i < n_boot:
            try:
                ix = np.random.choice(self.n, self.n)
                Xb, Yb = self.X[ix],  self.y[ix]
                beta_samples[i], beta_se_samples[i] = self._fit_mats(Xb, Yb)
                if ssq:
                    ssq_samples[i] = sumsq(Yb, Xb.dot(beta_samples[i]), self.p)
                i += 1
                pbar.update(1)
            except np.linalg.LinAlgError:
                pass
        pbar.close()
        # for i in range(n_boot):
        #     ix = np.random.choice(self.n, self.n)
        #     Xb, Yb = self.X[ix],  self.y[ix]
        #     beta_samples[i], beta_se_samples[i] = self._fit_mats(Xb, Yb)
        #     pbar.update(1)
        # pbar.close()
        return beta_samples, beta_se_samples, ssq_samples
    
    def bootstrap(self, n_boot=5_000, ssq=False):
        if hasattr(self, 'res')==False:
            self.fit()
        self.beta_samples, self.beta_se_samples, self.ssq_samples = self._bootstrap(n_boot, ssq)
        self.res.insert(self.res.columns.get_loc("SE")+1, "SE_boot", self.beta_samples.std(axis=0))
        self.res.insert(self.res.columns.get_loc("t")+1, "t_boot", self.res['beta']/self.res['SE_boot'])
        self.res.insert(self.res.columns.get_loc("p")+1, "p_boot",  
                        sp.stats.t(self.n-self.p).sf(np.abs(self.res['t_boot']))*2.0)
        if self.ssq_samples is not None:
            self.ssq_samples = pd.DataFrame(
                self.ssq_samples, columns=[
                    "SSR", "SSM", "SST", "dfr", "dfm", "dft", "MSR", "MSM", "MST",
                    "Rsquared", "Rsquared_Adj", "F", "p"
                    ]
                )

    def print_results(self):
        opt_cont = ('display.max_rows', None, 'display.max_columns', None,
                    'display.float_format', '{:.4f}'.format)
        with pd.option_context(*opt_cont):
            print(self.res)
    
    def robust_rcov(self, kind=None, resids=None, Ginv=None, X=None, U=None, 
                    W=None):
        kind = "HC3" if kind is None else kind
        if Ginv is None:
            if hasattr(self, "Ginv"):
                Ginv = self.Ginv
            else:
                Ginv = np.dot(self.Linv.T, self.Linv)
        if resids is None:
            if hasattr(self, "resids"):
                resids = self.resids
            else:
                resids = self.y - self.X.dot(self.beta)
        if U is None:
            if X is None:
                U, S, V, h = self.U, self.S, self.V, self.h
            else:
                U, S, Vt = np.linalg.svd(X, full_matrices=False)
                V = Vt.T
                h = diag_outer_prod(U, U)
 
        X = self.X if X is None else X
            
        n, p = X.shape
        u = resids**2
        if kind == "HC0":
            omega = u
        elif kind == "HC1":
            omega = n / (n - p) * u
        elif kind == "HC2":
            omega = u / (1.0 - h)
        elif kind == "HC3":
            omega = u / (1 - h)**2
        elif kind == "HC4":
            omega = u / (1.0 - h)**np.minimum(4.0, h / np.mean(h))
        
        if W is None:
            W = (V / S).dot(U.T)
        se = np.sqrt(diag_outer_prod(W*omega, W))
        return W, omega, se
            
    def get_influence(self, r=None, h=None):
        h = self.h if h is None else h
        r = self.resids if r is None else r
        ssr = np.sum(r**2)
        dfe = self.n - self.p - 1.0
        msr = ssr / dfe
        n = self.n
        p = self.p
        s = np.sqrt(msr)
        s_i = np.sqrt((ssr-r**2) / (dfe-1))
        
        r_students = r / (s_i * np.sqrt(1.0 - h))
        dfbeta = r / (1.0 - h) * self.W
        h_tilde = 1.0 / n + h

        cov_ratio = (s_i / s)**(2 * p) * 1.0 / (1.0 - h_tilde)

        
        d_residuals = r / (1.0 - h_tilde)
        
        s_star_i = 1/np.sqrt(n-p-1)*np.sqrt((n-p)*s**2/(1-h_tilde)-d_residuals**2)
        s_i = s_star_i * np.sqrt(1.0 - h_tilde)
        
        studentized_d_residuals = d_residuals / s_star_i
                
        dfits = r * np.sqrt(h) / (s_i * (1.0 -h))
        
        cooks_distance = d_residuals**2 * h_tilde / (s**2 * (p + 1))
        
        res = dict(r_students=r_students, cov_ratio=cov_ratio,
                   s_i=s_i, studentized_d_residuals=studentized_d_residuals,
                   dfits=dfits, cooks_distance=cooks_distance)
        res = pd.DataFrame(res)
        dfbeta = pd.DataFrame(dfbeta.T, columns=self.xcols)
        res = pd.concat([res, dfbeta], axis=1)
        res["leverage"] = h
        return res
    
    def loglike(self, params, X=None, y=None):
        X = self.X if X is None else X
        y = self.y if y is None else y
        if len(params)==(self.p+1):
            beta = params[:-1]
            sigma = np.exp(params[-1])
        else:
            beta = params
            r = y - X.dot(beta)
            sigma = np.sqrt(np.dot(r, r) / (X.shape[0]))
        
        n = X.shape[0]
        
        const = -n / 2.0 * np.log(2.0 * np.pi)
        
        lndetS = -n / 2.0 * np.log(sigma**2)
        
        r = y - X.dot(beta)
        
        ll = const + lndetS - 1.0 / (2.0 * sigma**2) * np.dot(r, r)
        return ll
        
    def predict(self, beta=None, X=None):
        beta = self.beta if beta is None else beta
        X = self.X if X is None else X
        y_hat = np.dot(X, beta)
        return y_hat
    
    def _compute_jackknife_coefs(self):
        beta = self.beta[None]
        X = self.X
        XtXi = np.dot(self.Linv.T, self.Linv)
        w = (self.resids / (1.0 - self.h))[:, None]
        dbeta = np.dot(X, XtXi) * w
        beta_jackknife = beta - dbeta
        beta_jackknife_mean = np.mean(beta_jackknife, axis=0)
        beta_jackknife_z = beta_jackknife - beta_jackknife_mean
        self.beta_jackknife = beta_jackknife
        self.beta_jackknife_mean = beta_jackknife_mean
        self.beta_jackknife_z = beta_jackknife_z
        
    
        
        
  
def sumsq(y, yhat, p):
    n = len(y)
    dfr = n - p
    dfm = p - 1
    dft = n - 1
    
    resids = y - yhat
    ymean = np.mean(y)
    ssr = np.sum(resids**2)
    ssm = np.sum((yhat - ymean)**2)
    sst = np.sum((y - ymean)**2)
    
    msr = ssr / dfr
    msm = ssm / dfm
    mst = sst / dft
    
    rsquared = 1.0 - ssr / sst
    rsquared_adj = 1.0 - (msr / mst)     
   
    fvalue = msm / msr
    fpval = sp.stats.f(dfm, dfr).sf(fvalue)
    res = np.array([ssr, ssm, sst, dfr, dfm, dft, msr, msm, mst, rsquared, rsquared_adj, fvalue, fpval])
    return res
      
        
        
        
    
    
            
            
