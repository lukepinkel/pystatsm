#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 20:08:47 2020

@author: lukepinkel
"""


import numpy as np 
import scipy as sp
import scipy.stats 
import pandas as pd
from ..utilities.data_utils import _check_shape
from ..utilities.linalg_operations import vec, invec, vech, invech
from ..utilities.special_mats import dmat, lmat, nmat, kmat



class SEM:
    
    def __init__(self, Lambda, Beta, Phi, Psi, data=None, S=None, 
                 indicator_vars=None):
        """
        Structural Equation Model
        
        Parameters
        ----------
        Lambda : dataframe
            A dataframe specifying measurement model structure, 
            with rows corresponding to observed variables and columns
            corresponding to potential latent variables.
        
        Beta : dataframe
            A dataframe specifying the structural model.  If variable
            i is being regressed onto a set of variables including variable j
            then the (i,j)th entry of Beta is nonzero.
        
        Phi : dataframe
            Dataframe specifying covariance of the (potentially) latent 
            variables.  For path models, variances are nonzero for
            variables that are not being regressed onto others.
        
        Psi : dataframe
            Dataframe specifying residual correlations.
        
        data : dataframe, optional if provided S
            A dataframe of size (n_ob s x n_vars) 
        
        S : dataframe, optional if data is passed
            Dataframe of the observed covariance matrix
        
        indicator_vars : dataframe, optional
            For a model with free latent variable covariances, 
            identification requires a fixed indicator variable, which is
            by default taken as the first specified variable
        
        Notes
        -----
        In order to specify the model, for each free parameter, set the 
        corresponding matrix element to a valid nonzero value 
        (i.e. for covariance matrices its usually best practice to set
         off diags to small values).  Fixed nonzero values aren't yet 
        technically supported, but they can be set by modifying the
        attribute 'params_template', setting the corresponding element
        to the desired fixed value.
        
        """
        if S is None:
            S = data.cov(ddof=0)
        Lambda = Lambda.loc[S.index] #Align loadings and variable order
        
        n_mvars, n_lvars = Lambda.shape 
        
        #If covariance is free, then an indicator variable needs to be set
        #Defaults to the first variable that loads onto the each latent var 
        
        LA = np.asarray(Lambda)
        BE = np.asarray(Beta)
        PH = np.asarray(Phi)
        PS = np.asarray(Psi)
        
        LAf = Lambda.copy() 
        
        if indicator_vars is None:
            for i, var in enumerate(Lambda.columns):
                if (Phi.loc[var, var]!=0.0) or ((LAf.loc[:, var]!=0).sum()==1):
                    vi = np.argmax(LAf.loc[:, var])
                    LAf.iloc[vi, i] = 0.0
        else:
            LAf = LAf - indicator_vars
        
        LAf = np.asarray(LAf)
        BEf = np.asarray(Beta.copy())
        PSf = np.asarray(Psi.copy())
        PHf = np.asarray(Phi.copy())
        
        theta_indices = np.concatenate([vec(LAf), vec(BEf),  vech(PHf), 
                                        vech(PSf)])!=0
        
        params_template = np.concatenate([vec(LA), vec(BE), 
                                          vech(PH), vech(PS)])
        theta = params_template[theta_indices]
        
        param_parts = np.cumsum([0, n_mvars*n_lvars,
                       n_lvars*n_lvars,
                       (n_lvars+1)*n_lvars//2,
                       (n_mvars+1)*n_mvars//2])
        slices = dict(LA=np.s_[param_parts[0]:param_parts[1]],
                      BE=np.s_[param_parts[1]:param_parts[2]],
                      PH=np.s_[param_parts[2]:param_parts[3]],
                      PS=np.s_[param_parts[3]:param_parts[4]])
        mat_shapes = dict(LA=(n_mvars, n_lvars),
                          BE=(n_lvars, n_lvars),
                          PH=(n_lvars, n_lvars),
                          PS=(n_mvars, n_mvars))
        
                
        ovn, svn = Lambda.index, Lambda.columns
        
        ix1, ix2 = np.where(LAf)
        lambda_labels = [f"{x}<-{y}" for x,y in list(zip(ovn[ix1], svn[ix2]))]
        
        ix1, ix2 = np.where(BEf)
        beta_labels = [f"{x}~{y}" for x,y in list(zip(svn[ix1], svn[ix2]))]
        
        ix1, ix2 = [], []
        v = vech(np.asarray(PHf))!=0
        for i, (x, y) in enumerate(list(zip(*np.triu_indices(PHf.shape[0])))):
            if v[i]==True:
                ix1.append(x)
                ix2.append(y)
            
        
        phi_labels = [f"cov({x}, {y})" for x,y in list(zip(svn[ix1], svn[ix2]))]
        
        
        ix1, ix2 = [], []
        v = vech(np.asarray(PSf))!=0
        for i, (x, y) in enumerate(list(zip(*np.triu_indices(PSf.shape[0])))):
            if v[i]==True:
                ix1.append(x)
                ix2.append(y)
            
        psi_labels = [f"residual cov({x}, {y})" for x,y in list(zip(ovn[ix1], ovn[ix2]))]
        
        
        self.labels = lambda_labels + beta_labels + phi_labels + psi_labels


        self.LA, self.BE, self.PH, self.PS, self.S = LA, BE, PH, PS, np.asarray(S)
        self.theta_indices = theta_indices
        self.params_template = params_template
        self.theta = theta
        self.param_parts = param_parts
        self.slices = slices
        self.mat_shapes = mat_shapes
        self.n_mvars, self.n_lvars = n_mvars, n_lvars
        self.n_params = len(params_template)
        self.I_nlvars = np.eye(n_lvars)
        self.Lp = lmat(self.n_mvars).toarray()
        self.Np = nmat(self.n_mvars).toarray()
        self.Ip = np.eye(self.n_mvars)
        self.Dk = dmat(self.n_lvars).toarray()
        self.Kq = kmat(self.n_lvars, self.n_lvars).toarray()
        self.Kp = kmat(self.n_mvars, self.n_lvars).toarray()
        self.Kkp = kmat(self.n_lvars, self.n_mvars).toarray()
        self.Dp = dmat(self.n_mvars).toarray()
        self.E = np.zeros((self.n_mvars, self.n_mvars))
        self.Ip2 = np.eye(self.n_mvars**2)
        self.DPsi = np.linalg.multi_dot([self.Lp, self.Ip2, self.Dp])        
        self.LpNp = np.dot(self.Lp, self.Np)
        self.bounds = np.concatenate([vec(np.zeros(self.LA.shape)), 
                                      vec(np.zeros(self.BE.shape)),
                                     vech(np.eye(self.PH.shape[0])),
                                     vech(np.eye(self.PS.shape[0]))])
        self.bounds = self.bounds[self.theta_indices]
        self.bounds = [(None, None) if x==0 else (0, None) for x in self.bounds]
        self.n_obs = data.shape[0]
        self.data = data
        self.p = self.S.shape[0]
        self.df_cov = self.p * (self.p+1) // 2
        self._default_null_model = {"Sigma":np.diag(np.diag(self.S)),
                                    "df":self.df_cov - self.p,
                                    "n_free_params":self.p}
        _, self.ldS = np.linalg.slogdet(self.S)
        
    def model_matrices(self, theta):
        """
        Parameters
        ----------
        
        theta : array_like
            Vector of free parameters
        
        
        Returns
        -------
        LA : ndarray
            Matrix of shape (p x q) specifying latent variable
            loadings
        
        BE : ndarray
            Matrix of shape (q x q) specifying the structural model
        
        IB : ndarray
            Matrix of shape (q x q) equal to the inverse of (I-BE)
        
        PH : ndarray
            Matrix of shape (q x q) containing the latent variable covariances
        
        PS : ndarray
            Matrix of shape (p x p) containing the residual covariance matrix
        
        """
        theta = _check_shape(theta, 1)
        params = self.params_template.copy()
        if theta.dtype==complex:
            params = params.astype(complex)
        params[self.theta_indices] = theta
        LA = invec(params[self.slices['LA']], *self.mat_shapes['LA'])
        BE = invec(params[self.slices['BE']], *self.mat_shapes['BE'])
        IB = np.linalg.pinv(self.I_nlvars - BE)
        PH = invech(params[self.slices['PH']])
        PS = invech(params[self.slices['PS']])
        return LA, BE, IB, PH, PS
    
    def implied_cov(self, theta):
        """
        Parameters
        ----------
        theta : array_like
            Vector of free parameters
        
        
        Returns
        -------
        Sigma: ndarray
            Matrix of size (p x p) containing the covariance matrix implied
            by the model evaluated at theta
            
        
        """
        LA, _, IB, PH, PS = self.model_matrices(theta)
        A = LA.dot(IB)
        Sigma = A.dot(PH).dot(A.T) + PS
        return Sigma
    
    def _dsigma(self, LA, BE, IB, PH, PS):
        """
        Parameters
        ----------
        LA : ndarray
            Matrix of shape (p x q) specifying latent variable
            loadings
        
        BE : ndarray
            Matrix of shape (q x q) specifying the structural model
        
        IB : ndarray
            Matrix of shape (q x q) equal to the inverse of (I-BE)
        
        PH : ndarray
            Matrix of shape (q x q) containing the latent variable covariances
        
        PS : ndarray
            Matrix of shape (p x p) containing the residual covariance matrix
        
        Returns
        -------
        G: ndarray
            Matrix containing the derivatives of Sigma with respect to 
            each of the model matrices
        
        Notes
        -----
        
        The matrix G has structure [DLambda, DBeta, DPhi, DPsi] with each
        sub-matrix containing the derivatives of the lower half unique components
        of Sigma with respect to a model matrix.  Each matrix shares the first
        dimension p(p+1)/2. The second dimension of each matrix is, in the 
        same order as in G, pq, q^2, q(q+1)/2, p(p+1)/2
        """
        A = np.dot(LA, IB)
        B = np.linalg.multi_dot([A, PH, IB.T])
        DLambda = np.dot(self.LpNp, np.kron(B, self.Ip))
        DBeta = np.dot(self.LpNp, np.kron(B, A))
        DPhi = np.linalg.multi_dot([self.Lp, np.kron(A, A), self.Dk])
        DPsi = self.DPsi        
        G = np.block([DLambda, DBeta, DPhi, DPsi])
        return G
    
    def dsigma(self, theta):
        """
        Parameters
        ----------
        theta : array_like
            Vector of free parameters
        
        Returns
        -------
        G: ndarray
            Matrix containing the derivatives of Sigma with respect to 
            each of the model matrices
        
        Notes
        -----
        
        The matrix G has structure [DLambda, DBeta, DPhi, DPsi] with each
        sub-matrix containing the derivatives of the lower half unique components
        of Sigma with respect to a model matrix.  Each matrix shares the first
        dimension p(p+1)/2. The second dimension of each matrix is, in the 
        same order as in G, pq, q^2, q(q+1)/2, p(p+1)/2
        """
        LA, BE, IB, PH, PS = self.model_matrices(theta)
        return self._dsigma(LA, BE, IB, PH, PS)
        
    
    def _gradient(self, theta):
        """
        Parameters
        ----------
        theta : array_like
            Vector of free parameters
        
        Returns
        -------
        g : ndarray
            Derivative of the negative log likelihood with respect to the
            model matrices 
        """
        LA, BE, IB, PH, PS = self.model_matrices(theta)
        A = LA.dot(IB)
        Sigma = A.dot(PH).dot(A.T) + PS
        Sigma_inv = np.linalg.pinv(Sigma)
        G = self.dsigma(theta)
        W = 0.5 * self.Dp.T.dot(np.kron(Sigma_inv, Sigma_inv)).dot(self.Dp)
        d = vech(self.S - Sigma)[:, None]
        g = -2.0 * G.T.dot(W).dot(d)
        return g
    
    def gradient(self, theta):
        """
        Parameters
        ----------
        theta : array_like
            Vector of free parameters
        
        Returns
        -------
        g : ndarray
            Derivative of the negative log likelihood with respect to the
            free parameters
        """
        g = self._gradient(theta)[self.theta_indices, 0]
        return g
    
    def _hessian(self, theta):
        """
        Parameters
        ----------
        theta : array_like
            Vector of free parameters
        
        Returns
        -------
        H : ndarray
            Second derivatives of the negative log likelihood with respect to 
            the model matrices
        """
        LA, BE, IB, PH, PS = self.model_matrices(theta)
        A = LA.dot(IB)
        Sigma = A.dot(PH).dot(A.T) + PS
        Sigma_inv = np.linalg.pinv(Sigma)
        Sdiff = self.S - Sigma
        d = vech(Sdiff)
        G = self.dsigma(theta)
        DGp = self.Dp.dot(G)
        W1 = np.kron(Sigma_inv, Sigma_inv)
        W2 = np.kron(Sigma_inv, Sigma_inv.dot(Sdiff).dot(Sigma_inv))
        H1 = 0.5 * DGp.T.dot(W1).dot(DGp)
        H2 = 1.0 * DGp.T.dot(W2).dot(DGp)

        Hpp = []
        U, A = IB.dot(PH).dot(IB.T), LA.dot(IB)
        Q = LA.dot(U)
        Kp, Kq, D, Dp, E = self.Kkp, self.Kq, self.Dk, self.Dp, self.E
        Hij = np.zeros((self.n_params, self.n_params))
        for i in range(self.n_mvars):
            for j in range(i, self.n_mvars):
                E[i, j] = 1.0
                T = E + E.T
                TA = T.dot(A)
                AtTQ = A.T.dot(T).dot(Q)
                AtTA = A.T.dot(TA)
                
                H11 = np.kron(U, T)
                H22 = np.kron(AtTQ.T, IB.T).dot(Kq)+Kq.dot(np.kron(AtTQ, IB))\
                      +np.kron(U, AtTA)
                H12 = (np.kron(U, TA)) + Kp.dot(np.kron(T.dot(Q), IB))
                H13 =  np.kron(IB, TA).dot(D) 
                H23 = D.T.dot(np.kron(IB.T, AtTA))
                
                
                Hij[self.slices['LA'], self.slices['LA']] = H11
                Hij[self.slices['BE'], self.slices['BE']] = H22
                Hij[self.slices['LA'], self.slices['BE']] = H12
                Hij[self.slices['LA'], self.slices['PH']] = H13
                Hij[self.slices['PH'], self.slices['BE']] = H23
                Hij[self.slices['BE'], self.slices['LA']] = H12.T
                Hij[self.slices['PH'], self.slices['LA']] = H13.T
                Hij[self.slices['BE'], self.slices['PH']] = H23.T  
                E[i, j] = 0.0
                Hpp.append(Hij[:, :, None])
                Hij = Hij*0.0
        W = np.linalg.multi_dot([Dp.T, W1, Dp])
        dW = np.dot(d, W)
        Hp = np.concatenate(Hpp, axis=2) 
        H3 = np.einsum('k,ijk ->ij', dW, Hp)      
        H = (H1 + H2 - H3 / 2.0)*2.0
        return H
    
    def hessian(self, theta):
        """
        Parameters
        ----------
        theta : array_like
            Vector of free parameters
        
        Returns
        -------
        H : ndarray
            Second derivatives of the negative log likelihood with respect to 
            the free parameters
        """
        return self._hessian(theta)[self.theta_indices][:, self.theta_indices]
    
    def loglike(self, theta):
        """
        Parameters
        ----------
        theta : array_like
            Vector of free parameters
        
        Returns
        -------
        LL : float
            Twice the negative log likelihood
        """
        LA, _, IB, PH, PS = self.model_matrices(theta)
        A = LA.dot(IB)
        Sigma = A.dot(PH).dot(A.T) + PS
        Sigma_inv = np.linalg.pinv(Sigma)
        LL = np.linalg.slogdet(Sigma)[1] + np.trace(self.S.dot(Sigma_inv))
        return LL
    
    def _goodness_of_fit(self, model_dict):
        """
        Parameters
        ----------
        model_dict : dict
            Dictionary with the key value pairs ('Sigma', implied covariance)
            (df, model degrees of freedom), and 
            ('n_free_params', number of free parameters)
        
        Returns
        -------
        overall_fit_measures : dict
            Dictionary containing goodness of fit measures
        """
        Sigma, df = model_dict['Sigma'], model_dict['df']
        t = model_dict["n_free_params"]
        _, ldSigma = np.linalg.slogdet(Sigma)
        SV = np.linalg.pinv(Sigma).dot(self.S)
        trSV = np.trace(SV)
        LL = ldSigma + trSV
        LLF = self.n_obs * LL + self.n_obs * self.p * np.log(2.0 * np.pi)
        fval =  LL - self.ldS - self.p
        Chi2 = (self.n_obs - 1) * fval
        Chi2_pval = sp.stats.chi2(t).sf(Chi2)
        GFI = 1.0 - np.trace((SV - self.Ip).dot(SV-self.Ip)) / np.trace(SV.dot(SV))
        if df!=0:
            AGFI = 1.0 - (self.df_cov / df) *  (1.0 - GFI)
            Standardized_Chi2 = (Chi2 - df) / np.sqrt(2.0 * df)
            RMSEA = np.sqrt(np.maximum(Chi2 - df, 0) / (df * (self.n_obs - 1)))
        else:
            AGFI = np.nan
            Standardized_Chi2 = np.nan
            RMSEA = np.nan
        CN01, CN05 = sp.stats.chi2(df).ppf([0.95, 0.99]) / fval + 1.0

        AIC = LLF + 2.0  * t
        BIC = LLF + np.log(self.n_obs) * t
        EVCI = LL + 2.0 * t / (self.n_obs - self.p - 2.0)
        resids = Sigma - self.S
        v = np.diag(np.sqrt(1.0 / np.diag(self.S)))
        std_sq_resids = vech(v.dot(resids**2).dot(v))
        resids = vech(resids)
        
        RMR = np.sqrt(np.mean(resids**2))
        SRMR = np.sqrt(np.mean(std_sq_resids))
        
        overall_fit_measures = dict(loglikelihood=-LLF/2.0, Chi2=Chi2, 
                                    Chi2_pval=Chi2_pval,GFI=GFI, AGFI=AGFI,
                                    CN01=CN01, CN05=CN05, Standardized_Chi2=Standardized_Chi2,
                                    AIC=AIC, BIC=BIC, EVCI=EVCI, RMR=RMR, SRMR=SRMR, 
                                    RMSEA=RMSEA)
        return overall_fit_measures         
    
    def fit(self, null_model=None, use_hess=False, opt_kws={}):
        """
        Parameters
        ----------
        null_model : dict
            Dictionary specifying the model to use when computing relative
            goodness of fit measures
        
        use_hess : bool
            If true the analytical hessian is used during optimization.
        
        opt_kws : dict
            Optimizer options for use in scipy.optimize.minimize
            
        """
        hess = self.hessian if use_hess else None
        null_model = self._default_null_model if null_model is None else null_model
        
        theta = self.theta.copy()
        opt = sp.optimize.minimize(self.loglike, theta, jac=self.gradient,
                                   hess=hess, method='trust-constr',
                                   bounds=self.bounds, **opt_kws)
        theta = opt.x
        theta_cov = np.linalg.pinv(self.hessian(theta))*2.0 / self.n_obs
        theta_se = np.sqrt(np.diag(theta_cov))
        t_values = theta / theta_se
        p_values = sp.stats.t(self.n_obs).sf(np.abs(t_values))*2.0
        
        res = np.vstack((theta, theta_se, t_values, p_values)).T
        res = pd.DataFrame(res, columns=['est', 'SE', 't', 'p'],
                           index=self.labels)
        
        Sigma = self.implied_cov(theta)
        full_model = dict(Sigma=Sigma, df=self.df_cov - len(theta),
                          n_free_params=len(theta))
        null_fit_indices = self._goodness_of_fit(null_model)
        full_fit_indices = self._goodness_of_fit(full_model)
        sumstats = full_fit_indices.copy()
        Chi2n, Chi2f = null_fit_indices["Chi2"], full_fit_indices["Chi2"]
        dfn, dff = null_model["df"], full_model["df"]
        sumstats["NFI1"] = (Chi2n - Chi2f) / Chi2n
        sumstats["NFI2"]= (Chi2n - Chi2f) / (Chi2n - dff)
        if dff!=0:
            sumstats["Rho1"] = (Chi2n / dfn - Chi2f / dff) / (Chi2n / dfn)
            sumstats["Rho2"] = (Chi2n / dfn - Chi2f / dff) / (Chi2n / dfn - 1.0)
        else:
            sumstats["Rho1"], sumstats["Rho2"] = np.nan, np.nan
        sumstats["CFI"] = 1.0 - np.maximum(Chi2f - dff, 0) / np.maximum(Chi2f - dff, Chi2n - dfn)
        self._null_fit_indices = null_fit_indices
        self._full_fit_indices = full_fit_indices
        self.sumstats = sumstats
        self.Sigma = Sigma
        
        self.opt = opt
        self.theta = theta
        self.theta_cov = theta_cov
        self.theta_se = theta_se
        self.t_values = t_values
        self.p_values = p_values
        self.res = res

        
        