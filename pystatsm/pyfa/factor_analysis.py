# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 18:11:20 2021

@author: lukepinkel
"""

import numpy as np
import scipy as sp
import scipy.stats
import pandas as pd
from .rotation import rotate, oblique_constraint_derivs
from ..utilities.linalg_operations import vec, invec, vech, vecl, invecl, vecl_inds
from ..utilities.special_mats import nmat, dmat, lmat
from ..utilities.numerical_derivs import so_gc_cd
from ..utilities.data_utils import _check_type, cov, eighs, flip_signs
from .fit_measures import srmr, lr_test, gfi, agfi


class FactorAnalysis(object):
   
    
    def __init__(self, X=None, S=None, n_obs=None, n_factors=None, 
                 rotation_method=None, **n_factor_kws):
        """
        Exploratory Factor Analysis
        
        Parameters
        ----------
        X : dataframe, optional
            A dataframe of size (n x p) containing the n observations and p
            variables to be analyzed
        
        S : arraylike
            Dataframe or ndarray containing (p x p) variables to be analyzed 
            
        
        n_obs : int or float, optional
            If X is not provided, and the covariance matrix S is, then 
            n_obs  is required to compute test statistics
        
        n_factors : int float or str, optional
            The number of factors to model.  If a string, must be one of
            'proportion' or 'eigmin'.
        
        rotation_method : str, optional
            One of the rotation methods, e.g. quartimax, varimax, equamax
        
        Keyword Arguments:
            
            proportion : float
                Float in [0, 1] specifying the cutoff variance explained when
                calculating the number of factors to model
            
            eigmin : float
                The eigenvalue cutoff when determining the number of factors
        
        
        Notes
        -----
        In the docs, p will be used to denote the the number of variables, 
        and q the number of factors, t the number of parameters, k the 
        number of covariance parameters to model, and m the number of
        parameters in the explicit 'augmented' model
        
        t = p * (q + 1)
        k = p * (p + 1) // 2
        m = p * (q + 1) + q * (q - 1) // 2 = t + q * (q - 1) // 2
        
        """
        self._process_data(X, S, n_obs)
        self._get_n_factors(n_factors, **n_factor_kws)
        self._rotation_method = rotation_method
        self._make_params()
        self.E = np.eye(self.n_vars)
        self.Ik = np.eye(self.n_facs)
        self.Nk = nmat(self.n_facs).A
        self.Lk = lmat(self.n_facs).A
        self.Ip = np.eye(self.n_vars)
        self.Dp = dmat(self.n_vars).A
        self.Lp = lmat(self.n_vars).A
        self.Np = nmat(self.n_vars).A
        self.LpNp = np.dot(self.Lp, self.Np)
        self.d_inds = vec(self.Ip)==1
        self.l_inds = vecl_inds(self.n_facs)

    def _process_data(self, X, S, n_obs):
        given_x = X is not None
        given_s = S is not None
        given_n = n_obs is not None
        if given_x and not given_s:
            X, cols, inds, _is_pd = _check_type(X)
            S = cov(X)
            n_obs, n_vars = X.shape
        if not given_x and given_s and given_n:
            S, cols, _, _is_pd = _check_type(S)
            n_vars = S.shape[0]
            inds = np.arange(n_obs)
        u, V = eighs(S)
        V = flip_signs(V)
        self.X, self.cols, self.inds, self._is_pd = X, cols, inds, _is_pd
        self.S, self.V, self.u = S, V, u
        self.cols, self.inds, self._is_pd = cols, inds, _is_pd
        self.n_obs, self.n_vars = n_obs, n_vars
        
    def _get_n_factors(self, n_factors, proportion=0.6, eigmin=1.0):
        if type(n_factors) is str:
            if n_factors == 'proportion':
                n_factors = np.sum((self.u.cumsum()/self.u.sum())<proportion)+1
            elif n_factors == 'eigmin':
                n_factors = np.sum(self.u>eigmin)
        elif type(n_factors) in [float, int]:
            n_factors = int(n_factors)
        self.n_facs = self.n_factors = n_factors
        
    def _make_params(self):
        self.n_pars = self.n_vars*self.n_facs + self.n_vars
        self.theta = np.zeros(self.n_pars)
        self.lix=np.arange(self.n_vars*self.n_facs)
        self.pix =np.arange(self.n_vars*self.n_facs, self.n_vars*self.n_facs+self.n_vars)
        L = self.V[:, :self.n_facs]
        psi = np.diag(self.S - np.dot(L, L.T))
        self.theta[self.lix] = vec(L)
        self.theta[self.pix] = np.log(psi)
    
    def model_matrices(self, theta):
        """

        Parameters
        ----------
        theta : ndarray
            ndarray of length t containing model parameters.

        Returns
        -------
        L : ndarray
            (p x q) matrix of loadings.
        Psi : ndarray
            (p x p) diagonal matrix of residual covariances.

        """
        L = invec(theta[self.lix], self.n_vars, self.n_facs)
        Psi = np.diag(np.exp(theta[self.pix]))
        return L, Psi
    
    def implied_cov(self, theta):
        """
        
        Parameters
        ----------
        theta : ndarray
             ndarray of length t containing model parameters.

        Returns
        -------
        Sigma : ndarray
            (p x p) implied covariance matrix.

        """
        L, Psi = self.model_matrices(theta)
        Sigma = L.dot(L.T) + Psi
        return Sigma
    
    def loglike(self, theta):
        """
        

        Parameters
        ----------
        theta : ndarray
             ndarray of length t containing model parameters.

        Returns
        -------
        ll : float
            Loglikelihood of the model.

        """
        Sigma = self.implied_cov(theta)
        _, lndS = np.linalg.slogdet(Sigma)
        trSV = np.trace(np.linalg.solve(Sigma, self.S))
        ll = lndS + trSV
        return ll
    
    def gradient(self, theta):
        """
        
        Parameters
        ----------
        theta : ndarray
             ndarray of length t containing model parameters.

        Returns
        -------
        g : ndarray
             ndarray of length t containing the derivatives of the 
             loglikelihood with respect to theta.

        """
        L, Psi = self.model_matrices(theta)
        Sigma = L.dot(L.T) + Psi
        V = np.linalg.pinv(Sigma)
        VRV = V.dot(Sigma - self.S).dot(V)
        g1 = 2 * vec(VRV.dot(L))
        g2 = np.diag(VRV.dot(Psi))
        g = np.zeros(self.n_pars)
        g[self.lix] = g1
        g[self.pix] = g2
        return g
    
    def hessian_approx(self, theta):
        H = so_gc_cd(self.gradient, theta)
        return H
        
    def dsigma(self, theta):
        """
        
        Parameters
        ----------
        theta : ndarray
             ndarray of length t containing model parameters.

        Returns
        -------
        G : ndarray
            (k x t) matrix of derivatives of the implied covariance with 
            respect to parameters

        """
        L, Psi = self.model_matrices(theta)
        DLambda = np.dot(self.LpNp, np.kron(L, self.Ip))
        DPsi = np.dot(self.Lp, np.diag(vec(Psi)))[:, self.d_inds]
        G = np.block([DLambda, DPsi])
        return G
    
    def hessian(self, theta):
        """
        

        Parameters
        ----------
        theta : ndarray
             ndarray of length t containing model parameters.

        Returns
        -------
        H : ndarray
             (t x t) matrix of second derivative of the log likelihood with 
             respect to the parameters

        """
        L, Psi = self.model_matrices(theta)
        Sigma = L.dot(L.T) + Psi
        Sigma_inv = np.linalg.inv(Sigma)
        Sdiff = self.S - Sigma
        d = vech(Sdiff)
        G = self.dsigma(theta)
        DGp = self.Dp.dot(G)
        W1 = np.kron(Sigma_inv, Sigma_inv)
        W2 = np.kron(Sigma_inv, Sigma_inv.dot(Sdiff).dot(Sigma_inv))
        H1 = 0.5 * DGp.T.dot(W1).dot(DGp)
        H2 = 1.0 * DGp.T.dot(W2).dot(DGp)

        Hpp = []
        Dp, Ik, E = self.Dp, self.Ik, self.E
        Hij = np.zeros((self.n_pars, self.n_pars))
        for i in range(self.n_vars):
            for j in range(i, self.n_vars):
                eij = np.zeros(self.n_vars)
                if i==j:
                    eij[i] = 1.0
                E[i, j] = 1.0
                T = E + E.T
                H11 = np.kron(Ik, T)
                H22 = np.diag(Psi) * eij
                Hij[self.lix, self.lix[:, None]] = H11
                Hij[self.pix, self.pix] = H22
                E[i, j] = 0.0
                Hpp.append(Hij[:, :, None])
                Hij = Hij*0.0
        W = np.linalg.multi_dot([Dp.T, W1, Dp])
        dW = np.dot(d, W)
        Hp = np.concatenate(Hpp, axis=2) 
        H3 = np.einsum('k,ijk ->ij', dW, Hp)      
        H = (H1 + H2 - H3 / 2.0)*2.0
        return H
    
    def _make_augmented_params(self, L, Phi, Psi):
        """
        

        Parameters
        ----------
        L : ndarray
            (p x q) array of loadings.
        Phi : ndarray
            (q x q) factor covariance matrix.
        Psi : ndarray
            (p x p) diagonal matrix of residual covariances.

        Returns
        -------
        None.
        
        
        Notes
        -----
        
        If rotation_method is not None, then a params vector is added to
        account for the restricted parameters when computing the hessian.  
        Makes implicit assumptions about factor covariance explicit by 
        augmenting theta with (q - 1) * q // 2 factor covariance parameters.

        """
        p, q = self.n_vars, self.n_facs
        nl = p * q
        nc = q * (q - 1) // 2  if self._rotation_method is not None else 0
        nr = p
        nt = nl + nc + nr
        params = np.zeros(nt)
        ixl = np.arange(nl)
        ixc = np.arange(nl, nl+nc)
        ixr = np.arange(nl+nc, nl+nc+nr)
        params[ixl] = vec(L)
        if self._rotation_method is not None:
            params[ixc] = vecl(Phi)
        params[ixr] = np.diag(Psi)
        
        self.nl, self.nc, self.nr, self.nt = nl, nc, nr, nt
        self.ixl, self.ixc, self.ixr = ixl, ixc, ixr
        self.params = params
    
    def model_matrices_augmented(self, params):
        """
        

        Parameters
        ----------
        params : ndarray
            ndarray of length m containing model parameters..

        Returns
        -------
        L : ndarray
            (p x q) array of loadings.
        Phi : ndarray
            (q x q) factor covariance matrix..
        Psi : ndarray
            (p x p) diagonal matrix of residual covariances.

        """
        L = invec(params[self.ixl], self.n_vars, self.n_facs)
        if self._rotation_method is not None:
            Phi = invecl(params[self.ixc])
        else:
            Phi = np.eye(self.n_facs)
        Psi = np.diag(params[self.ixr])
        return L, Phi, Psi
    
    def dsigma_augmented(self, params):
        """
        

        Parameters
        ----------
        params : ndarray
            ndarray of length m containing model parameters.

        Returns
        -------
        G : ndarray
            (k x m) matrix of derivatives of the implied covariance with 
            respect to parameters

        """
        L, Phi, Psi = self.model_matrices_augmented(params)
        DLambda = np.dot(self.LpNp, np.kron(L.dot(Phi), self.Ip))
        DPhi = self.Lp.dot(np.kron(L, L))[:, self.l_inds]
        DPsi = np.dot(self.Lp, np.diag(vec(np.eye(self.n_vars))))[:, self.d_inds]
        if self._rotation_method is not None:
            G = np.block([DLambda, DPhi, DPsi])
        else:
            G = np.block([DLambda, DPsi])
        return G
    
    def implied_cov_augmented(self, params):
        """
        

        Parameters
        ----------
        params : ndarray
            ndarray of length m containing model parameters.


        Returns
        -------
        Sigma : ndarray
            (p x p) implied covariance matrix.

        """
        L, Phi, Psi = self.model_matrices_augmented(params)
        Sigma = L.dot(Phi).dot(L.T) + Psi
        return Sigma
    
    def loglike_augmented(self, params):
        """
        

        Parameters
        ----------
        params : ndarray
            ndarray of length m containing model parameters.

        Returns
        -------
        ll : float
            Loglikelihood of the model.

        """
        Sigma = self.implied_cov_augmented(params)
        _, lndS = np.linalg.slogdet(Sigma)
        trSV = np.trace(np.linalg.solve(Sigma, self.S))
        ll = lndS + trSV
        return ll
    
    def gradient_augmented(self, params):
        """
        
        Parameters
        ----------
        params : ndarray
             ndarray of length m containing model parameters.

        Returns
        -------
        g : ndarray
             ndarray of length m containing the derivatives of the 
             loglikelihood with respect to params.

        """
        L, Phi, Psi = self.model_matrices_augmented(params)
        Sigma = L.dot(Phi).dot(L.T) + Psi
        V = np.linalg.pinv(Sigma)
        VRV = V.dot(Sigma - self.S).dot(V)
        gL = 2 * vec(VRV.dot(L.dot(Phi)))
        gPhi = 2*vecl(L.T.dot(VRV).dot(L))
        gPsi = np.diag(VRV)
        g = np.zeros(self.nt)
        g[self.ixl] = gL
        if self._rotation_method is not None:
            g[self.ixc] = gPhi
        g[self.ixr] = gPsi
        return g
    
    def hessian_augmented(self, params):
        """
        

        Parameters
        ----------
        params : ndarray
             ndarray of length m containing model parameters.

        Returns
        -------
        H : ndarray
             (m x m) matrix of second derivative of the log likelihood with 
             respect to params

        """
        L, Phi, Psi = self.model_matrices_augmented(params)
        Sigma = L.dot(Phi).dot(L.T) + Psi
        Sigma_inv = np.linalg.inv(Sigma)
        Sdiff = self.S - Sigma
        d = vech(Sdiff)
        G = self.dsigma_augmented(params)
        DGp = self.Dp.dot(G)
        W1 = np.kron(Sigma_inv, Sigma_inv)
        W2 = np.kron(Sigma_inv, Sigma_inv.dot(Sdiff).dot(Sigma_inv))
        H1 = 0.5 * DGp.T.dot(W1).dot(DGp)
        H2 = 1.0 * DGp.T.dot(W2).dot(DGp)

        Hpp = []
        Dp, Ik, E = self.Dp, self.Ik, self.E
        Hij = np.zeros((self.nt, self.nt))
        for i in range(self.n_vars):
            for j in range(i, self.n_vars):
                E[i, j] = 1.0
                T = E + E.T
                H11 = np.kron(Phi, T)
                H22 = np.kron(Ik, T.dot(L))[:, self.l_inds]
                Hij[self.ixl, self.ixl[:, None]] = H11
                Hij[self.ixl, self.ixc[:, None]] = H22.T
                Hij[self.ixc, self.ixl[:, None]] = H22
                E[i, j] = 0.0
                Hpp.append(Hij[:, :, None])
                Hij = Hij*0.0
        W = np.linalg.multi_dot([Dp.T, W1, Dp])
        dW = np.dot(d, W)
        Hp = np.concatenate(Hpp, axis=2) 
        H3 = np.einsum('k,ijk ->ij', dW, Hp)      
        H = (H1 + H2 - H3 / 2.0)*2.0
        return H
    
    def _unrotated_constraint_dervs(self, L, Psi):
        """
        

        Parameters
        ----------
        L : ndarray
            (p x q) matrix of loadings.
        Psi : ndarray
            (p x p) diagonal matrix of residual covariances.

        Returns
        -------
        J : ndarray
            (q * (q - 1) // 2 x t) matrix of derivatives of the constraints for
            the orthogonal model.

        """
        Psi_inv = np.diag(1.0 / np.diag(Psi))
        A = L.T.dot(Psi_inv)
        J1 = self.Lk.dot(self.Nk.dot(np.kron(self.Ik, A)))
        J2 =-self.Lk.dot(np.kron(A, A)[:, self.d_inds])
        J = np.concatenate([J1, J2], axis=1)
        i, j = np.tril_indices(self.n_facs)
        J = J[i!=j]
        return J
        
    def constraint_derivs(self, theta):
        """
        

        Parameters
        ----------
        theta : ndarray
             ndarray of length t containing model parameters.

        Returns
        -------
        J : ndarray
            (q * (q - 1) // 2 x t) matrix of derivatives of the constraints for
            the orthogonal model.

        """
        L, Psi = self.model_matrices(theta)            
        J = self._unrotated_constraint_dervs(L, Psi)
        return J
    
    def _reorder_factors(self, L, T, theta):
        """
        

        Parameters
        ----------
        L : ndarray
            (p x q) matrix of loadings.
        T : ndarray
            (q x q)  rotation matrix

        theta :  ndarray
             ndarray of length t containing model parameters.

        Returns
        -------
        L : ndarray
            (p x q) matrix of loadings.
        T : ndarray
            (q x q)  rotation matrix

        theta :  ndarray
             ndarray of length t containing model parameters.


        """
        v = np.sum(L**2, axis=0)
        order = np.argsort(v)[::-1]
        if type(L) is pd.DataFrame:
            cols = L.columns
            L = L.iloc[:, order]
            j = np.argmax(np.abs(L), axis=0)
            s = np.sign(L[j, np.arange(L.shape[1])])
            L = s * L
            L.columns = cols
            theta[self.lix] = vec(L.values)
        else:
            L = L[:, order]
            j = np.argmax(np.abs(L), axis=0)
            s = np.sign(L[j, np.arange(L.shape[1])])
            L = s * L
            theta[self.lix] = vec(L)
        T = T[:, order] * s[:, None]
        return L, T, theta
        
    def _fit_indices(self, Sigma):
        """
        

        Parameters
        ----------
        Sigma : ndarray
            (p x p) implied covariance matrix.

        Returns
        -------
        sumstats : dict

        """
        t = (self.n_vars + 1.0) * self.n_vars / 2.0
        k = len(self.theta)
        degfree = t  - k
        GFI = gfi(Sigma, self.S)
        AGFI = agfi(Sigma, self.S, degfree)
        chi2, chi2p = lr_test(Sigma, self.S, degfree, self.n_obs-1)
        stdchi2 = (chi2 - degfree) /  np.sqrt(2 * degfree)
        RMSEA = np.sqrt(np.maximum(chi2 -degfree, 0)/(degfree*(self.n_obs-1)))
        SRMR = srmr(Sigma, self.S, degfree)
        llval = -(self.n_obs-1)*self.loglike(self.theta)/2
        BIC = k * np.log(self.n_obs) -2 * llval
        AIC = 2 * k - 2*llval
        sumstats = dict(GFI=GFI, AGFI=AGFI, chi2=chi2, chi2_pvalue=chi2p,
                         std_chi2=stdchi2, RMSEA=RMSEA, SRMR=SRMR,
                         loglikelihood=llval,
                         AIC=AIC,
                         BIC=BIC)
        return sumstats
    
    def _fit(self, hess=True, **opt_kws):
        """
        

        Parameters
        ----------
        hess : bool, optional
            Whether or not to use the analytic hessian. The default is True.

        Returns
        -------
        None.

        """
        hess = self.hessian if hess else None
        
        self.opt = sp.optimize.minimize(self.loglike, self.theta, jac=self.gradient,
                                        hess=hess, method='trust-constr', **opt_kws)
        self.theta = self.opt.x
        self.L, self.Psi = self.model_matrices(self.theta)
        
        if self._rotation_method is not None:
            self.L, self.T, _, _, _, vgq, gamma = rotate(self.L, self._rotation_method)
            self.theta[self.lix] = vec(self.L)
        else:
            self.T = np.eye(self.n_facs)
            vgq, gamma = None, None
            
        self.L, self.T, self.theta = self._reorder_factors(self.L, self.T, self.theta)
        self.Phi = np.dot(self.T, self.T.T)
        self._make_augmented_params(self.L, self.Phi, self.Psi)
        self._vgq, self._gamma = vgq, gamma
        self.Sigma = self.implied_cov(self.theta)
        self.H = so_gc_cd(self.gradient_augmented, self.params)
        if self._rotation_method is not None:
            self.J = oblique_constraint_derivs(self.params, self)
            i, j = np.indices((self.n_facs, self.n_facs))
            i, j = i.flatten(), j.flatten()
            #self.J = self.J[j>=i]
            self.J = self.J[i!=j]
        else:
            self.J = self.constraint_derivs(self.theta)
        q = self.J.shape[0]
        self.Hc = np.block([[self.H, self.J.T], [self.J, np.zeros((q, q))]])
        self.se_params = np.sqrt(1.0 / np.diag(np.linalg.inv(self.Hc))[:-q]/self.n_obs)
        self.L_se = invec(self.se_params[self.lix], self.n_vars, self.n_facs)
        
    def fit(self, compute_factors=True, factor_method='regression', hess=True,
            **opt_kws):
        """
        

        Parameters
        ----------
        compute_factors : bool, optional
            Whether or not to compute factors. The default is True.
        factor_method : str, optional
            Method to compute factors. The default is 'regression'.
        hess : bool, optional
             Whether or not to use the analytic hessian. The default is True.
        **opt_kws : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self._fit(hess, **opt_kws)
        self.sumstats = self._fit_indices(self.Sigma)
        z = self.params / self.se_params
        p = sp.stats.norm(0, 1).sf(np.abs(z)) * 2.0
        
        param_labels = []
        for j in range(self.n_facs):
            for i in range(self.n_vars):
                param_labels.append(f"L[{i}][{j}]")
        if self._rotation_method is not None:
            for i in range(self.n_facs):
                for j in range(i):
                    param_labels.append(f"Phi[{i}][{j}]")
        for i in range(self.n_vars):
            param_labels.append(f"Psi[{i}]")
        res_cols = ["param", "SE", "z", "p"]
        fcols = [f"Factor{i}" for i in range(self.n_facs)]
        self.res = pd.DataFrame(np.vstack((self.params, self.se_params, z, p)).T,
                                columns=res_cols, index=param_labels)
        self.L = pd.DataFrame(self.L, index=self.cols, columns=fcols)
        
        if compute_factors:
            factor_coefs, factors = self.compute_factors(factor_method)
        self.factor_coefs = pd.DataFrame(factor_coefs, index=self.cols, columns=fcols)
        self.factors = pd.DataFrame(factors, index=self.inds, columns=fcols)
        
    def compute_factors(self, method="regression"):
        """
        

        Parameters
        ----------
        method : str, optional
            Method to compute factors. The default is "regression".

        Returns
        -------
        factor_coefs : ndarray
            Array to compute factors.
        factors : ndarray
            Array of factors.

        """
        X = self.X - np.mean(self.X, axis=0)
        if method=='regression':
            factor_coefs = np.linalg.inv(self.S).dot(self.L)
            factors = X.dot(factor_coefs)
        elif method=='bartlett':
            A = self.L.T.dot(np.diag(1/np.diag(self.Psi)))
            factor_coefs = (np.linalg.inv(A.dot(self.L.values)).dot(A)).T
            factors =  X.dot(factor_coefs)
        return factor_coefs, factors



                
            
            