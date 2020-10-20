#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 22:30:15 2020

@author: lukepinkel
"""


import numpy as np
import scipy as sp
import scipy.stats
import scipy.sparse as sps
from ..utilities.optimizer_utils import process_optimizer_kwargs
from sksparse.cholmod import cholesky# analysis:ignore

from ..utilities.linalg_operations import (_check_np,
                                                 _check_shape,
                                                 _check_shape_nb,
                                                 dummy,
                                                 invech,
                                                 scholesky,
                                                 sparse_cholesky,
                                                 sparse_woodbury_inversion )
from .model_matrices import (construct_model_matrices, 
                             create_gmats,
                             get_jacmats2,
                             lsq_estimate, 
                             make_theta,
                             update_gmat)
from ..utilities.output import get_param_table
from .families import (Binomial, ExponentialFamily, Gamma, Gaussian,  # analysis:ignore
                            InverseGaussian, Poisson, NegativeBinomial)

def lndet_gmat(theta, dims, indices):
    lnd = 0.0
    for key, value in dims.items():
        if key!='error':
            dims_i = dims[key]
            ng = dims_i['n_groups']
            Sigma_i = invech(theta[indices[key]])
            lnd += ng*np.linalg.slogdet(Sigma_i)[1]
    return lnd
    
def lndet_cmat(M):  
    L = sparse_cholesky(M)
    LA = L.A
    logdetC = np.sum(2*np.log(np.diag(LA))[:-1])
    return logdetC


class WLME:
    
    def __init__(self, formula, data, weights=None):
        if weights is None:
            weights = np.eye(len(data))
        self.weights = sps.csc_matrix(weights)
        self.weights_inv = sps.csc_matrix(np.linalg.inv(weights))
        
        X, Z, y, dims = construct_model_matrices(formula, data)
        dims['error'] = dict(n_groups=len(X), n_vars=1)

        theta, indices = make_theta(dims)
        XZ = sps.hstack([X, Z])
        C = XZ.T.dot(XZ)
        Xty = X.T.dot(y)
        Zty = Z.T.dot(y)
        b = np.vstack([Xty, Zty])
        Gmats, g_indices = create_gmats(theta, indices, dims)
        Gmats_inverse, _ = create_gmats(theta, indices, dims, inverse=True)
        G = sps.block_diag(list(Gmats.values())).tocsc()
        Ginv =  sps.block_diag(list(Gmats_inverse.values())).tocsc()
        Zs = sps.csc_matrix(Z)
        Ip = sps.eye(Zs.shape[0])
        self.bounds = [(None, None) if int(x)==0 else (0, None) for x in theta]
        self.G = G
        self.Ginv = Ginv
        self.g_indices = g_indices
        self.X = _check_shape_nb(_check_np(X), 2)
        self.Z = Z
        self.y = _check_shape_nb(_check_np(y), 2)
        self.XZ = XZ
        self.C = C
        self.Xty = Xty
        self.Zty = Zty
        self.b = b
        self.dims = dims
        self.indices = indices
        self.formula = formula
        self.data = data
        self.theta = lsq_estimate(dims, theta, indices, X, XZ, self.y)
        self.Zs = Zs
        self.Ip = Ip
        self.yty = y.T.dot(y)
        self.jac_mats = get_jacmats2(self.Zs, self.dims, self.indices, 
                                     self.g_indices, self.theta)
        self.t_indices = list(zip(*np.triu_indices(len(theta))))
    
    
    def _params_to_model(self, theta):
        G = update_gmat(theta, self.G.copy(), self.dims, self.indices, self.g_indices)
        Ginv = update_gmat(theta, self.G.copy(), self.dims, self.indices, self.g_indices, inverse=True)
        s = theta[-1]
        R = (self.weights.dot(self.Ip * s).dot(self.weights))
        Rinv =  (self.weights_inv.dot(self.Ip / s).dot(self.weights_inv))
        V = self.Zs.dot(G).dot(self.Zs.T) + R
        Vinv = sparse_woodbury_inversion(self.Zs, Cinv=Ginv, Ainv=Rinv.tocsc())
        W = Vinv.A.dot(self.X)
        return G, Ginv, R, Rinv, V, Vinv, W, s
    
    def _mme(self, theta):
        _, Ginv, _, Rinv, _, _, _, s = self._params_to_model(theta)
        F = self.XZ
        C = F.T.dot(Rinv).dot(F)
        k =  Ginv.shape[0]
        C[-k:, -k:] += Ginv
        yty = np.array(np.atleast_2d(self.y.T.dot(Rinv.A).dot(self.y)))
        b = self.XZ.T.dot(Rinv).dot(self.y)
        M = sps.bmat([[C, b],
                      [b.T, yty]])
        return M
    
    def loglike(self, theta, sparse_chol=False):
        G, Ginv, R, Rinv, V, Vinv, W, s = self._params_to_model(theta)
        F = self.XZ
        C = F.T.dot(Rinv).dot(F)
        k =  Ginv.shape[0]
        C[-k:, -k:] += Ginv
        logdetR = np.sum(np.log(R.diagonal()))
        logdetG = lndet_gmat(theta, self.dims, self.indices)
        yty = np.array(np.atleast_2d(self.y.T.dot(Rinv.A).dot(self.y)))
        b = self.XZ.T.dot(Rinv).dot(self.y)
        M = sps.bmat([[C, b],
                      [b.T, yty]])
        #L = sparse_cholesky(M)
        if sparse_chol:
            L = scholesky(M.tocsc(), ordering_method='natural').A
        else:
            #print(theta)
            #Off diagonals can become > diagonals in binomial models
            #Maybe switch to cholesky parameterization to avoid this problem
            #or add constraints st G_{ij}<minimum(G_{ii}, G_{jj})
            L = np.linalg.cholesky(M.A)
            
        ytPy = np.diag(L)[-1]**2
        logdetC = np.sum(2*np.log(np.diag(L))[:-1])
        ll = logdetC+logdetG+logdetR+ytPy
        return ll
    
    def gradient(self, theta):
        dims = self.dims
        G, Ginv, R, Rinv, V, Vinv, W, s = self._params_to_model(theta)
        XtW = W.T.dot(self.X)
        XtW_inv = np.linalg.inv(XtW)
        P = Vinv - np.linalg.multi_dot([W, XtW_inv, W.T])
        Py = P.dot(self.y)
        grad = []
        for key in dims.keys():
            for dVdi in self.jac_mats[key]:
                #gi = np.trace(dVdi.dot(P)) - Py.T.dot(dVdi.dot(Py))
                gi = np.einsum("ij,ji->", dVdi.A, P) - Py.T.dot(dVdi.dot(Py))
                grad.append(gi)
        grad = np.concatenate(grad)
        grad = _check_shape(np.array(grad))
        return grad
    
    def hessian(self, theta):
        dims = self.dims
        G, Ginv, R, Rinv, V, Vinv, W, s = self._params_to_model(theta)
        XtW = W.T.dot(self.X)
        XtW_inv = np.linalg.inv(XtW)
        P = Vinv - np.linalg.multi_dot([W, XtW_inv, W.T])
        Py = P.dot(self.y)
        H = []
        PJ, yPJ = [], []
        for key in dims.keys():
            J_list = self.jac_mats[key]
            for i in range(len(J_list)):
                Ji = J_list[i].T
                PJ.append((Ji.dot(P)).T)
                yPJ.append((Ji.dot(Py)).T)
        t_indices = self.t_indices
        for i, j in t_indices:
            PJi, PJj = PJ[i], PJ[j]
            yPJi, JjPy = yPJ[i], yPJ[j].T
            Hij = -(PJi.dot(PJj)).diagonal().sum()\
                        + (2 * (yPJi.dot(P)).dot(JjPy))[0]
            H.append(np.array(Hij[0]))
        H = invech(np.concatenate(H)[:, 0])
        return H
    
    def _optimize_theta(self, optimizer_kwargs={}, hess=None):
        optimizer_kwargs = process_optimizer_kwargs(optimizer_kwargs,
                                                    'trust-constr')
        optimizer = sp.optimize.minimize(self.loglike, self.theta, hess=hess,
                                         bounds=self.bounds, jac=self.gradient, 
                                         **optimizer_kwargs)
        theta = optimizer.x
        return optimizer, theta
    
    def _acov(self, theta=None):
        if theta is None:
            theta = self.theta
        H_theta = self.hessian(theta)
        Hinv_theta = np.linalg.inv(H_theta)
        SE_theta = np.sqrt(np.diag(Hinv_theta))
        return H_theta, Hinv_theta, SE_theta
    
    def _compute_effects(self, theta=None):
        G, Ginv, R, Rinv, V, Vinv, WX, s = self._params_to_model(theta)
        XtW = WX.T
        XtWX = XtW.dot(self.X)
        XtWX_inv = np.linalg.inv(XtWX)
        beta = _check_shape(XtWX_inv.dot(XtW.dot(self.y)))
        fixed_resids = _check_shape(self.y) - _check_shape(self.X.dot(beta))
        
        Zt = self.Zs.T
        u = G.dot(Zt.dot(Vinv)).dot(fixed_resids)
        
        return beta, XtWX_inv, u, G, R, Rinv, V, Vinv
    
    def _fit(self, optimizer_kwargs={}, hess=None):
        optimizer, theta = self._optimize_theta(optimizer_kwargs, hess)
        beta, XtWX_inv, u, G, R, Rinv, V, Vinv = self._compute_effects(theta)
        
        self.theta, self.beta, self.u = theta, beta, u
        self.params = np.concatenate([beta, theta])
        self.Hinv_beta = XtWX_inv
        self.se_beta = np.sqrt(np.diag(XtWX_inv))
        self.G = G
        self.R = R
        self.Rinv = Rinv
        self.V = V
        self.Vinv = Vinv

    def _postfit(self, theta=None):
        if theta is None:
            theta = self.theta
        _, _, self.se_theta = self._acov(theta)
        self.se_params = np.concatenate([self.se_beta, self.se_theta])

    def predict(self, X=None, Z=None):
        if X is None:
            X = self.X
        if Z is None:
            Z = self.Z
        return X.dot(self.beta)+Z.dot(self.u)

        
            
    
class GLMM(WLME):
    '''
    Currently an ineffecient implementation of a GLMM, mostly done 
    for fun.  A variety of implementations for GLMMs have been proposed in the
    literature, and a variety of names have been used to refer to each model;
    the implementation here is based of off linearization using a taylor
    approximation of the error (assumed to be gaussian) around the current
    estimates of fixed and random effects.  This type of approach may be 
    referred to as penalized quasi-likelihood, or pseudo-likelihood, and 
    may be abbreviated PQL, REPL, RPL, or RQL.

    '''
    def __init__(self, formula, data, weights=None, fam=None):
        if isinstance(fam, ExponentialFamily) is False:
            fam = fam()
        self.f = fam
        self.mod = WLME(formula, data, weights=None)        
        self.mod._fit()
        self.y = self.mod.y
        self.non_continuous = [isinstance(self.f, Binomial),
                               isinstance(self.f, NegativeBinomial),
                               isinstance(self.f, Poisson)]
        if np.any(self.non_continuous):
            self.mod.bounds = self.mod.bounds[:-1]+[(1, 1)]
        self._nfixed_params = self.mod.X.shape[1]
        self._n_obs = self.mod.X.shape[0]
        self._n_cov_params = len(self.mod.bounds)
        self._df1 = self._n_obs - self._nfixed_params
        self._df2 = self._n_obs - self._nfixed_params - self._n_cov_params - 1
        self._ll_const = self._df1 / 2 * np.log(2*np.pi)
        
    
    def _update_model(self, W, nu):
        nu = _check_shape(nu, 2)
        self.mod.weights = sps.csc_matrix(W)
        self.mod.weights_inv = sps.csc_matrix(np.diag(1.0/np.diag((W))))
        self.mod.y = nu
        self.mod.Xty = self.mod.X.T.dot(nu)
        self.mod.Zty = self.mod.Z.T.dot(nu)
        self.mod.theta = lsq_estimate(self.mod.dims, 
                                      self.mod.theta,
                                      self.mod.indices, 
                                      self.mod.X, 
                                      self.mod.XZ, 
                                      nu)
        self.mod.yty = nu.T.dot(nu)
        
        
    
    def _get_pseudovar(self):
        eta = self.mod.predict()
        mu = self.f.inv_link(eta)
        var_mu = _check_shape(self.f.var_func(mu=mu), 1)
        gp = self.f.dlink(mu)
        nu = eta + gp * (_check_shape(self.y, 1) - mu)
        W = np.diag(np.sqrt(var_mu * (self.f.dlink(mu)**2)))
        return W, nu
    
    def _sandwich_cov(self, r):
        M = self.mod.Hinv_beta
        X = self.mod.X
        Vinv = self.mod.Vinv
        B = (Vinv.dot(X))
        d = _check_shape(r, 2)**2
        B = B * d
        C = B.T.dot(B)
        Cov = M.dot(C).dot(M)
        return Cov

    
    def predict(self, fixed=True, random=True):
        yhat = 0.0
        if fixed:
            yhat += self.mod.X.dot(self.mod.beta)
        if random:
            yhat += self.mod.Z.dot(self.mod.u)
        return yhat        
        

    def fit(self, n_iters=200, tol=1e-3, optimizer_kwargs={}, 
            verbose_outer=True, hess=False):
        if 'options' in optimizer_kwargs.keys():
            if 'verbose' not in optimizer_kwargs['options'].keys():
                optimizer_kwargs['options']['verbose'] = 0
        else:
            optimizer_kwargs['options'] = dict(verbose=0)
        
        if hess:
            hessian = self.mod.hessian
        else:
            hessian = None
            
        theta = self.mod.theta.copy()
        fit_hist = {}
        for i in range(n_iters):
            W, nu = self._get_pseudovar()
            self._update_model(W, nu)
            self.mod._fit(optimizer_kwargs, hessian)
            tvar = (np.linalg.norm(theta)+np.linalg.norm(self.mod.theta))
            eps = np.linalg.norm(theta - self.mod.theta) / tvar
            fit_hist[i] = dict(param_change=eps, theta=self.mod.theta,
                               nu=nu)
            if verbose_outer:
                print(eps)
            if eps < tol:
                break
            theta = self.mod.theta.copy()
        self.mod._postfit()
        self.res = get_param_table(self.mod.params, self.mod.se_params, 
                                   self.mod.X.shape[0]-len(self.mod.params))
        
        
        eta_fe = self.predict(fixed=True, random=False)
        eta = self.predict(fixed=True, random=True)
        mu = self.f.inv_link(eta)
        gp = self.f.dlink(mu)
        var_mu  =  _check_shape(self.f.var_func(mu=mu), 1)
        r_eta_fe = _check_shape(self.mod.y, 1) - eta_fe

        generalized_chi2 = r_eta_fe.T.dot(self.mod.Vinv.dot(r_eta_fe))
        resids_raw_linear = _check_shape(self.mod.y, 1) - eta
        resids_raw_mean = _check_shape(self.y, 1) - mu
        
        var_pearson_linear = self.mod.R.diagonal() / gp**2
        var_pearson_mean = var_mu
        
        resids_pearson_linear = resids_raw_linear / np.sqrt(var_pearson_linear)
        resids_pearson_mean = resids_raw_mean / np.sqrt(var_pearson_mean)
        
        pll = self.mod.loglike(self.mod.theta) / -2.0 - self._ll_const
        aicc = -2 * pll + 2 * self._n_cov_params  * self._df1 / self._df2
        bic = -2 * pll + self._n_cov_params * np.log(self._df1)
        self.sumstats = dict(generalized_chi2=generalized_chi2,
                             pseudo_loglike=pll,
                             AICC=aicc,
                             BIC=bic)
        self.resids = dict(resids_raw_linear=resids_raw_linear,
                           resids_raw_mean=resids_raw_mean,
                           resids_pearson_linear=resids_pearson_linear,
                           resids_pearson_mean=resids_pearson_mean)
        
        
        
def gh_rules(n, wn=True):
    z, w =  sp.special.roots_hermitenorm(n)
    if wn:
        w = w / np.sum(w)
    f = sp.stats.norm(0, 1).logpdf(z)
    return z, w, f

def vech2vec(vh):
    A = invech(vh)
    v = A.reshape(-1, order='F')
    return v
    
def approx_hess(f, x, *args, eps=None):
    p = len(x)
    if eps is None:
        eps = (np.finfo(float).eps)**(1./3.)
    H = np.zeros((p, p))
    ei = np.zeros(p)
    ej = np.zeros(p)
    for i in range(p):
        for j in range(i+1):
            ei[i], ej[j] = eps, eps
            if i==j:
                dn = -f(x+2*ei)+16*f(x+ei)-30*f(x)+16*f(x-ei)-f(x-2*ei)
                nm = 12*eps**2
                H[i, j] = dn/nm  
            else:
                dn = f(x+ei+ej)-f(x+ei-ej)-f(x-ei+ej)+f(x-ei-ej)
                nm = 4*eps*eps
                H[i, j] = dn/nm  
                H[j, i] = dn/nm  
            ei[i], ej[j] = 0.0, 0.0
    return H
    
                
            
    
class GLMM_AGQ:
    def __init__(self, formula, data, family):
        if isinstance(family, ExponentialFamily)==False:
            family = family()
        self.f = family
        X, Z, y, dims = construct_model_matrices(formula, data)
        theta, indices = make_theta(dims)
        Gmats, g_indices = create_gmats(theta, indices, dims)
        Gmats_inverse, _ = create_gmats(theta, indices, dims, inverse=True)
        G = sps.block_diag(list(Gmats.values())).tocsc()
        Ginv =  sps.block_diag(list(Gmats_inverse.values())).tocsc()
        self.X = _check_shape_nb(_check_np(X), 2)
        self.y = _check_shape_nb(_check_np(y), 1)
        self.Z = Z
        self.Zs = sps.csc_matrix(Z)
        self.Zt = self.Zs.T
        group_var, = list(dims.keys())
        self.J = dummy(data[group_var])
        n_vars = dims[group_var]['n_vars']
        self.n_indices = data.groupby(group_var).indices
        self.Xg, self.Zg, self.yg = {}, {}, {}
        self.u_indices, self.c_indices = {}, {}
        k = 0
        for j, (i, ix) in enumerate(self.n_indices.items()):
            self.Xg[i] = self.X[ix]
            self.Zg[i] = self.Z[ix, j][:, None]
            self.yg[i] = self.y[ix]
            self.u_indices[i] = np.arange(k, k+n_vars)
            self.c_indices[i] = (np.arange(k, k+n_vars)[:, None].T, 
                                 np.arange(k, k+n_vars)[:, None])
            k+=n_vars
        self.n_groups = len(self.Xg)
        self.n, self.p = self.X.shape
        self.q = self.Z.shape[1]
        self.nt = len(theta)
        self.params = np.zeros(self.p+self.nt)
        self.params[-self.nt:] = theta
        self.bounds = [(None, None) for i in range(self.p)]+\
                 [(None, None) if int(x)==0 else (0, None) for x in theta]
        
        self.D = np.eye(n_vars)
        self.W = sps.csc_matrix((np.ones(self.n), (np.arange(self.n), 
                                                   np.arange(self.n))))
        
        self.G = G
        self.Ginv = Ginv
        self.g_indices = g_indices
        self.dims = dims
        self.indices = indices
    
    def pirls(self, params):
        beta, theta = params[:self.p], params[self.p:]
        Psi_inv = update_gmat(theta, self.G.copy(), self.dims, self.indices, 
                           self.g_indices, inverse=True)
        D = cholesky(Psi_inv).L()
        u = np.zeros(self.q)
        Xb = self.X.dot(beta)
        for i in range(100):
            eta = Xb + self.Z.dot(u)
            mu = self.f.inv_link(eta)
            var_mu = self.f.var_func(mu=mu)
            self.W.data = var_mu
            ZtWZ = self.Zt.dot(self.W.dot(self.Zs))
            Ztr = self.Zt.dot(_check_shape(self.y, 1) - mu)
            RtR, r = ZtWZ + Psi_inv, Ztr - Psi_inv.dot(u)
            u_new = u + sps.linalg.spsolve(RtR, r)
            diff = np.linalg.norm(u_new - u) / len(u)
            if diff<1e-6:
                break
            u = u_new
        eta = Xb + self.Zs.dot(u)
        mu = self.f.inv_link(eta)
        var_mu = self.f.var_func(mu=mu)
        self.W.data = var_mu
        ZtWZ = self.Zt.dot(self.W.dot(self.Zs))
        Ztr = self.Zt.dot(self.y - mu)
        RtR, r = ZtWZ + Psi_inv, Ztr - Psi_inv.dot(u)
        Q = cholesky(RtR.tocsc()).L()
        Qinv = sps.linalg.inv(Q)
        return dict(u=u, D=D, Xb=Xb, Qinv=Qinv, Q=Q)
    
    def _dloglike(self, db, Xb, Qinv, D, u):
        db = np.zeros(Qinv.shape[0]) + db
        u_tilde = Qinv.dot(db) + u
        eta = (self.Z.dot(u_tilde)) + Xb
        mu = self.f.inv_link(eta)
        T = self.f.canonical_parameter(mu)
        bT = self.f.cumulant(T)
        Du = D.dot(u_tilde)
        ll = (np.exp((self.y * T - bT).dot(self.J) - Du**2 / 2))
        return ll

    
    def loglike(self, params, nagq=20):
        pirls_dict = self.pirls(params)
        z, w, f = gh_rules(nagq, False)
        args = (pirls_dict['Xb'], pirls_dict['Qinv'], pirls_dict['D'], 
                pirls_dict['u'])
        sq2 = np.sqrt(2)
        ll_i = np.sum([self._dloglike(z[i], *args) * w[i] 
                     for i in range(len(w))], axis=0) * sq2
        ll_i = np.log(ll_i)
        lnd = np.linalg.slogdet(pirls_dict['D'].A)[1]\
              -np.linalg.slogdet(pirls_dict['Q'].A)[1]
        ll = -(np.sum(ll_i) + lnd)
        return ll
    
    def fit(self):
        self.optimizer = sp.optimize.minimize(self.loglike, self.params, 
                                   bounds=self.bounds, 
                                   options=dict(disp=1))
        self.params = self.optimizer.x
        self.hess_theta = approx_hess(self.loglike, self.optimizer.x)
        self.se_params = np.sqrt(np.diag(np.linalg.inv(self.hess_theta)))
        self.res = get_param_table(self.params, self.se_params, 
                                   self.X.shape[0]-len(self.params))


    
        
       
               