#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 07:59:44 2023

@author: lukepinkel
"""

import numpy as np
import scipy as sp
import pandas as pd

from ..utilities.linalg_operations import  _vech, _vec, _invec, _invech
from ..utilities.func_utils import handle_default_kws, triangular_number
from .derivatives import _dloglike_mu, _d2loglike_mu0, _d2loglike_mu1, _d2loglike_mu2 #from .cov_derivatives import _d2sigma, _dsigma, _dloglike, _d2loglike
from .param_table import ParameterTable
from .model_data import ModelData
from ..utilities import indexing_utils

def _sparse_post_mult(A, S):
    prod = S.T.dot(A.T)
    prod = prod.T
    return prod

pd.set_option("mode.chained_assignment", None)

class SimpleEqualityConstraint:
    def __init__(self, indicators):
        indicators = self.check_and_convert(indicators)
        self.unique_values, self.inverse_mapping, self.indices = indexing_utils.unique(indicators)

    @staticmethod
    def check_and_convert(indicators):
        if isinstance(indicators, (pd.DataFrame, pd.Series)):
            indicators = indicators.values
        elif isinstance(indicators, list):
            indicators = np.array(indicators)
        elif isinstance(indicators, dict):
            indicators = np.array(list(indicators.values()))
        if indicators.ndim > 1:
            indicators = indicators.flatten()
        return indicators
    
    def forward(self, x, check_arr=False):
        if type(x) is not np.array:
            x = np.asarray(x)
        return x.copy().flatten()[self.indices]
    
    def inverse(self, y, check_arr=False):
        if type(y) is not np.array:
            y= np.asarray(y)
        return y.copy().flatten()[self.inverse_mapping]
    
def equality_constraint_mat(unique_locs):
    n = unique_locs.max()+1
    m = len(unique_locs)
    row = np.arange(m)
    col = unique_locs
    data = np.ones(m)
    arr = sp.sparse.csc_matrix((data, (row, col)), shape=(m, n))
    return arr

class ModelSpecification(object):
    """
    A class that manages the setup of model specifications, particularly for handling equality constraints.
    
    Methods
    -------
    __init__(formula, data, group_col)
        Initialize the ModelSpecification object, assigns group IDs, and calls create_group_tables method.
    
    create_group_tables(formula, data, group_col)
        Creates group tables based on the formula and data provided, creates ParameterTable objects, and prepares model matrices.
    
    add_matrix_equalities(shared)
        Manages parameter equality constraints across matrices in the model.
    """

    
    def __init__(self, formula, data, group_col):
        """
        Initializes the ModelSpecification object, assigns group IDs based on unique values in group_col, 
        and calls create_group_tables method with the provided formula, data, and group column.

        Parameters
        ----------
        formula : str
            A string that specifies the model formula.
        data : pandas.DataFrame
            The data to be used in the model.
        group_col : str
            The name of the column in 'data' that contains the group identifiers.
        """
        group_ids = np.unique(data[group_col])
        n_groups = len(group_ids)
        data[group_col] = data[group_col].replace(dict(zip(group_ids, np.arange(n_groups)))) 
        self.n_groups = n_groups
        self.create_group_tables(formula, data, group_col)
        
    def create_group_tables(self, formula, data, group_col):
        """
        Processes the data based on the provided formula and group column, and prepares necessary structures for the model.
        Creates ParameterTable objects, and constructs model matrices.

        Parameters
        ----------
        formula : str
            A string that specifies the model formula.
        data : pandas.DataFrame
            The data to be used in the model.
        group_col : str
            The name of the column in 'data' that contains the group identifiers.
        """
        self.ptables, self.ftables , self.ptable_objects= {}, {}, {}
        self.p_templates, self.indexers = {}, {}
        self.sample_covs = {}
        self.sample_means = {}
        self.llconst = {}
        self.mat_cols, self.mat_rows, self.mat_dims = {}, {}, {}
        for i in range(self.n_groups):
            dfi = data.loc[data[group_col]==i].iloc[:, :-1]
            sample_cov = dfi.cov(ddof=0)
            sample_mean = pd.DataFrame(dfi.mean()).T
            ptable_object = ParameterTable(formula, sample_cov, sample_mean)
            pt, ix, mr, mc, md = ptable_object.construct_model_mats(ptable_object.ptable,
                                                        ptable_object.var_names, 
                                                        ptable_object.lav_order, 
                                                        ptable_object.obs_order)
            self.mat_rows[i] = mr
            self.mat_cols[i] = mc
            self.mat_dims[i] = md
            self.p_templates[i], self.indexers[i] = pt, ix
            self. ptable_objects[i] = ptable_object
            self.ptables[i] = ptable_object.ptable
            self.ftables[i] = ptable_object.ftable
            self.sample_covs[i] = sample_cov.values
            self.sample_means[i] = sample_mean.values
            self.llconst[i] = -np.linalg.slogdet(self.sample_covs[i])[1]-self.sample_covs[i].shape[0]
        self.ftable = pd.concat(self.ftables, axis=0).reset_index()
        self.frees = {}
        self.q = len((self.ptable_objects[0]).lav_order)
        self.p = len((self.ptable_objects[0]).obs_order)

        for i in range(self.n_groups):
            self.frees[i] = self.p_templates[i][self.indexers[i].flat_indices]
        ftable = pd.concat(self.ftables, axis=0).reset_index()
        label = ftable[["level_0", "lhs", "rel","rhs"]].astype(str).agg(' '.join, axis=1)
        ftable.loc[:, "label"] = ftable.loc[:, "label"].fillna(label)
        
        self.ftable = ftable
        
    def add_matrix_equalities(self, shared):
        """
        Manages parameter equality constraints across matrices in the model.
        Generates unique labels for each parameter, flags duplicate parameters
        across different groups, creates a transformation matrix, and a dictionary 
        to map from  the free parameters of each group to the full set of 
        free parameters.

        Parameters
        ----------
        shared : list of bool
            A list of booleans indicating whether each matrix in the model is
            shared across groups or not.
        """
        # This section of the code is focused on label generation for each parameter 
        # defined in the formula and tracking potential parameter duplications. 
        
        # It starts by copying the "mod" column into a new column called "label", 
        # which will be used to store the unique identifiers for each row (parameter). 
        # A "duplicate" column is also created to keep track of whether a parameter 
        # is duplicated across different groups.
        
        # The code then loops over all possible types of matrices (from 0 to 5), 
        # creating a unique label for each parameter. This is achieved by joining 
        # strings from various columns. If a matrix type is flagged as "shared", 
        # the label doesn't include the group identifier (level_0), allowing parameters 
        # to be shared across groups. Otherwise, the group identifier is included in the 
        # label to ensure it's unique across all groups.
        
        # After label creation, the code iterates over each unique group, and for each 
        # matrix type, if it's shared, it marks parameters as duplicates if they are not 
        # in the first group (i.e., if level_0 > 0). 
        
        # Finally, the function "indexing_utils.unique" is called on the "label" column 
        # to obtain the unique labels, their inverse mapping, and indices, which will 
        # be useful for further processing.
        self.shared = shared
        ftable = self.ftable
        ix = ftable["mod"].isnull()
        ftable["label"]= ftable["mod"].copy() 
        # ftable["duplicate"] = False
        for i in range(6):
            #Identify parameters (in matrix i) without labels already specified particular in the formula
            ix = ftable["mat"]==i
            not_null = ~ftable.loc[ix, "label"].isnull()
            ix1 = ix & not_null
            #Either add a label that will be unique across groups by adding group id or shared across groups
            if self.shared[i]:
                label = ftable[["lhs", "rel","rhs"]].astype(str).agg(' '.join, axis=1)
            else:
                label = ftable[["level_0", "lhs", "rel","rhs"]].astype(str).agg(' '.join, axis=1)
                ftable.loc[ix1, "label"] = ftable.loc[ix1, ["level_0", "label"]].astype(str).agg(' '.join, axis=1)
            ftable.loc[ix, "label"] = ftable.loc[ix, "label"].fillna(label)
        
        
        # for i in np.unique(ftable["level_0"]):
        #     group_ix = ftable["level_0"] == i
        #     for j in range(6):
        #         mat_ix = ftable["mat"]==j
        #         if self.shared[j]:
        #             ftable.loc[mat_ix & group_ix, "duplicate"] = i>0
            
        unique_values, inverse_mapping, indices = indexing_utils.unique(ftable["label"])
        unique_labels = pd.Series(unique_values)
        label_to_ind = pd.Series(unique_labels.index, index=unique_labels.values)
        
        self._unique_locs, self._first_locs = inverse_mapping, indices
        self.ftable = ftable
        self.ftable["theta_index"] = ftable["label"].map(label_to_ind)
        self.theta_to_free = equality_constraint_mat(self._unique_locs)
        self.free_to_theta = equality_constraint_mat(self._first_locs)
        self.free = self.ftable["start"].fillna(0)
        self.free_to_group_free = {}
        for i in range(self.n_groups):
            ix = self.ftable.loc[self.ftable["level_0"]==i, "theta_index"]
            cols = ix
            nrows = len(ix)
            ncols = self.ftable.shape[0]
            rows = np.arange(nrows)
            d = np.ones(nrows)
            self.free_to_group_free[i] = sp.sparse.csc_array((d, (rows, cols)), shape=(nrows, ncols))
        self.n_total_free = len(self.ftable)

        
class SEM:
    
    def __init__(self, formula, data, group_col=None, model_spec_kws=None, group_kws=None):
        default_model_spec_kws = dict(extension_kws=dict(fix_lv_var=False))
        model_spec_kws = handle_default_kws(model_spec_kws, default_model_spec_kws)
        group_kws = [False]*5 if group_kws is None else group_kws
        if group_col is None:
            group_col = "groups"
            data["groups"] = 0
        self.mspec = ModelSpecification(formula, data, group_col=group_col)
        self.mspec.add_matrix_equalities(**group_kws)
        self.indexer = self.mspec.indexers[0]

        self.model_data = ModelData(data=data)
        bounds = self.mspec.ftable[["lb", "ub"]]
        bounds = bounds.values
        self.bounds = [tuple(x) for x in bounds.tolist()]
        self.bounds_theta = [tuple(x) for x in bounds[self.mspec._first_locs]]
        self.p_templates = self.mspec.p_templates

    
        self.free = self.mspec.free
        self.p, self.q = self.mspec.p, self.mspec.q
        self.p2, self.q2 = triangular_number(self.p), triangular_number(self.q)
        self.nf = len(self.indexer.flat_indices)
        self.nf2 = triangular_number(self.nf)
        self.nt = len(self.indexer.first_locs)
        self.make_derivative_matrices()
        self.theta = self.mspec.free_to_theta.dot(self.free)
        self.n_par = len(self.p_templates[0])
        self.ll_const =  1.8378770664093453 * self.p
        self.n_groups =self. mspec.n_groups
        self.indexers = self.mspec.indexers
        self.gsizes = self.model_data.data_df["group"].value_counts().values
        self.gweights = self.gsizes / np.sum(self.gsizes)

    def make_derivative_matrices(self):
        self.indexer.create_derivative_arrays([(0, 0), (1, 0), (2, 0), (1 ,1), (2, 1), (5, 0), (5, 1)])
        self.dA = self.indexer.dA
        self.dSm = np.zeros((self.p2+self.p, self.nf))
        self.d2Sm = np.zeros((self.p2+self.p, self.nf, self.nf))
        self.m_size = self.indexer.block_sizes                    
        self.m_kind = self.indexer.block_indices                 
        self.d2_kind = self.indexer.block_pair_types             
        self.d2_inds = self.indexer.colex_descending_inds  
        self.J_theta = sp.sparse.csc_array(
            (np.ones(self.nf), (np.arange(self.nf), self.indexer.unique_locs)), 
             shape=(self.nf, self.nt))
        self.dSm = np.zeros((self.p2+self.p, self.nf))
        self.d2Sm = np.zeros((self.p2+self.p, self.nf, self.nf))
        s, r = np.triu_indices(self.p, k=0)
        self._vech_inds = r+s*self.p
        self.unique_locs = self.indexer.unique_locs
        self.ptable = self.mspec.ptables[0]
        self.free_table = self.mspec.ftables[0]
        self.free_names = self.mspec.ftable["label"]
        self.theta_names = self.mspec.ftable.iloc[self.mspec._first_locs]["label"]
        self._grad_kws = dict(dA=self.dA, m_size=self.m_size, m_type=self.m_kind, 
                              vind=self._vech_inds, n=self.nf, p2=self.p2)
        self._hess_kws = dict(dA=self.dA, m_size=self.m_size, d2_inds=self.d2_inds,
                              first_deriv_type=self.m_kind,  second_deriv_type=self.d2_kind,
                              n=self.nf,  vech_inds=self._vech_inds)

     
    
    def func(self, theta, per_group=False):
        free = self._theta_to_free(theta)
        if per_group:
            f = np.zeros(self.n_groups)
            if np.iscomplexobj(theta):
                f = f.astype(complex)
        else:
            f = 0.0
        for i in range(self.n_groups):
            group_free = self._free_to_group_free(free, i)
            par = self._free_to_par(group_free, i)
            mats = self._par_to_model_mats(par, i)
            Sigma, mu = self._implied_cov_mean(*mats)
            r = (self.mspec.sample_means[i]-mu).flatten()
            V = np.linalg.inv(Sigma)
            trSV = np.trace(V.dot(self.mspec.sample_covs[i]))
            rVr = np.dot(r.T.dot(V), r) 
            if np.any(np.iscomplex(Sigma)):
                s, lnd = np.linalg.slogdet(Sigma)
                lndS = np.log(s)+lnd
            else:
                s, lndS = np.linalg.slogdet(Sigma)
            fi = (rVr + lndS + trSV + self.mspec.llconst[i]) * self.gweights[i]
            if (s==-1) or (fi < -1):
                fi += np.inf
            if per_group:
                f[i] = fi
            else:
                f += fi
        return f
    
    def gradient(self, theta, per_group=False):
        free = self._theta_to_free(theta)
        if per_group:
            g = np.zeros((self.n_groups, self.mspec.n_total_free))
        else:
            g = np.zeros(self.mspec.n_total_free)
        for i in range(self.n_groups):
            group_free = self._free_to_group_free(free, i)
            par = self._free_to_par(group_free, i)
            L, B, F, P, a, b = self._par_to_model_mats(par, i)
            B = np.linalg.inv(np.eye(B.shape[0]) - B)
            LB = np.dot(L, B)
            mu = (a+LB.dot(b.T).T).reshape(-1)
            a, b = a.flatten(), b.flatten()
            Sigma = LB.dot(F).dot(LB.T) + P
            R = self.mspec.sample_covs[i] - Sigma
            Sinv = np.linalg.inv(Sigma)
            VRV = Sinv.dot(R).dot(Sinv)
            vecVRV = _vec(VRV)
            rtV = (self.mspec.sample_means[i].flatten() - mu).dot(Sinv)
            gi = np.zeros(self.nf)
            kws = self._grad_kws
            gi = _dloglike_mu(gi, L, B, F, b, a,vecVRV, rtV, **kws) * self.gweights[i]
            if per_group:
                g[i] = self.mspec.free_to_group_free[i].T.dot(gi)
            else:
                g = g + self.mspec.free_to_group_free[i].T.dot(gi)
        g = (self.mspec.free_to_theta.dot(g.T)).T
        return g
            

    def hessian(self, theta, per_group=False, method=0):
        free = self._theta_to_free(theta)
        if per_group:
            H = np.zeros((self.n_groups, self.mspec.n_total_free, self.mspec.n_total_free))
        else:
            H = np.zeros((self.mspec.n_total_free, self.mspec.n_total_free))
        for i in range(self.n_groups):
            group_free = self._free_to_group_free(free, i)
            par = self._free_to_par(group_free, i)
            L, B, F, P, a, b = self._par_to_model_mats(par, i)
            B = np.linalg.inv(np.eye(B.shape[0]) - B)   
            LB = np.dot(L, B)
            mu = (a+LB.dot(b.T).T).reshape(-1)
            d = (self.mspec.sample_means[i].flatten() - mu)
            a, b = a.flatten(), b.flatten()
            kws = self._hess_kws
            Sigma = LB.dot(F).dot(LB.T) + P
            S = self.mspec.sample_covs[i]
            R = S - Sigma
            Sinv = np.linalg.inv(Sigma)
            VRV = Sinv.dot(R).dot(Sinv)
            vecVRV = _vec(VRV)
            vecV = _vec(Sinv)
            Hi = np.zeros((self.nf,)*2)
            d1Sm = np.zeros((self.nf, self.p, self.p+1))
            if method == 0:
                Hi = _d2loglike_mu0(H=Hi, d1Sm=d1Sm, L=L, B=B, F=F, P=P, a=a, b=b, Sinv=Sinv, S=S, d=d, 
                               vecVRV=vecVRV, vecV=vecV, **kws)
            elif method == 1:
                Hi = _d2loglike_mu1(H=Hi, d1Sm=d1Sm, L=L, B=B, F=F, P=P, a=a, b=b, Sinv=Sinv, S=S, d=d, 
                               vecVRV=vecVRV, vecV=vecV, **kws)
            elif method == 2:
                Hi = _d2loglike_mu2(H=Hi, d1Sm=d1Sm, L=L, B=B, F=F, P=P, a=a, b=b, Sinv=Sinv, S=S, d=d, 
                               vecVRV=vecVRV, vecV=vecV, **kws)
            Hi = _sparse_post_mult(self.mspec.free_to_group_free[i].T.dot(Hi),
                                     self.mspec.free_to_group_free[i]) * self.gweights[i]
            if per_group:
                H[i] = Hi
            else:
                H = H + Hi
        J = self.mspec.free_to_theta
        H = _sparse_post_mult(J.dot(H), J.T)
        return H
    
    def _par_to_model_mats(self, par, i):
        slices = self.indexers[i].slices
        shapes = self.indexers[i].shapes
        L = _invec(par[slices[0]], *shapes[0])
        B = _invec(par[slices[1]], *shapes[1])
        F = _invech(par[slices[2]])
        P = _invech(par[slices[3]])
        a = _invec(par[slices[4]], *shapes[4])
        b = _invec(par[slices[5]], *shapes[5])
        return L, B, F, P, a, b

    def _free_to_par(self, free, i):
        par = self.p_templates[i].copy()
        if np.iscomplexobj(free):
            par = par.astype(complex)
        par[self.indexers[i].flat_indices] = free
        return par
    
    def _free_to_group_free(self, free, i):
        S = self.mspec.free_to_group_free[i]
        group_free = S.dot(free)
        return group_free

        
    def _theta_to_free(self, theta):
        free = self.mspec.theta_to_free.dot(theta)
        return free

    def _implied_cov_mean(self, L, B, F, P, a, b):
        B = np.linalg.inv(np.eye(B.shape[0])-B)
        LB = np.dot(L, B)
        Sigma = LB.dot(F).dot(LB.T) + P
        mu = (a+LB.dot(b.T).T).reshape(-1)
        return Sigma, mu
    
    def implied_cov_mean(self, theta):
        free = self._theta_to_free(theta)
        Sigmas = {}
        mus = {}
        for i in range(self.n_groups):
            group_free = self._free_to_group_free(free, i)
            par = self._free_to_par(group_free, i)
            mats = self._par_to_model_mats(par, i)
            Sigmas[i], mus[i] = self._implied_cov_mean(*mats)
        return Sigmas, mus
    
    def implied_sample_stats(self, free):
        L, B, F, P, a, b = self.free_to_model_mats(free)
        B = np.linalg.inv(np.eye(B.shape[0])-B)
        LB = np.dot(L, B)
        Sigma = LB.dot(F).dot(LB.T) + P
        mu = (a+LB.dot(b.T).T).reshape(-1)
        s, mu = _vech(Sigma), mu.flatten()
        sm = np.concatenate([s, mu])
        return sm
    
    def loglike(self, free, reduce=True):
        Sigma, mu = self.implied_cov_mean(free)
        L = sp.linalg.cholesky(Sigma) #np.linalg.cholesky(Sigma)
        Y = self.model_data.data - mu
        Z = sp.linalg.solve_triangular(L, Y.T, trans=1).T #np.dot(X, np.linalg.inv(L.T))
        t1 = 2.0 * np.log(np.diag(L)).sum() 
        t2 = np.sum(Z**2, axis=1)
        ll = (t1 + t2 + self.ll_const) / 2
        if reduce:
            ll = np.sum(ll)
        return ll
    
    def _fit(self, theta_init=None, minimize_kws=None, minimize_options=None, use_hess=False):
        bounds = self.bounds_theta
        func = self.func
        grad = self.gradient
        if use_hess:
            hess = self.hessian
        else:
            hess = None
        theta = self.theta if theta_init is None else theta_init
        default_minimize_options = dict(initial_tr_radius=1.0, verbose=3)
        minimize_options = handle_default_kws(minimize_options, default_minimize_options)
                
        default_minimize_kws = dict(method="trust-constr", options=minimize_options)
        minimize_kws = handle_default_kws(minimize_kws, default_minimize_kws)
        res = sp.optimize.minimize(func, x0=theta, jac=grad, hess=hess,  
                                   bounds=bounds,  **minimize_kws)
        return res
    
    def fit(self,  theta_init=None, minimize_kws=None, minimize_options=None, use_hess=False):
        res = self._fit(theta_init=theta_init, minimize_kws=minimize_kws, minimize_options=minimize_options, use_hess=use_hess)
        if np.linalg.norm(res.grad)>1e16:
            if minimize_options is None:
                minimize_options = {}
            minimize_options["initial_tr_radius"]=0.01
            res = self._fit(minimize_kws=minimize_kws, minimize_options=minimize_options, use_hess=use_hess)
        self.opt_res = res
        self.theta = res.x
        self.free = self._theta_to_free(self.theta)
        self.res = pd.DataFrame(res.x, index=self.theta_names, 
                                columns=["estimate"])
        self.res["se"] = np.sqrt(np.diag(np.linalg.inv(self.hessian(self.theta)*self.model_data.n_obs/2)))
        self.res_free = pd.DataFrame(self.mspec.theta_to_free.dot(self.res.values), 
                                     index=self.free_names, columns=self.res.columns)
        mats = {}
        free = self._theta_to_free(self.theta)
        mat_names = ["L", "B", "F", "P", "a", "b"]
        for i in range(self.n_groups):
            group_free = self._free_to_group_free(free, i)
            par = self._free_to_par(group_free, i)
            mlist = self._par_to_model_mats(par, i)
            mats[i] = {}
            for j,  mat in enumerate(mlist):
                mats[i][j] = pd.DataFrame(mat, index=self.mspec.mat_rows[i][j],
                                          columns=self.mspec.mat_cols[i][j])
        self.mats = mats
        
