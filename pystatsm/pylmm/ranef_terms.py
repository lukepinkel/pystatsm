#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 19:13:13 2022

@author: lukepinkel
"""
import patsy
import numpy as np
import scipy as sp
import scipy.sparse as sps
from ..utilities.data_utils import dummy_encode, _dummy_encode
from ..utilities.linalg_operations import invech_chol, invech, vech
from ..utilities.special_mats import nmat, lmat, kmat


def get_d2_chol(p):
    """
    Compute the Hessian of the cholesky factor of a covariance matrix.

    Parameters
    ----------
    p : int
        Dimension of the covariance matrix.

    Returns
    -------
    H : ndarray
        Hessian of the cholesky factor, has shape (p*(p+1)//2, p, p).
    """
    Lp = lmat(p).A
    T = np.zeros((p, p))
    H = []
    Ip = np.eye(p)
    for j, i in list(zip(*np.triu_indices(p))):
        T[i, j] = 1
        Hij = (Lp.dot(np.kron(Ip, T+T.T)).dot(Lp.T))[np.newaxis]
        H.append(Hij)
        T[i, j] = 0
    H = np.concatenate(H, axis=0)
    return H

class RandomEffectTerm(object):
    """
    Term for a random effect in a mixed model.

    Parameters
    ----------
    re_form : str
        Formula for the random effect.
    gr_form : str
        Grouping variable.
    data : DataFrame
        Data used for the design matrices.

    Attributes
    ----------
    G_deriv : list of csc_matrix
        Derivative of the G matrix with respect to the parameters, has length m.
    L_deriv : list of csc_matrix
        Derivative of the L matrix with respect to the parameters, has length m.
    re_form : str
        Formula for the random effect.
    gr_form : str
        Grouping variable.
    Xi : ndarray
        Design matrix for the random effect, has shape (n, p).
    j_rows : ndarray
        Array for the rows of the design matrix, has length n.
    j_cols : ndarray
        Array for the columns of the design matrix, has length n.
    z_rows : ndarray
        Array for the rows of the Z matrix, has length n*p.
    z_cols : ndarray
        Array for the columns of the Z matrix, has length n*p.
    g_rows : ndarray
        Array for the rows of the G matrix, has length p*q*p.
    g_cols : ndarray
        Array for the columns of the G matrix, has length p*q*p.
    l_rows : ndarray
        Array for the rows of the L matrix, has length m*q.
    l_cols : ndarray
        Array for the columns of the L matrix, has length m*q.
    n_group : int
        Number of groups.
    n_rvars : int
        Number of random variables.
    n_param : int
        Number of parameters.
    g_size : int
        Size of the G matrix.
    """
    def __init__(self, re_form, gr_form, data):
        # re_form: formula for the random effects
        # gr_form: formula for the grouping variable
        # data: dataframe containing the data
        
        # Create a design matrix for the random effect term using the provided formula 're_form' and the data.
        # The design matrix Xi is a 2D array (shape: n x p) that contains the values of the random effect variables for each observation.
        Xi = patsy.dmatrix(re_form, data=data, return_type='dataframe').values
        
        # Dummy encode the group variable using the provided formula 'gr_form' and the data.
        # The function _dummy_encode returns row indices, column indices and the number of groups (q).
        j_rows, j_cols, q =  _dummy_encode(data[gr_form])
        
        # The row indices of the group variable are sorted to maintain the original order of groups in the data.
        # This is crucial because the subsequent operations rely on this specific order (C, row-major, or lexicographical order) to correctly map the random effects to their respective groups.
        j_sort = np.argsort(j_rows)
        
        # Get the number of observations (n) and random effect variables (p) from the shape of the design matrix Xi.
        n, p = Xi.shape
        
        z_rows = np.repeat(np.arange(n), p)
        z_cols = np.repeat(j_cols[j_sort] * p, p) + np.tile(np.arange(p), n)
        
        g_cols = np.repeat(np.arange(p * q), p)
        g_rows = np.repeat(np.arange(q)*p, p * p) + np.tile(np.arange(p), p * q)
        ij = g_rows>= g_cols
        l_rows, l_cols = g_rows[ij], g_cols[ij]
        m = int(p * (p + 1) //2)
        d_theta = np.zeros(m)
        G_deriv = []
        L_deriv = []
        for i in range(m):
            d_theta[i] = 1
            dG_theta = invech(d_theta).reshape(-1, order='F')
            dGi = sps.csc_matrix((np.tile(dG_theta, q), (g_rows, g_cols)))
            dLi = sps.csc_matrix((np.tile(d_theta, q), (l_rows, l_cols)))
            G_deriv.append(dGi)
            L_deriv.append(dLi)
            d_theta[i] = 0
            
            
        self.G_deriv = G_deriv
        self.L_deriv = L_deriv
        self.re_form = re_form
        self.gr_form = gr_form
        self.Xi = Xi
        self.j_rows, self.j_cols = j_rows, j_cols
        self.z_rows, self.z_cols = z_rows, z_cols
        self.g_rows, self.g_cols = g_rows, g_cols
        self.l_rows, self.l_cols = l_rows, l_cols
        self.n_group = self.q = q
        self.n_rvars = self.p = p
        self.n_param = m
        self.g_size = p * q


class RandomEffects(object):
    """
    Random effects for a mixed model.

    Parameters
    ----------
    terms : list of RandomEffectTerm
        Terms for the random effects in the model.

    Attributes
    ----------
    Z : csc_matrix
        Design matrix for the random effects, has shape (n_rows, n_cols).
    G : csc_matrix
        G matrix for the random effects, has shape (n_cols, n_cols).
    L : csc_matrix
        L matrix for the random effects, has shape (n_cols, n_cols).
    terms : list of RandomEffectTerm
        Terms for the random effects in the model.
    theta : ndarray
        Parameters for the random effects, has length n_par.
    t_inds : list of ndarray
        Indices for the parameters, has length levels+1.
    l_inds : list of ndarray
        Indices for the L matrix, has length levels.
    g_inds : list of ndarray
        Indices for the G matrix, has length levels.
    jac_inds : list of ndarray
        Indices for the Jacobian, has length levels.
    g_data : ndarray
        Data for the G matrix, has length n_cols*n_cols.
    l_data : ndarray
        Data for the L matrix, has length n_cols*n_cols.
    group_sizes : list of int
        Sizes of the groups, has length levels.
    n_pars : list of int
        Number of parameters for each term, has length levels.
    n_rvars : list of int
        Number of random variables for each term, has length levels.
    n_par : int
        Total number of parameters.
    levels : int
        Number of levels in the model.
    H : ndarray
        Hessian matrix, has shape (n_par, n_par).
    elim_mats : dict of ndarray
        Elimination matrices for each term, has length levels.
    symm_mats : dict of ndarray
        Symmetry matrices for each term, has length levels.
    iden_mats : dict of ndarray
        Identity matrices for each term, has length levels.
    d2g_dchol : dict of ndarray
        Second derivative of the G matrix with respect to the cholesky factor for each term, has length levels.
    """

    
    def __init__(self, terms):
        z_offset, g_offset, l_offset, t_offset, cov_offset = 0, 0, 0, 0, 0
        z_data, z_rows, z_cols = [], [], []
        g_data, g_cols, g_rows = [], [], []
        l_data, l_cols, l_rows = [], [], []
        g_inds, l_inds, t_inds = [], [], []
        theta = []
        jac_inds = []

        for ranef in terms: 
            zi_rows, zi_cols = ranef.z_rows, ranef.z_cols + z_offset
            z_rows.append(zi_rows)
            z_cols.append(zi_cols)
            z_data.append(ranef.Xi.flatten())
            jac_inds.append(np.arange(z_offset, z_offset + ranef.p * ranef.q))
            Gi = np.eye(ranef.n_rvars)
            g_vech = vech(Gi)
            g_vec = Gi.reshape(-1, order='F')
            gi_rows, gi_cols = ranef.g_rows + cov_offset, ranef.g_cols + cov_offset
            g_inds.append(np.arange(g_offset, g_offset + ranef.p * ranef.p * ranef.q))

            g_rows.append(gi_rows)
            g_cols.append(gi_cols)
            g_data.append(np.tile(g_vec, ranef.n_group))
            
            li_rows, li_cols = ranef.l_rows + cov_offset, ranef.l_cols + cov_offset
            l_inds.append(np.arange(l_offset, l_offset + ranef.n_param * ranef.q))
            l_rows.append(li_rows)
            l_cols.append(li_cols)
            l_data.append(np.tile(g_vech, ranef.n_group))
            
            theta.append(g_vech)
            t_inds.append(np.arange(t_offset, t_offset+ranef.n_param))
            z_offset = z_offset + ranef.p * ranef.q
            t_offset = t_offset + ranef.n_param
            g_offset = g_offset + ranef.p * ranef.p * ranef.q
            l_offset = l_offset + ranef.n_param * ranef.q
            cov_offset = cov_offset + ranef.g_size
        theta.append(np.ones(1))
        t_inds.append(np.arange(t_offset, t_offset+1))
        theta = np.concatenate(theta)  
        n_rows = terms[0].Xi.shape[0]
        n_cols = z_offset
        z_rows, z_cols = np.concatenate(z_rows),  np.concatenate(z_cols)
        g_rows, g_cols = np.concatenate(g_rows),  np.concatenate(g_cols)
        l_rows, l_cols = np.concatenate(l_rows),  np.concatenate(l_cols)

        z_data = np.concatenate(z_data)
        g_data = np.concatenate(g_data)
        l_data = np.concatenate(l_data)
        
        self.z_rows, self.z_cols, self.z_data = z_rows, z_cols, z_data
        self.g_rows, self.g_cols = g_rows, g_cols
        self.l_rows, self.l_cols = l_rows, l_cols
        self.Z = sps.csc_matrix((z_data, (z_rows, z_cols)), shape=(n_rows, n_cols))
        self.G = sps.csc_matrix((g_data, (g_rows, g_cols)), shape=(n_cols, n_cols))
        self.L = sps.csc_matrix((l_data, (l_rows, l_cols)), shape=(n_cols, n_cols))
        self.terms = terms
        self.theta = theta
        self.t_inds = t_inds
        self.l_inds = l_inds
        self.g_inds = g_inds
        self.jac_inds = jac_inds
        self.g_data = g_data
        self.l_data = l_data
        self.group_sizes = [term.n_group for term in terms]
        self.n_pars = [term.n_param for term in terms]
        self.n_rvars = [term.n_rvars for term in terms]
        self.n_par = len(self.theta)
        self.levels = len(self.terms)
        self.H = np.zeros((self.n_par, self.n_par))
        self.elim_mats, self.symm_mats, self.iden_mats = {}, {}, {}
        self.d2g_dchol = {}
        for i in range(self.levels):
            p = self.n_rvars[i]
            self.elim_mats[i] = lmat(p).A
            self.symm_mats[i] = nmat(p).A
            self.iden_mats[i] = np.eye(p)
            self.d2g_dchol[i] = get_d2_chol(self.n_rvars[i])
        
            
    def get_u_indices(self): 
        """
        Get the indices for the random effects.

        Returns
        -------
        u_indices : dict of ndarray
            Indices for the random effects for each term.
        """
        u_indices = {}
        start=0
        for i in range(self.levels):
            q = self.group_sizes[i] * self.n_rvars[i]
            u_indices[i] = np.arange(start, start+q)
            start+=q
        return u_indices
    
    def wishart_info(self):
        """
        Get the Wishart information for the random effects.

        Returns
        -------
        ws : dict of dict
            Wishart information for each term.
        """
        ws = {}
        for i in range(self.levels):
            ws[i] = {}
            q = self.group_sizes[i]
            k = self.n_rvars[i]
            nu = q-(k+1)
            ws[i]['q'] = q
            ws[i]['k'] = k
            ws[i]['nu'] = nu
        return ws

            
                
