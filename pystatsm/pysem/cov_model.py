
import re
import numba
import pandas as pd
import numpy as np
import scipy as sp #analysis:ignore
from .cov_derivatives import _d2sigma, _dsigma, CovarianceDerivatives
from ..utilities.indexing_utils import (vec_inds_forwards,  #analysis:ignore
                                        vec_inds_reverse,
                                        vech_inds_forwards,
                                        vech_inds_reverse,
                                        tril_indices,
                                        unique, nonzero)
from ..utilities.linalg_operations import ( _vec, _invec, _vech, _invech, _vech_nb) #analysis:ignore
from ..utilities.special_mats import lmat, nmat, dmat #analysis:ignore
from ..utilities.func_utils import sizes_to_ind_arrs #analysis:ignore
from ..utilities.numerical_derivs import jac_approx, hess_approx #analysis:ignore



class CovarianceStructure(object):
    """
    A class for specifying a structural equation model (SEM) in matrix form.

    Attributes
    ----------
    matrix_names : list of str
        The names of the matrices in the model ("L", "B", "F", "P").

    is_symmetric : dict
        A dictionary indicating whether each matrix is symmetric.

    matrix_order : dict
        A dictionary mapping each matrix name to an index.

    {name}_free : np.ndarray
        The free parameters for the {name} matrix.

    {name}_fixed : np.ndarray
        The fixed parameters for the {name} matrix.

    {name}_fixed_loc : np.ndarray
        The locations of the fixed parameters for the {name} matrix.

    p1, q1 : int
        The dimensions of the L matrix.

    p2, q2 : int
        The number of parameters for each matrix.

    pq, qq : int
        The number of parameters for the L and B matrices.

    n_par : int
        The total number of parameters.

    Iq : np.ndarray
        An identity matrix of size q1.

    par_inds_by_mat : dict
        A dictionary mapping each matrix name to the indices of its parameters.

    par_mats : np.ndarray
        A 2D array storing the parameters and their corresponding matrix indices.
        the first column has integers 0,...,n_par-1 and the second an integer
        between 0 and 3

    mat_dims : dict
        A dictionary storing the dimensions of each matrix.
        {'L': (p1, q1), 'B': (q1, q1), 'F': (q1, q1), 'P': (p1, p1)}

    mat_dim_arr : np.ndarray
        An array storing the dimensions of each matrix - the first column consists of
        p1 q1 q1 p1 and the second q1 q1 q1 p1

    par_ind : np.ndarray
        An array of size n_par storing the indices of the parameters.  Each integer
        index specified by {name}_free matrices are in the position corresponding to
        the respective location when the matrices are vectorized and stacked

    par_to_free_ind : np.ndarray
        An array of size nf1 mapping the parameters to the free parameter.  The
        indices correspond to the positions in par_ind

    par_to_theta_ind : np.ndarray
        An array of size nt1 mapping the parameters to theta.  The
        indices correspond to the positions in par_ind

    free_ind : np.ndarray
        Free indices specified in {name}_free including repetition for equality
        constraints

    free_to_mat : np.ndarray
        An array mapping the free parameters to their corresponding matrices.

    free_to_theta_ind : np.ndarray
        An array mapping the free parameters to the theta parameters.

    nf1 : int
        The number of free parameters.

    theta_ind : np.ndarray
       The theta indices specified in {name}_free

    theta_to_free_ind : np.ndarray
        An array mapping the theta parameters to the free parameters. By repeating
        equal parameters
        par[self.par_to_free_ind]= theta[self.theta_to_free_ind]

    theta_to_mat : np.ndarray
        An array mapping the theta parameters to their corresponding matrices.

    nt1 : int
        The number of theta parameters.

    nf2, nt2 : int
        The triangular numbers associated with nf1 and nt1.

    free_hess_inds : np.ndarray
        The indices of the Hessian of the free parameters.

    """
    pairs = [(0, 0),  #L L
             (1, 0),  #B L
             (2, 0),  #F L
                      #P L is 0
             (1, 1),  #B B
             (2, 1)   #F B
                      #P B is 0
                      #F F is 0
                      #F P is 0
             ]
    matrix_names = ["L", "B", "F", "P"]
    matrix_order = dict(L=0, B=1, F=2, P=3)
    is_symmetric = {0:False, 1:False, 2:True, 3:True}
    def __init__(self, **kwargs):
        """
        Initialize the CovarianceStructure instance.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments representing the free and fixed matrices for
            the model.  Each matrix is expected to be provided as a 2D numpy
            array. The keys should be of the form "{name}_free" and
            "{name}_fixed", and "{name}_fixed_loc",  where {name} is one of
            the matrix names
            ("L", "B", "F", "P").
        """
        self._row_col_names = {}
        for name in self.matrix_names:
            self._set_matrix(name, 'free', kwargs)
            self._set_matrix(name, 'fixed', kwargs)
            self._set_matrix(name, 'fixed_loc', kwargs, dtype=bool)

        # Extract the dimensions of the L matrix
        self.p1, self.q1 = self.L_free.shape


        # Compute the number of parameters for each matrix
        self.p2 = (self.p1 + 1) * self.p1 // 2
        self.q2 = (self.q1 + 1) * self.q1 // 2
        self.n_par = self.p1 * self.q1 + self.q1 * self.q1 + self.q2 + self.p2
        self.pq = self.p1*self.q1
        self.qq = self.q1 * self.q1

        # Compute the total number of parameters
        self.n_par = self.pq + self.qq + self.q2 + self.p2

        # Create an identity matrix of size q1
        self.Iq = np.eye(self.q1)

        # Compute the indices of the parameters for each matrix
        self.par_inds_by_mat = sizes_to_ind_arrs([self.pq, self.qq, self.q2, self.p2], keys=[0, 1, 2, 3])

        # Initialize a matrix to store the parameters and their corresponding matrix indices
        # The first column is a range from 0 to n_par, representing the parameters
        # The second column is filled with the indices of the matrices ("L", "B", "F", "P") that each parameter belongs to
        # This allows us to map from the free parameters and theta to the matrix of the corresponding parameter
        self.par_mats = np.zeros((self.n_par, 2), dtype=int)
        self.par_mats[:, 0] = np.arange(self.n_par)
        for i, (name, idx) in enumerate(self.matrix_order.items()):
            self.par_mats[self.par_inds_by_mat[i], 1] = idx


        # Store the dimensions of each matrix
        self.mat_dims = {}
        self.mat_dim_arr = np.zeros((4, 2), dtype=int)
        for i, name in enumerate(self.matrix_names):
            mat = getattr(self, f"{name}_free")
            self.mat_dims[i] = mat.shape
            self.mat_dim_arr[i] = mat.shape

        # Initialize the free parameters, parameter template, and derivative matrices
        self.make_free_params()
        self.make_param_template()
        self.cov_der = CovarianceDerivatives(self.p1, self.q1, self.nf1, 
                                             self.nt1, self.par_to_free_ind,
                                             self.theta_to_free_ind, n_matrices=4)

    def make_free_params(self):
        """
        Create the mapping between the parameters, the free parameters,
        and the theta parameters.
        This method populates several instance variables that store these
        mappings, as well as the indices of the free parameters and
        theta parameters.
        """
        # Initialize a zero array to store the parameter indices
        par_ind = np.zeros(self.n_par)

        # Iterate over the matrix names and populate the parameter indices
        for i, name in enumerate(self.matrix_names):
            matrix = getattr(self, f"{name}_free")
            if self.is_symmetric[i]:
                par_ind[self.par_inds_by_mat[i]] = _vech(np.tril(matrix))
            else:
                par_ind[self.par_inds_by_mat[i]] = _vec(matrix)

        # Compute the indices of the free parameters and the number of free parameters
        par_to_free_ind = nonzero(par_ind >0).squeeze()
        free_ind = par_ind[par_ind > 0]
        nf = len(free_ind)

        # Compute the matrix indices of the free parameters
        free_to_mat = self.par_mats[par_to_free_ind][:, 1].T


        # Compute the unique indices of the theta parameters and the mapping between the theta parameters and the free parameters
        theta_ind, theta_to_free_ind, free_to_theta_ind = unique(free_ind)
        nt = len(theta_ind)
        par_to_theta_ind = par_to_free_ind[free_to_theta_ind]
        theta_to_mat = free_to_mat[free_to_theta_ind]

        # Store the computed values as instance variables

        self.par_ind = par_ind.astype(int)
        self.par_to_free_ind = par_to_free_ind
        self.par_to_theta_ind = par_to_theta_ind

        self.free_ind = free_ind.astype(int)
        self.free_to_mat = free_to_mat
        self.free_to_theta_ind = free_to_theta_ind
        self.nf1 = nf

        self.theta_ind = theta_ind.astype(int)
        self.theta_to_free_ind = theta_to_free_ind
        self.theta_to_mat = theta_to_mat
        self.nt1 = nt

        self.nf2 = self.nf1 * (self.nf1 + 1) // 2
        self.nt2 = self.nt1 * (self.nt1 + 1) // 2

        # Compute the indices of the Hessian of the free parameters
    def _check_input(self, name, a):
        if isinstance(a, pd.DataFrame):
            arr, index, columns = a.values, a.index, a.columns
        else:
            index, columns = self.generate_default_names(name, a)
            arr = a
        return arr, index, columns

    def _set_matrix(self, name, postfix, kwargs, dtype=None):
        """
        Helper function to set the free, fixed and fixed_loc matrices.
        If 'free' matrix is not set before trying to set 'fixed' or 'fixed_loc',
        this will raise an error.
        """
        # Check if the matrix is provided, else set to zero matrix
        matrix = kwargs.get(f"{name}_{postfix}")
        if matrix is not None:
            arr, index, columns = self._check_input(f"{name}", matrix)
            setattr(self, f"{name}_{postfix}", arr)
            if postfix == "free":
                self._row_col_names[f"{name}"] = index, columns
        else:
            try:
                base_matrix = getattr(self, f"{name}_free")
                arr = np.zeros_like(base_matrix, dtype=dtype)
                index, columns = self.generate_default_names(name, arr)
                setattr(self, f"{name}_{postfix}", arr)
                self._row_col_names[f"{name}_{postfix}"] = index, columns
            except AttributeError:
                raise AttributeError(f"Trying to set '{name}_{postfix}' before '{name}_free' is set. "
                                     f"Please ensure '{name}_free' is set before  '{name}_{postfix}'")

    def generate_default_names(self, name, a):
        if name in ["L", "P"]:
            index = [f"x{i}" for i in range(1, a.shape[0]+1)]
        elif name in ["F", "B"]:
            index = [f"z{i}" for i in range(1, a.shape[0]+1)]

        if name in ["L", "B", "F"]:
            columns = [f"z{i}" for i in range(1, a.shape[1]+1)]
        else:
            columns = [f"x{i}" for i in range(1, a.shape[0]+1)]
        return index, columns


    def make_param_template(self):
        """
        Create a template for the parameters of the model.

        This method populates several instance variables that store the indices of the free and fixed parameters,
        as well as the parameter template and the theta parameters.
        """
        # Initialize dictionaries to store the indices of the free and fixed parameters for each matrix
        self.mat_inds = {}
        self.mat_fixed_inds = {}
        self.diag_inds = np.zeros(self.n_par, dtype=bool)

        # Initialize an array to store the parameter template
        self.p_template = np.zeros(self.n_par)


        # Initialize a list to store the combined indices of the free parameters for each matrix
        self.mat_inds_comb = []

        # Iterate over the matrix names and populate the parameter template and indices
        for i, name in enumerate(self.matrix_names):
            # Get the free, fixed, and fixed location matrices for the current matrix
            matrix_free = getattr(self, f"{name}_free")
            matrix_fixed = getattr(self, f"{name}_fixed")
            matrix_fixed_loc = getattr(self, f"{name}_fixed_loc")


            # Compute the indices of the free and fixed parameters
            if self.is_symmetric[i]:
                inds = nonzero(np.tril(matrix_free), True)
            else:
                inds = nonzero(matrix_free, True)
            self.mat_inds[i] = inds
            self.mat_inds_comb.append(nonzero(matrix_free))
            inds_fixed = nonzero(matrix_fixed_loc, True) #inds_fixed = nonzero(matrix_fixed, True)
            self.mat_fixed_inds[i] = inds_fixed

            # Create a template for the current matrix
            mat_template = np.zeros(self.mat_dims[i])
            mat_template[inds_fixed] = matrix_fixed[inds_fixed]
            mat_template[inds] = 0.01
            if name in ["F", "P"]:
                diagonal_mask = inds[0]==inds[1]
                diagonal_inds = inds[0][diagonal_mask], inds[1][diagonal_mask]
                mat_template[diagonal_inds] = 1.0
                if name == "F":
                    id_mat = np.eye(self.q1)
                elif name == "P":
                    id_mat = np.eye(self.p1)
                self.diag_inds[self.par_inds_by_mat[i]] = _vech(id_mat)
            setattr(self, name, mat_template)

            # Add the template to the parameter template
            if self.is_symmetric[i]:
                self.p_template[self.par_inds_by_mat[i]] = _vech(mat_template)
            else:
                self.p_template[self.par_inds_by_mat[i]] = _vec(mat_template)

        # Extract theta
        self.theta = self.p_template[self.par_to_theta_ind]


        # Combine the indices of the free parameters for all matrices
        self.mat_inds_comb = np.concatenate(self.mat_inds_comb, axis=1)

    def _constraint_func(self, theta):
        Sigma = self.implied_cov(theta)
        s, d = np.linalg.slogdet(Sigma)
        return np.array([s*np.exp(d)/(1+np.exp(d))])

    def make_bounds(self):
        lb, ub = np.repeat(None, self.nt1), np.repeat(None, self.nt1)
        lb[self.diag_inds[self.par_to_theta_ind]] = 0
        bounds = [(lb[i], ub[i]) for i in range(len(lb))]
        return bounds

    def make_constraints(self):
        constr = sp.optimize.NonlinearConstraint(self._constraint_func,
                                                 lb=np.zeros(1), ub=np.array([np.inf]))
        return constr




