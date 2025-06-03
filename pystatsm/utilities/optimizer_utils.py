#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 21:42:34 2020

@author: lukepinkel
"""
import numpy as np
import scipy as sp
#from scipy.optimize.optimize import MemoizeJac

LBFGSB_options = dict(disp=10, maxfun=5000, maxiter=5000, gtol=1e-8)
SLSQP_options = dict(disp=True, maxiter=1000)
TrustConstr_options = dict(verbose=0, gtol=1e-8)
TrustNewton_options = dict(gtol=1e-5)
default_opts = {'L-BFGS-B':LBFGSB_options,
                'l-bfgs-b':LBFGSB_options,
                'lbfgsb':LBFGSB_options,
                'LBFGSB':LBFGSB_options,
                'SLSQP':SLSQP_options,
                'slsqp':SLSQP_options,
                'trust-constr':TrustConstr_options,
                'trust-ncg':TrustNewton_options}

def process_optimizer_kwargs(optimizer_kwargs, default_method='trust-constr'):
    keys = optimizer_kwargs.keys()

    if 'method' not in keys:
        optimizer_kwargs['method'] = default_method

    if 'options' not in keys:
        optimizer_kwargs['options'] = default_opts[optimizer_kwargs['method']]
    else:
        options_keys = optimizer_kwargs['options'].keys()
        for dfkey, dfval in default_opts[optimizer_kwargs['method']].items():
            if dfkey not in options_keys:
                optimizer_kwargs['options'][dfkey] = dfval
    return optimizer_kwargs

class MemoizeJac:
    def __init__(self, fun):
        self.fun = fun
        self.jac = None
        self._value = None
        self.x = None

    def _compute_if_needed(self, x, *args):
        if not np.all(x == self.x) or self._value is None or self.jac is None:
            self.x = np.asarray(x).copy()
            fg = self.fun(x, *args)
            self.jac = fg[1]
            self._value = fg[0]

    def __call__(self, x, *args):
        self._compute_if_needed(x, *args)
        return self._value

    def derivative(self, x, *args):
        self._compute_if_needed(x, *args)
        return self.jac



class MemoizeGradHess(MemoizeJac):
    """ Decorator that caches the return vales of a function returning
       (fun, grad, hess) each time it is called.
        https://stackoverflow.com/a/68608349
    """
    def __init__(self, fun):
        super().__init__(fun)
        self.hess = None

    def _compute_if_needed(self, x, *args):
        if not np.all(x == self.x) or self._value is None or self.jac is None or self.hess is None:
            self.x = np.asarray(x).copy()
            self._value, self.jac, self.hess = self.fun(x, *args)

    def hessian(self, x, *args):
        self._compute_if_needed(x, *args)
        return self.hess


class ZeroConstraint(object):

    def __init__(self, n_params, zero_indices):
        self.n = self.n_params = n_params
        self.m = self.n_zeroed = len(zero_indices)
        self.zero_indices = zero_indices
        jac = np.zeros((self.m, self.n))
        jac[np.arange(self.m), zero_indices] = 1.0
        self.jac = jac

    def func(self, params):
        params_zeroed = params[self.zero_indices]
        return params_zeroed

    def grad(self, params):
        return self.jac

    def make_constraint(self):
        nlc = sp.optimize.NonlinearConstraint(self.func,
                                        jac=self.grad,
                                        lb=np.zeros(self.m),
                                        ub=np.zeros(self.m))
        return nlc
