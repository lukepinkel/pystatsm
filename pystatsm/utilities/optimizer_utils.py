#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 21:42:34 2020

@author: lukepinkel
"""
import numpy as np
from scipy.optimize.optimize import MemoizeJac

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
    
    
    