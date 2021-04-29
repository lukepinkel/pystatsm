# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 19:45:38 2021

@author: lukepinkel
"""
import pystan
import numpy as np
import pandas as pd
from .transforms import diagonalize_smooth
from .gam import GAM
from ..pyglm.families import Gamma, Gaussian, Binomial


stan_code1 = """
data {
      int N;
      int n_pena;
      int n_free;
      int n_vars;
      int n_smth;
      int ixl[n_pena];
      int ixp[n_pena];
      int ixf[n_free];
      vector[N] y;
      matrix[N, n_vars] X;
      matrix[n_vars, n_vars] D;
}

parameters {
    vector[n_vars] b;
    vector[n_smth] rho;
    real<lower=0> tau;
}
transformed parameters {
    vector[n_smth] lam;
    vector[n_vars] s;
    lam = exp(rho);
    for (i in 1:n_pena)
        s[ixp[i]] = lam[ixl[i]];
    for (i in 1:n_free)
        s[ixf[i]] = 10;
    
}
model {
    rho ~ uniform(-15, 15);
    b ~ normal(0, 1.0 ./ (s));
    y ~ """
stan_code2 = """\n}
generated quantities {
    vector[n_vars] beta;
    beta = D*b;
}
"""

def write_stancode(family):
    fams = {"Gaussian":"normal(X*b, tau);",
            "Gamma":"gamma(tau, tau ./ exp(X*b));"}
    return stan_code1+fams[family.name]+stan_code2

class GAM_MCMC(GAM):
    
    def __init__(self, formula, data, family):
        super().__init__(formula, data, family)       
        repara, rpmats, ixp, ixf, ixl = diagonalize_smooth(self.X.shape[1], 
                                                           self.smooths.copy())
        
        Xt = self.X.dot(rpmats['T'])
        self.n_pena = np.sum(self.ranks)
        self.n_free = self.nx - self.n_pena        
        data = dict(N=self.n_obs, n_pena=self.n_pena, n_free=self.n_free, 
                    n_vars=self.nx, n_smth=self.ns, y=self.y, 
                    ixp=ixp, ixf=ixf, ixl=ixl, X=Xt,
                    D=rpmats['T'])
        self.repara, self.rpmats = repara, rpmats
        self.ixp, self.ixf, self.ixl = ixp, ixf, ixl
        self.Xt = Xt
        self.data = data
        self.stan_code = write_stancode(self.f)
        self.stan_model = pystan.StanModel(model_code=self.stan_code)
        b_init = np.linalg.inv(self.Xt.T.dot(self.Xt)).dot(self.Xt.T.dot(self.y))
        self.init = dict(b=b_init, rho=self.theta[:-1], tau=np.exp(self.theta[-1]))
        
    
    def sample(self, iter=2000, chains=4, warmup=None):
        init = [self.init for i in range(chains)]
        self.trace = self.stan_model.sampling(data=self.data, init=init, iter=iter, 
                                            chains=chains, algorithm="NUTS", 
                                            warmup=warmup)
        res = self.trace.summary(pars=["beta", "rho", "tau"])
        self.res = pd.DataFrame(res['summary'], columns=res['summary_colnames'],
                                index=res['summary_rownames'])
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            