# pystats
This project is recreational and consists of unfinished and unpolished programs to fit statistical models. Currently includes rough implementations of 
- Exploratory Factor Analysis
	- The FactorAnalysis class in factor_analysis.py under pyfa supports exploratory factor analysis with both orthogonal and oblique rotation.  Standard errors based on the augmented information matrix are available for the unrotated and oblique case
- Generalized Additive Models.  Supported smooths include cubic regression splines, cyclic cubic splines, and b-splines
	- The GAM class in gam.py under pygam supports Gaussian, Inverse Gaussian, and Gamma distributions
	- The GauLS class in gauls.py under pygam supports gaussian additive models for location and scale 
- Generalized Linear Models.  
 	- The GLM class supports Gaussian, Binomial, Gamma, Gaussian, Inverse Gaussian, Poisson and Negative Binomial distributionsm and Cloglog, Logit, Log, Log Complement, Probit, Negative Binomial and Reciprocal links.
	- Cumulative link models for ordinal regression. 
	- Negative Binomial Models
 	- Currently only supports NB2, although plans exist to implement NB1 
	- Zero Inflated Poisson models
	- Beta Regression 
	- An OLS class conducive to resampling based tests, currently implementing the standard nonparametric bootstrap, Freedman-Lane permutation testing, and maxT permutation testing
- Elastic net penalized generalized linear models
	- Currently only binomial and gaussian models are supported.  So far, provides similar results to glmnet, and at a similar speed, although much functionality needs to be implemented (e.g. intercepts, variable penalties, convergence conditions etc, etc)
- ICA using ML
- Latent variable correlations for handling polychorric, polytomous and tetrachoric correlation
- Linear mixed models 
	- LMM permits flexible random effect covariance and uses a cholesky parameterization.  Analytic gradient and hessian computation is available, but can become prohibitively slow for large models with cross random effects
	 - GLMM can fit (so far binomial and poisson) mixed models using PQL.  GLMM inherits LMM methods to repeatedly fit a weighted mixed model.
	 - GLMM_AGQ can be used to fit models with a single random intercept using numerical integration.
	 - MixedMCMC can fit normal, binomial, poisson, and ordered probit mixed models using MCMC.  
	 	- Normal models are fit through Gibbs sampling
	 	- Binary binomial (Bernoulli) models are fit through slice within gibbs sampling
	 	- Poisson, binomial, and ordered probit models can be fit using Metropolis Hastings within Gibbs. 
	 - MLMM can be used to fit multivariate outcomes (i.e. multiple dependent variables) 
- Robust linear regression with Hubers T, Tukeys Bisquare (Biweight), and Hampels function.
- Nonnegative matrix factorization using the seqNMF algorithm/approach
- Structural equation modeling
 	- Both exploratory factor analysis (via the FactorAnalysis class) and confirmatory factor analysis (via the SEM class)
 	- More general Structural equation modeling via the SEM class (e.g. path models to full SEM models)
 	- Factor rotation using Jennrich/Bernaard's gradient projection algorithms
- Sparse Canonical Correlation using the penalized matrix decomposition
- Nonparametric independence testing
- Random correlation matrix generation via the vine method, onion method, or factor method
# Requirements
All models have been written in python 3.7 using
- numpy 1.17.2
- numba 0.45.1
- scipy 1.5.3
- tqdm 4.36.1
- skpsarse 0.4.4
- matplotlib 3.3.
- patsy 0.5.1
- jax 0.1.72
- pandas 1.2.1

scikit-sparse is not supported on windows, but the workaround provided by https://github.com/EmJay276/scikit-sparse should allow for installation.


