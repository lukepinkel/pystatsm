import numpy as np
import scipy as sp
import scipy.stats
import pandas as pd
import patsy
from dataclasses import dataclass
from .sample_design import design_sandwich


@dataclass
class FitResult:
    params: np.ndarray   # (p,)
    score_i: np.ndarray  # (n, p), already weighted by design.w
    bread: np.ndarray    # (p, p), inverse weighted Hessian


class WLSAdapter:
    """Weighted least squares fitter that conforms to the model contract."""

    def __init__(self, formula):
        self.formula = formula

    def fit(self, design):
        y, X = patsy.dmatrices(self.formula, data=design.df, return_type="dataframe")
        X = X.values
        y = y.values.reshape(-1)
        w = design.w
        WX = X * w[:, None]
        XtWX = np.dot(X.T, WX)
        XtWy = np.dot(WX.T, y)
        bread = np.linalg.inv(XtWX)
        beta = np.dot(bread, XtWy)
        score_i = WX * (y - np.dot(X, beta))[:, None]
        return FitResult(params=beta, score_i=score_i, bread=bread)


class GLMAdapter:
    """Adapter around pystatsm.pyglm2.glm.GLM.

    The pystatsm GLM minimizes the negative log-likelihood, so
    `model.params_cov = inv(observed Fisher information)` is the bread, and
    `model.gradient_i(params)` returns the per-observation score of -loglik
    (already weighted by the `weights` vector). Sign of the score doesn't
    affect the meat (it enters as an outer product) so we forward both
    quantities directly. Validated against R `survey::svyglm` in stratified.py.
    """

    def __init__(self, formula, family=None, opt_kws=None):
        from ...pyglm2.glm import GLM
        from ...pyglm2 import families
        self.GLM = GLM
        self.formula = formula
        self.family = family if family is not None else families.Gaussian
        self.opt_kws = opt_kws

    def fit(self, design):
        model = self.GLM(formula=self.formula, data=design.df,
                         family=self.family, weights=design.w)
        model.fit(opt_kws=self.opt_kws)
        return FitResult(
            params=np.asarray(model.params),
            score_i=model.gradient_i(model.params),
            bread=model.params_cov,
        )


@dataclass
class ReplicationResult:
    params: np.ndarray         # (n_rep, p)
    Vs: np.ndarray             # (n_rep, p, p)
    theta_true: np.ndarray     # (p,)
    parameter_names: list = None

    @property
    def n_rep(self):
        return self.params.shape[0]

    @property
    def p(self):
        return self.params.shape[1]

    def bias(self):
        return self.params.mean(axis=0) - self.theta_true

    def empirical_cov(self):
        return np.cov(self.params.T, ddof=1)

    def mean_analytical_cov(self):
        return self.Vs.mean(axis=0)

    def se_emp(self):
        return np.sqrt(np.diag(self.empirical_cov()))

    def se_ana(self):
        return np.sqrt(np.diag(self.mean_analytical_cov()))

    def se_ratio(self):
        return self.se_emp() / self.se_ana()

    def z_scores(self):
        d = self.params - self.theta_true
        ses = np.sqrt(np.diagonal(self.Vs, axis1=1, axis2=2))
        return d / ses

    def coverage(self, alpha=0.05):
        z_crit = sp.stats.norm.ppf(1.0 - alpha / 2.0)
        return np.mean(np.abs(self.z_scores()) < z_crit, axis=0)

    def summary(self, alpha=0.05):
        names = (self.parameter_names if self.parameter_names is not None
                 else [f"theta_{i}" for i in range(self.p)])
        return pd.DataFrame({
            "true": self.theta_true,
            "mean": self.params.mean(axis=0),
            "bias": self.bias(),
            "se_emp": self.se_emp(),
            "se_ana": self.se_ana(),
            "ratio": self.se_ratio(),
            "coverage": self.coverage(alpha=alpha),
        }, index=names)


def replicate(population, design_factory, model_adapter, n_rep,
              rng=None, progress=True):
    """Run a Monte Carlo recovery study.

    Each replicate draws a fresh sample from `population`, builds a
    `SampleDesign` via `design_factory(df)`, fits the model with
    `model_adapter`, and computes the design-corrected sandwich.
    """
    rng = np.random.default_rng() if rng is None else rng
    theta_true = np.asarray(population.theta_true)
    p = theta_true.size
    params = np.zeros((n_rep, p))
    Vs = np.zeros((n_rep, p, p))
    iterator = range(n_rep)
    if progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator)
        except ImportError:
            pass
    for r in iterator:
        df = population.draw(rng)
        design = design_factory(df)
        fit = model_adapter.fit(design)
        Vs[r] = design_sandwich(fit.bread, fit.score_i, design)
        params[r] = fit.params
    names = getattr(population, "parameter_names", None)
    return ReplicationResult(params=params, Vs=Vs, theta_true=theta_true,
                             parameter_names=names)
