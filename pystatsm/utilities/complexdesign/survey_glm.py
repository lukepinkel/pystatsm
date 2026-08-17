import numpy as np
import pandas as pd
from tqdm import tqdm
from ...pyglm2.glm import GLM
from ...pyglm2.families import Gaussian
from ..output import get_param_table
from .repweights import replicate_variance


class SurveyGLM:

    def __init__(self, formula, design, family=Gaussian, **model_kws):
        self.design = design
        self.formula = formula
        self.w_fit = design.w / np.mean(design.w)
        self.model = GLM(formula=formula, data=design.df, family=family,
                         weights=self.w_fit, **model_kws)
        if self.model.n != design.n:
            raise ValueError("the model dropped rows (missing data?); build "
                             "the design on complete cases so scores align "
                             "with the stratum/PSU layout")
        if self.model.scale_estimator == "NR":
            raise ValueError("scale_estimator='NR' places the scale in "
                             "params; use 'M' (Pearson dispersion, as in "
                             "svyglm)")
        self.f = self.model.f
        self.param_labels = list(self.model.param_labels)
        self.n_params = len(self.param_labels)
        self.keep = self.w_fit > 0
        self.X_k = self.model.X[self.keep]
        self.y_k = self.model.y[self.keep]
        self.w_k = self.w_fit[self.keep]
        self.psu_k = design.row_psu[self.keep]

    def _optimize(self, w, t_init=None, opt_kws=None):
        # w lives on the kept-row grid; rows a replicate zeroes out are
        # masked again so every family criterion stays finite.
        pos = w > 0
        data = (self.X_k[pos], self.y_k[pos], w[pos])
        return self.model._optimize(t_init=t_init, opt_kws=opt_kws, data=data)

    def _fit_replicates(self, mult, opt_kws=None, progress=True):
        n_rep = mult.shape[0]
        params_rep = np.zeros((n_rep, self.n_params))
        iterator = range(n_rep)
        if progress:
            iterator = tqdm(iterator, smoothing=1e-3)
        for r in iterator:
            opt = self._optimize(self.w_k * mult[r, self.psu_k],
                                 t_init=self.params, opt_kws=opt_kws)
            params_rep[r] = opt.x
        return params_rep

    def fit(self, vcov="linearized", n_rep=500, rng=None, center=None,
            opt_kws=None, rep_opt_kws=None, progress=True):
        self.opt = self._optimize(self.w_k, opt_kws=opt_kws)
        self.params = np.asarray(self.opt.x)
        data_k = (self.X_k, self.y_k, self.w_k)
        self.params_hess = self.model.hessian(self.params, data=data_k)
        self.bread = np.linalg.inv(self.params_hess)
        self.score_i = np.zeros((self.design.n, self.n_params))
        self.score_i[self.keep] = self.model.gradient_i(self.params,
                                                        data=data_k)
        M = self.design.meat(self.score_i)
        self.vcov_linearized = np.dot(self.bread, np.dot(M, self.bread))
        if vcov == "linearized":
            V = self.vcov_linearized
        else:
            mult, rscales, scale = self._replicates(vcov, n_rep, rng)
            self.params_rep = self._fit_replicates(mult, opt_kws=rep_opt_kws,
                                                   progress=progress)
            c = self.params if center == "estimate" else center
            V = replicate_variance(self.params_rep, rscales, scale, center=c)
            setattr(self, f"vcov_{vcov}", V)
        self.vcov_method = vcov
        self.params_cov = V
        self.params_se = np.sqrt(np.diag(V))
        self.degf_design = self.design.degf()
        self.ddf = self.degf_design - self.n_params + 1
        self.res = get_param_table(self.params, self.params_se,
                                   degfree=self.ddf, index=self.param_labels,
                                   parameter_label=list(self.model.ycols)[0])
        self.sumstats = pd.DataFrame(
            [self.design.n, self.design.n_grp, self.design.n_str,
             self.degf_design, self.ddf],
            index=["n_obs", "n_psu", "n_strata", "degf_design", "df_resid"],
            columns=["value"])
        return self

    def _replicates(self, vcov, n_rep, rng):
        if vcov == "jackknife":
            return self.design.jackknife_replicates()
        if vcov == "bootstrap":
            return self.design.bootstrap_replicates(n_rep=n_rep, rng=rng)
        raise ValueError(f"unknown vcov: {vcov!r}")
