import numpy as np
import pandas as pd
import patsy


class Population:
    """Base class for population simulators.

    Subclasses must override `theta_true` and `draw(rng)`. `parameter_names`
    is optional — return a list of names matching `theta_true` if you want
    nicely labeled diagnostic output from the replication harness.
    """

    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng

    @property
    def theta_true(self):
        raise NotImplementedError

    @property
    def parameter_names(self):
        return None

    def draw(self, rng=None):
        raise NotImplementedError


class _ClusteredPopulation(Population):
    """Shared X / strata / PSU / cluster-random-effect machinery.

    Subclasses override `draw` to map the linear predictor `eta` to a
    response. Total n = n_strata * n_psu_per_stratum * n_obs_per_psu (balanced).
    """

    def __init__(self, response, x_vars, rhs_formula, beta,
                 n_strata, n_psu_per_stratum, n_obs_per_psu,
                 x_cov=None, x_mean=None, intra_cluster_var=0.0,
                 weight_fn=None, rng=None):
        super().__init__(rng)
        self.response = response
        self.x_vars = list(x_vars)
        self.rhs_formula = rhs_formula
        self.formula = f"{response} ~ {rhs_formula}"
        self.beta = np.asarray(beta, dtype=np.float64).reshape(-1)
        self.n_strata = n_strata
        self.n_psu_per_stratum = n_psu_per_stratum
        self.n_obs_per_psu = n_obs_per_psu
        self.n_obs = n_strata * n_psu_per_stratum * n_obs_per_psu
        self.intra_cluster_var = float(intra_cluster_var)
        self.weight_fn = weight_fn
        nv = len(self.x_vars)
        self.x_cov = np.eye(nv) if x_cov is None else np.asarray(x_cov)
        self.x_mean = np.zeros(nv) if x_mean is None else np.asarray(x_mean)
        n_psu_total = n_strata * n_psu_per_stratum
        self._strata = np.repeat(np.arange(n_strata),
                                 n_psu_per_stratum * n_obs_per_psu)
        self._psu = np.repeat(np.arange(n_psu_total), n_obs_per_psu)
        dummy = pd.DataFrame(np.zeros((2, nv)), columns=self.x_vars)
        X = patsy.dmatrix(self.rhs_formula, data=dummy, return_type="dataframe")
        if X.shape[1] != self.beta.size:
            raise ValueError(
                f"beta has size {self.beta.size} but design matrix has "
                f"{X.shape[1]} columns: {list(X.columns)}"
            )
        self._param_names = list(X.columns)

    @property
    def theta_true(self):
        return self.beta

    @property
    def parameter_names(self):
        return self._param_names

    def _draw_eta(self, rng):
        n = self.n_obs
        n_psu = self.n_strata * self.n_psu_per_stratum
        X = rng.multivariate_normal(self.x_mean, self.x_cov, size=n)
        df = pd.DataFrame(X, columns=self.x_vars)
        df["strata"] = self._strata
        df["psu"] = self._psu
        if self.intra_cluster_var > 0:
            u = rng.normal(scale=np.sqrt(self.intra_cluster_var), size=n_psu)
            cluster_eff = u[df["psu"].to_numpy()]
        else:
            cluster_eff = 0.0
        Xd = patsy.dmatrix(self.rhs_formula, data=df, return_type="dataframe").values
        eta = np.dot(Xd, self.beta) + cluster_eff
        df["w"] = 1.0 if self.weight_fn is None else self.weight_fn(df)
        return df, eta


class LinearPopulation(_ClusteredPopulation):
    """Gaussian outcome: y = eta + eps, eps ~ N(0, resid_var)."""

    def __init__(self, response, x_vars, rhs_formula, beta,
                 n_strata, n_psu_per_stratum, n_obs_per_psu,
                 x_cov=None, x_mean=None,
                 intra_cluster_var=0.0, resid_var=1.0,
                 weight_fn=None, rng=None):
        super().__init__(response, x_vars, rhs_formula, beta,
                         n_strata, n_psu_per_stratum, n_obs_per_psu,
                         x_cov=x_cov, x_mean=x_mean,
                         intra_cluster_var=intra_cluster_var,
                         weight_fn=weight_fn, rng=rng)
        self.resid_var = float(resid_var)

    def draw(self, rng=None):
        rng = self.rng if rng is None else rng
        df, eta = self._draw_eta(rng)
        eps = rng.normal(scale=np.sqrt(self.resid_var), size=self.n_obs)
        df[self.response] = eta + eps
        return df


class BinomialPopulation(_ClusteredPopulation):
    """Bernoulli outcome: y ~ Bernoulli(sigmoid(eta))."""

    def draw(self, rng=None):
        rng = self.rng if rng is None else rng
        df, eta = self._draw_eta(rng)
        mu = 1.0 / (1.0 + np.exp(-eta))
        df[self.response] = rng.binomial(1, mu).astype(float)
        return df
