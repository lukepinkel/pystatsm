#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multivariate (weighted) least squares with a log-Cholesky precision
parameterization.  Beyond estimation the model reports classic measures
of multivariate association and MANOVA-style tests of the general linear
hypothesis L B M = 0.

@author: lukepinkel
"""
import numpy as np
import scipy as sp
import scipy.linalg
import scipy.stats
import pandas as pd
from .regression_model import RegressionMixin
from .likelihood_model import LikelihoodModel
from ..utilities.linalg_operations import _vec

LN2PI = np.log(2 * np.pi)

MANOVA_STATS = ['Wilks lambda', 'Pillai trace',
                'Hotelling-Lawley trace', 'Roy greatest root']
MANOVA_COLUMNS = ['Value', 'Num DF', 'Den DF', 'F Value', 'P Value']


class MMRParameterPacker:
    """Pack (B, L) <-> theta = (vec_F(B), lambda).

    L is the lower Cholesky of the precision Omega = L L'.  lambda holds
    the strict-lower entries of L in row-major order with the diagonal
    log-transformed.
    """

    def __init__(self, n_terms, n_responses, term_names=None,
                 response_names=None):
        q = self.n_responses = int(n_responses)
        self.n_terms = int(n_terms)
        self.term_names = (list(term_names) if term_names is not None
                           else ['x{}'.format(i) for i in range(n_terms)])
        self.response_names = (list(response_names) if response_names is not None
                               else ['y{}'.format(i + 1) for i in range(q)])
        self.n_beta = self.n_terms * q
        self.n_cov = q * (q + 1) // 2
        self.n_params = self.n_beta + self.n_cov
        self.tril_rows, self.tril_cols = np.tril_indices(q)
        self.diag_mask = self.tril_rows == self.tril_cols

    def precision_chol_to_lambda(self, L):
        lamb = L[self.tril_rows, self.tril_cols].astype(float).copy()
        lamb[self.diag_mask] = np.log(lamb[self.diag_mask])
        return lamb

    def lambda_to_precision_chol(self, lamb):
        L = np.zeros((self.n_responses, self.n_responses))
        vals = np.asarray(lamb, dtype=float).copy()
        vals[self.diag_mask] = np.exp(vals[self.diag_mask])
        L[self.tril_rows, self.tril_cols] = vals
        return L

    def pack(self, beta, precision_chol):
        return np.concatenate([np.asarray(beta).ravel(order='F'),
                               self.precision_chol_to_lambda(precision_chol)])

    def unpack_beta(self, theta):
        return np.asarray(theta[:self.n_beta]).reshape(
            self.n_terms, self.n_responses, order='F')

    def unpack_precision_chol(self, theta):
        return self.lambda_to_precision_chol(theta[self.n_beta:])

    def unpack(self, theta):
        return self.unpack_beta(theta), self.unpack_precision_chol(theta)

    def beta_labels(self):
        return ['{}~{}'.format(r, t)
                for r in self.response_names for t in self.term_names]

    def cov_labels(self):
        names = []
        for r, c in zip(self.tril_rows, self.tril_cols):
            tag = 'log_chol_prec' if r == c else 'chol_prec'
            names.append('{}[{},{}]'.format(
                tag, self.response_names[r], self.response_names[c]))
        return names

    def theta_labels(self):
        return self.beta_labels() + self.cov_labels()


def _triangular_inv(L):
    Linv, _ = sp.linalg.lapack.dtrtri(L, lower=1)
    return Linv


def _precision_chol_from_sigma(Sigma):
    L_sig = np.linalg.cholesky(Sigma)
    Linv = _triangular_inv(L_sig)
    Omega = np.dot(Linv.T, Linv)
    return np.linalg.cholesky(Omega)


def _omega_first_stack(L, packer):
    """Build Omega_a = D_a L' + L D_a' for every lambda index a.

    Returned shape: (n_cov, q, q).  factor[a] is L[r,r] when a is a
    diagonal entry of L, else 1.
    """
    q = L.shape[0]
    rows, cols = packer.tril_rows, packer.tril_cols
    factor = np.where(packer.diag_mask, np.diag(L)[rows], 1.0)
    L_cols_T = L[:, cols].T
    Omega = np.zeros((packer.n_cov, q, q))
    a = np.arange(packer.n_cov)
    Omega[a, rows, :] += factor[:, None] * L_cols_T
    Omega[a, :, rows] += factor[:, None] * L_cols_T
    return Omega, factor


def _wilks_ftest(wilks, n_y, df_h, df_e):
    """Rao's F approximation for Wilks' lambda."""
    base = n_y * n_y + df_h * df_h - 5.0
    t = np.sqrt((n_y * n_y * df_h * df_h - 4.0) / base) if base > 0 else 1.0
    r = df_e - (n_y - df_h + 1.0) / 2.0
    u = (n_y * df_h - 2.0) / 4.0
    num_df = n_y * df_h
    den_df = r * t - 2.0 * u
    lam_t = wilks ** (1.0 / t)
    f = (1.0 - lam_t) / lam_t * den_df / num_df
    return f, num_df, den_df


def _pillai_ftest(pillai, s, m_par, n_par):
    """F approximation for Pillai's trace."""
    num_df = s * (2.0 * m_par + s + 1.0)
    den_df = s * (2.0 * n_par + s + 1.0)
    f = den_df / num_df * pillai / (s - pillai)
    return f, num_df, den_df


def _hotelling_ftest(hotelling, s, m_par, n_par, n_y, df_h):
    """F approximation for the Hotelling-Lawley trace."""
    if n_par > 0:
        b = (n_y + 2.0 * n_par) * (df_h + 2.0 * n_par)
        b = b / (2.0 * (2.0 * n_par + 1.0) * (n_par - 1.0))
        num_df = n_y * df_h
        den_df = 4.0 + (n_y * df_h + 2.0) / (b - 1.0)
        c = (den_df - 2.0) / (2.0 * n_par)
        f = den_df / num_df * hotelling / c
    else:
        num_df = s * (2.0 * m_par + s + 1.0)
        den_df = s * (s * n_par + 1.0)
        f = den_df / num_df / s * hotelling
    return f, num_df, den_df


def _roy_ftest(roy, n_y, df_h, df_e):
    """F approximation (an upper bound) for Roy's greatest root."""
    r = max(n_y, df_h)
    num_df = r
    den_df = df_e - r + df_h
    f = den_df / num_df * roy
    return f, num_df, den_df


def _canonical_eigenvalues(H, E, tol=1e-9):
    """Eigenvalues of (E + H)^{-1} H, i.e. squared canonical correlations.

    H and E are symmetric SSCP matrices; the result is sorted in
    descending order and clipped to [0, 1).
    """
    eigvals = sp.linalg.eigh(H, E + H, eigvals_only=True)
    eigvals = np.clip(np.real(eigvals), 0.0, 1.0 - tol)
    return np.sort(eigvals)[::-1]


def _manova_stats(eigvals, n_y, df_h, df_e):
    """Four classic MANOVA statistics with their approximate F tests.

    Parameters
    ----------
    eigvals : array
        Eigenvalues of (E + H)^{-1} H.
    n_y : int
        Number of response dimensions (rank of E).
    df_h, df_e : int
        Hypothesis and error degrees of freedom.

    Returns
    -------
    DataFrame
        Rows are the four statistics; columns hold the value, the
        numerator and denominator degrees of freedom, the approximate
        F statistic and its p-value.
    """
    rho2 = np.asarray(eigvals, dtype=float)
    rho2 = rho2[rho2 > 1e-9]
    eig_h = rho2 / (1.0 - rho2)
    s = min(n_y, df_h)
    m_par = (abs(n_y - df_h) - 1.0) / 2.0
    n_par = (df_e - n_y - 1.0) / 2.0
    wilks = float(np.prod(1.0 - rho2))
    pillai = float(np.sum(rho2))
    hotelling = float(np.sum(eig_h))
    roy = float(np.max(eig_h)) if eig_h.size else 0.0
    rows = [(wilks,) + _wilks_ftest(wilks, n_y, df_h, df_e),
            (pillai,) + _pillai_ftest(pillai, s, m_par, n_par),
            (hotelling,) + _hotelling_ftest(hotelling, s, m_par, n_par,
                                            n_y, df_h),
            (roy,) + _roy_ftest(roy, n_y, df_h, df_e)]
    table = pd.DataFrame(rows, index=MANOVA_STATS,
                         columns=['Value', 'F Value', 'Num DF', 'Den DF'])
    table['P Value'] = sp.stats.f.sf(table['F Value'].to_numpy(),
                                     table['Num DF'].to_numpy(),
                                     table['Den DF'].to_numpy())
    return table[MANOVA_COLUMNS]


class MVOLS(RegressionMixin, LikelihoodModel):
    """Multivariate Gaussian regression with log-Cholesky precision.

    Model: y_i ~ N_q(B' x_i, Sigma), where Sigma = Omega^{-1} and Omega
    has lower Cholesky L.  Parameters theta = (vec_F(B), lambda) with
    lambda the log-Cholesky parameterization of Omega.

    Fitting populates coefficient inference (``res``), per-response and
    multivariate measures of association (``rsquared``, ``assoc``,
    ``canonical_corr``) and MANOVA-style term tests (``mv_tests``).
    Arbitrary linear hypotheses are tested with ``test_hypothesis``.

    Parameters
    ----------
    formula : str or list of str, optional
        Patsy formula(s) (LHS may list multiple responses, e.g.
        "y1 + y2 ~ x1 + x2").
    data : DataFrame, optional
        Frame providing the variables referenced in the formula.
    X : array_like, optional
        Predictor matrix of shape (n, p).
    y : array_like, optional
        Response matrix of shape (n, q).
    weights : array_like, optional
        Observation weights of length n.
    """

    def __init__(self, formula=None, data=None, X=None, y=None, weights=None,
                 *args, **kwargs):
        super().__init__(formula=formula, data=data, X=X, y=y, weights=weights,
                         flatten_y=False, *args, **kwargs)
        self.xinds, self.yinds = self.model_data.indexes
        self.xcols, self.ycols = self.model_data.columns
        self.X, self.Y, self.weights = self.model_data
        if self.Y.ndim == 1:
            self.Y = self.Y.reshape(-1, 1)
        self.n = self.n_obs = self.X.shape[0]
        self.p = self.n_var = self.X.shape[1]
        self.q = self.Y.shape[1]
        self.design_info = self.model_data.design_info
        self.x_design_info, self.y_design_info = self.model_data.design_info
        self.formula = formula
        self.term_names = list(self.xcols)
        if isinstance(self.ycols, pd.Index):
            self.response_names = list(self.ycols)
        else:
            self.response_names = ['y{}'.format(i + 1) for i in range(self.q)]
        self.packer = MMRParameterPacker(self.p, self.q,
                                         term_names=self.term_names,
                                         response_names=self.response_names)
        self.param_labels = self.packer.theta_labels()
        self.params_init = self._closed_form_theta()

    def _closed_form_theta(self):
        sw = np.sqrt(self.weights)
        Xw = self.X * sw[:, None]
        Yw = self.Y * sw[:, None]
        L_x = np.linalg.cholesky(np.dot(Xw.T, Xw))
        Linv_x = _triangular_inv(L_x)
        B = np.dot(Linv_x.T, np.dot(Linv_x, np.dot(Xw.T, Yw)))
        E = self.Y - np.dot(self.X, B)
        Sigma = np.dot(E.T, E * self.weights[:, None]) / float(np.sum(self.weights))
        L_omega = _precision_chol_from_sigma(Sigma)
        return self.packer.pack(B, L_omega)

    @staticmethod
    def _loglike(params, data, packer):
        X, Y, w = data
        B, L = packer.unpack(params)
        E = Y - np.dot(X, B)
        U = np.dot(E, L)
        log_det = float(np.sum(np.log(np.diag(L))))
        n_eff = float(np.sum(w))
        quad = float(np.sum(np.sum(U * U, axis=1) * w))
        ll = log_det * n_eff - 0.5 * quad - 0.5 * packer.n_responses * LN2PI * n_eff
        return -ll

    def loglike(self, params, data=None, packer=None):
        data = self.model_data if data is None else data
        packer = self.packer if packer is None else packer
        return self._loglike(params, data, packer)

    @staticmethod
    def _gradient(params, data, packer):
        X, Y, w = data
        B, L = packer.unpack(params)
        Omega = np.dot(L, L.T)
        E = Y - np.dot(X, B)
        OmegaE = np.dot(E, Omega)
        WX = X * w[:, None]
        WU = np.dot(E, L) * w[:, None]
        M = np.dot(E.T, WU)
        g_beta = np.dot(WX.T, OmegaE)
        g_lamb = -M[packer.tril_rows, packer.tril_cols]
        g_lamb[packer.diag_mask] = float(np.sum(w)) - np.diag(L) * np.diag(M)
        return -np.concatenate([g_beta.ravel(order='F'), g_lamb])

    def gradient(self, params, data=None, packer=None):
        data = self.model_data if data is None else data
        packer = self.packer if packer is None else packer
        return self._gradient(params, data, packer)

    @staticmethod
    def _gradient_i(params, data, packer):
        X, Y, w = data
        B, L = packer.unpack(params)
        Omega = np.dot(L, L.T)
        E = Y - np.dot(X, B)
        OmegaE = np.dot(E, Omega)
        U = np.dot(E, L)
        n = X.shape[0]
        g_beta = (OmegaE[:, :, None] * X[:, None, :]).reshape(n, packer.n_beta)
        g_lamb = -E[:, packer.tril_rows] * U[:, packer.tril_cols]
        r_diag = packer.tril_rows[packer.diag_mask]
        g_lamb[:, packer.diag_mask] = (1.0 - np.diag(L)[r_diag]
                                       * E[:, r_diag] * U[:, r_diag])
        g = np.concatenate([g_beta, g_lamb], axis=1)
        return -g * w[:, None]

    def gradient_i(self, params, data=None, packer=None):
        data = self.model_data if data is None else data
        packer = self.packer if packer is None else packer
        return self._gradient_i(params, data, packer)

    @staticmethod
    def _hessian(params, data, packer):
        X, Y, w = data
        B, L = packer.unpack(params)
        Omega = np.dot(L, L.T)
        E = Y - np.dot(X, B)
        WX = X * w[:, None]
        WE = E * w[:, None]
        XtWX = np.dot(X.T, WX)
        XtWE = np.dot(X.T, WE)
        EtWE = np.dot(E.T, WE)
        nb, nc = packer.n_beta, packer.n_cov
        H = np.zeros((nb + nc, nb + nc))
        H[:nb, :nb] = -np.kron(Omega, XtWX)
        Omega_stack, factor = _omega_first_stack(L, packer)
        cross = np.einsum('ik,akj->aij', XtWE, Omega_stack)
        H_cross = _vec(cross).T
        H[:nb, nb:nb + nc] = H_cross
        H[nb:nb + nc, :nb] = H_cross.T
        rows, cols = packer.tril_rows, packer.tril_cols
        cols_eq = (cols[:, None] == cols[None, :])
        F = factor[:, None] * factor[None, :]
        DD = F * EtWE[rows[:, None], rows[None, :]] * cols_eq
        H_lamb = -DD
        inner = np.einsum('akl,kl->a', Omega_stack, EtWE)
        diag_idx = np.where(packer.diag_mask)[0]
        H_lamb[diag_idx, diag_idx] += -0.5 * inner[diag_idx]
        H[nb:, nb:] = H_lamb
        return -H

    def hessian(self, params, data=None, packer=None):
        data = self.model_data if data is None else data
        packer = self.packer if packer is None else packer
        return self._hessian(params, data, packer)

    def _optimize(self, t_init=None, opt_kws=None, data=None, packer=None):
        t_init = self.params_init if t_init is None else t_init
        data = self.model_data if data is None else data
        packer = self.packer if packer is None else packer
        opt_kws = {} if opt_kws is None else opt_kws
        default_kws = dict(method='trust-constr',
                           options=dict(verbose=0, gtol=1e-8, xtol=1e-10))
        opt_kws = {**default_kws, **opt_kws}
        args = (data, packer)
        return sp.optimize.minimize(self.loglike, t_init, args=args,
                                    jac=self.gradient, hess=self.hessian,
                                    **opt_kws)

    def _fit(self, method='closed_form', opt_kws=None):
        if method == 'mle':
            opt = self._optimize(opt_kws=opt_kws)
            return opt.x, opt
        return self._closed_form_theta(), None

    def fit(self, method='closed_form', opt_kws=None):
        self.params, self.opt = self._fit(method, opt_kws=opt_kws)
        self._store_estimates()
        self._store_association()
        self.mv_tests = self.multivariate_tests()
        self._store_summary()

    def _store_estimates(self):
        """Unpack the fitted theta into coefficients, covariances and
        residual quantities."""
        n, nb = self.n, self.packer.n_beta
        self.n_params = len(self.params)
        self.n_beta = nb
        self.params_hess = self.hessian(self.params)
        self.params_cov = np.linalg.pinv(self.params_hess)
        self.params_se = np.sqrt(np.diag(self.params_cov))
        self.res = self._parameter_inference(self.params, self.params_se,
                                             n - self.n_params,
                                             self.param_labels)
        self.beta = self.coefs = self.packer.unpack_beta(self.params)
        self.beta_cov = self.coefs_cov = self.params_cov[:nb, :nb]
        self.beta_se = self.coefs_se = self.params_se[:nb]
        self.precision_chol = self.packer.unpack_precision_chol(self.params)
        self.precision = np.dot(self.precision_chol, self.precision_chol.T)
        precision_chol_inv = _triangular_inv(self.precision_chol)
        self.sigma_mle = np.dot(precision_chol_inv.T, precision_chol_inv)
        self.fitted_values = np.dot(self.X, self.beta)
        self.residuals = self.Y - self.fitted_values
        self.llf = -self.loglike(self.params)
        self._store_sscp()

    def _store_sscp(self):
        """Cross-product matrices used by the association measures and
        the multivariate hypothesis tests."""
        weighted_x = self.X * self.weights[:, None]
        weighted_resid = self.residuals * self.weights[:, None]
        self.df_resid = max(self.n - self.p, 1)
        self.gram = np.dot(self.X.T, weighted_x)
        gram_chol_inv = _triangular_inv(np.linalg.cholesky(self.gram))
        self.gram_inv = np.dot(gram_chol_inv.T, gram_chol_inv)
        self.resid_sscp = np.dot(self.residuals.T, weighted_resid)
        self.sigma_unbiased = self.resid_sscp / self.df_resid
        centered_y = self.Y - np.average(self.Y, axis=0, weights=self.weights)
        self.total_sscp = np.dot(centered_y.T, centered_y * self.weights[:, None])

    @staticmethod
    def _hypothesis_sscp(beta, gram_inv, resid_sscp, L, M=None):
        """Hypothesis SSCP H and error SSCP E for the linear hypothesis
        L B M = 0."""
        coef = beta if M is None else np.dot(beta, M)
        error = resid_sscp if M is None else np.dot(M.T, np.dot(resid_sscp, M))
        lbm = np.dot(L, coef)
        middle = np.dot(L, np.dot(gram_inv, L.T))
        hypothesis = np.dot(lbm.T, np.linalg.solve(middle, lbm))
        return hypothesis, error

    def test_hypothesis(self, L, M=None):
        """Multivariate test of the general linear hypothesis L B M = 0.

        Parameters
        ----------
        L : array_like
            Contrast matrix with one column per predictor (p columns).
        M : array_like, optional
            Response transform with one row per response (q rows).  When
            omitted the responses are left untransformed.

        Returns
        -------
        DataFrame
            The four MANOVA statistics with their approximate F tests.
        """
        L = np.atleast_2d(np.asarray(L, dtype=float))
        M = None if M is None else np.atleast_2d(np.asarray(M, dtype=float))
        H, E = self._hypothesis_sscp(self.beta, self.gram_inv,
                                     self.resid_sscp, L, M)
        df_h = int(np.linalg.matrix_rank(L))
        eigvals = _canonical_eigenvalues(H, E)
        return _manova_stats(eigvals, E.shape[0], df_h, self.df_resid)

    def _term_contrasts(self):
        """Contrast matrix L for each non-intercept design term."""
        if self.x_design_info is not None:
            names = self.x_design_info.term_names
            spans = [(nm, self.x_design_info.term_name_slices[nm])
                     for nm in names if nm != 'Intercept']
        else:
            constant = np.ptp(self.X, axis=0) == 0
            spans = [(self.term_names[i], slice(i, i + 1))
                     for i in range(self.p) if not constant[i]]
        columns = np.arange(self.p)
        contrasts = {}
        for name, span in spans:
            idx = columns[span]
            L = np.zeros((len(idx), self.p))
            L[np.arange(len(idx)), idx] = 1.0
            contrasts[name] = L
        return contrasts

    def multivariate_tests(self, M=None):
        """MANOVA-style tests for each design term and for the joint
        hypothesis that every slope is zero.

        Returns
        -------
        DataFrame
            A frame indexed by (term, statistic).  The 'Regression'
            block holds the joint test of all slopes.
        """
        contrasts = self._term_contrasts()
        tables = {name: self.test_hypothesis(L, M=M)
                  for name, L in contrasts.items()}
        if len(contrasts) > 1:
            L_all = np.concatenate(list(contrasts.values()), axis=0)
            tables['Regression'] = self.test_hypothesis(L_all, M=M)
        if not tables:
            return pd.DataFrame(columns=MANOVA_COLUMNS)
        return pd.concat(tables, names=['Term', 'Statistic'])

    def _measures_of_association(self):
        """Multivariate effect-size measures for the all-slopes
        hypothesis.  Returns the table, the canonical correlations and
        the generalized R-squared 1 - Wilks."""
        contrasts = self._term_contrasts()
        if not contrasts:
            return pd.DataFrame(columns=['Value']), np.array([]), np.nan
        L = np.concatenate(list(contrasts.values()), axis=0)
        H, E = self._hypothesis_sscp(self.beta, self.gram_inv,
                                     self.resid_sscp, L)
        df_h = np.linalg.matrix_rank(L)
        rho2 = _canonical_eigenvalues(H, E)
        rho2 = rho2[rho2 > 1e-9]
        s = max(min(self.q, df_h), 1)
        wilks = float(np.prod(1.0 - rho2)) if rho2.size else 1.0
        pillai = float(np.sum(rho2))
        hotelling = float(np.sum(rho2 / (1.0 - rho2)))
        roy = float(np.max(rho2 / (1.0 - rho2))) if rho2.size else 0.0
        measures = {
            "Wilks' lambda": wilks,
            "Pillai's trace": pillai,
            'Hotelling-Lawley trace': hotelling,
            "Roy's greatest root": roy,
            'Generalized R2 (1 - Wilks)': 1.0 - wilks,
            'Eta-squared (Wilks)': 1.0 - wilks ** (1.0 / s),
            'Eta-squared (Pillai)': pillai / s,
            'Eta-squared (Hotelling)': (hotelling / s) / (1.0 + hotelling / s),
            'Eta-squared (Roy)': roy / (1.0 + roy),
            'Mean squared canonical corr':
                float(np.mean(rho2)) if rho2.size else 0.0}
        table = pd.DataFrame({'Value': measures})
        return table, np.sqrt(rho2), 1.0 - wilks

    def _response_rsquared(self):
        """Per-response coefficient of determination and its adjustment."""
        sse = np.diag(self.resid_sscp)
        sst = np.diag(self.total_sscp)
        r2 = 1.0 - sse / sst
        r2_adj = 1.0 - (1.0 - r2) * (self.n - 1.0) / self.df_resid
        return pd.DataFrame({'R2': r2, 'R2_adj': r2_adj},
                            index=self.response_names)

    def _store_association(self):
        """Compute and store univariate and multivariate measures of
        association."""
        self.rsquared = self._response_rsquared()
        self.assoc, self.canonical_corr, self.r2_det =  self._measures_of_association()
        sse_tr = np.sum(np.diag(self.resid_sscp))
        sst_tr = np.sum(np.diag(self.total_sscp))
        self.r2_trace = 1.0 - sse_tr / sst_tr

    def _store_summary(self):
        """Information criteria and a compact summary table."""
        self.aic, self.aicc, self.bic, self.caic = self._get_information(
            -self.llf, self.n_params, self.n)
        sumstats = {'AIC': self.aic, 'AICC': self.aicc, 'BIC': self.bic,
                    'CAIC': self.caic, 'LLF': self.llf,
                    'R2_trace': self.r2_trace, 'R2_det': self.r2_det}
        self.sumstats = pd.DataFrame(sumstats, index=['Statistic']).T

    def predict(self, X=None, params=None):
        params = self.params if params is None else params
        X = self.X if X is None else np.asarray(X, dtype=float)
        return np.dot(X, self.packer.unpack_beta(params))
