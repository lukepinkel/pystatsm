import numpy as np
import scipy as sp
import pandas as pd
from .regression_model import RegressionMixin
from .likelihood_model import LikelihoodModel
from ..utilities.linalg_operations import _vec

LN2PI = np.log(2 * np.pi)


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


class MVOLS(RegressionMixin, LikelihoodModel):
    """Multivariate Gaussian regression with log-Cholesky precision.

    Model: y_i ~ N_q(B' x_i, Sigma), where Sigma = Omega^{-1} and Omega
    has lower Cholesky L.  Parameters theta = (vec_F(B), lambda) with
    lambda the log-Cholesky parameterization of Omega.

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
        n, n_params = self.n, len(self.params)
        nb = self.packer.n_beta
        self.n_params = n_params
        self.params_hess = self.hessian(self.params)
        self.params_cov = np.linalg.pinv(self.params_hess)
        self.params_se = np.sqrt(np.diag(self.params_cov))
        self.res = self._parameter_inference(self.params, self.params_se,
                                             n - n_params, self.param_labels)
        self.beta = self.coefs = self.packer.unpack_beta(self.params)
        self.precision_chol = self.packer.unpack_precision_chol(self.params)
        self.precision = np.dot(self.precision_chol, self.precision_chol.T)
        Linv = _triangular_inv(self.precision_chol)
        self.sigma_mle = np.dot(Linv.T, Linv)
        self.beta_cov = self.coefs_cov = self.params_cov[:nb, :nb]
        self.beta_se = self.coefs_se = self.params_se[:nb]
        self.n_beta = nb
        self.fitted_values = np.dot(self.X, self.beta)
        self.residuals = self.Y - self.fitted_values
        self.llf = -self.loglike(self.params)
        E = self.residuals
        S_e = np.dot(E.T, E * self.weights[:, None])
        df_resid = max(float(n - self.p), 1.0)
        self.sigma_unbiased = S_e / df_resid
        Yc = self.Y - self.Y.mean(axis=0)
        sse_tr = float(np.trace(np.dot(E.T, E)))
        sst_tr = float(np.trace(np.dot(Yc.T, Yc)))
        try:
            r2_det = 1.0 - (np.linalg.det(np.dot(E.T, E))
                            / np.linalg.det(np.dot(Yc.T, Yc)))
        except np.linalg.LinAlgError:
            r2_det = np.nan
        self.r2_trace = 1.0 - sse_tr / sst_tr
        self.r2_det = r2_det
        self.aic, self.aicc, self.bic, self.caic = self._get_information(
            self.llf, n_params, n)
        sumstats = {'AIC': self.aic, 'AICC': self.aicc, 'BIC': self.bic,
                    'CAIC': self.caic, 'LLF': self.llf,
                    'R2_trace': self.r2_trace, 'R2_det': self.r2_det}
        self.sumstats = pd.DataFrame(sumstats, index=['Statistic']).T

    def predict(self, X=None, params=None):
        params = self.params if params is None else params
        X = self.X if X is None else np.asarray(X, dtype=float)
        return np.dot(X, self.packer.unpack_beta(params))
