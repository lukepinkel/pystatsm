# -*- coding: utf-8 -*-
"""
Data generating process for multivariate (W)LS testing & benchmarking.

Y = X B + E,   E_i ~ N_q(0, Sigma).

Returns the true packed theta in the same column-major / log-Cholesky
convention used by mvols.MMRParameterPacker (a self-contained copy of
that packer is included here so this module has no local imports).
"""
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# self-contained packer (mirrors mvols.MMRParameterPacker)
# ----------------------------------------------------------------------

class MMRParameterPacker(object):

    def __init__(self, n_terms, n_responses, term_names=None,
                 response_names=None):
        self.n_terms = int(n_terms)
        self.n_responses = int(n_responses)
        self.term_names = (list(term_names) if term_names is not None
                           else ["x{}".format(i) for i in range(n_terms)])
        self.response_names = (list(response_names) if response_names is not None
                               else ["y{}".format(i + 1) for i in range(n_responses)])
        self.n_beta = self.n_terms * self.n_responses
        self.n_cov = self.n_responses * (self.n_responses + 1) // 2
        self.n_params = self.n_beta + self.n_cov
        self.tril_rows, self.tril_cols = np.tril_indices(self.n_responses)
        self.diag_mask = self.tril_rows == self.tril_cols

    def precision_chol_to_lambda(self, L):
        lamb = L[self.tril_rows, self.tril_cols].astype(float).copy()
        lamb[self.diag_mask] = np.log(lamb[self.diag_mask])
        return lamb

    def pack(self, beta, precision_chol):
        beta_vec = np.asarray(beta).ravel(order="F")
        return np.concatenate([beta_vec,
                               self.precision_chol_to_lambda(precision_chol)])


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

def _toeplitz_corr(d, rho):
    idx = np.arange(d)
    return rho ** np.abs(idx[:, None] - idx[None, :])


def _ar1_chol(d, rho):
    return np.linalg.cholesky(_toeplitz_corr(d, rho))


def _make_x(n, p, x_correlation, rho_x, rng):
    if x_correlation == "toeplitz":
        Sigma_X = _toeplitz_corr(p, rho_x)
    elif x_correlation in ("identity", "iid", None):
        Sigma_X = np.eye(p)
    else:
        raise ValueError("unknown x_correlation: " + str(x_correlation))
    Lx = np.linalg.cholesky(Sigma_X)
    X = np.dot(rng.standard_normal((n, p)), Lx.T)
    return X, Sigma_X


def _make_beta_slope(p, q, structure, rank, sparsity, rng):
    if structure == "low_rank_plus_sparse":
        U = rng.standard_normal((p, rank))
        D = np.abs(rng.standard_normal(rank)) + 0.5
        V = rng.standard_normal((q, rank))
        low = np.dot(U * D, V.T)
        mask = rng.uniform(size=(p, q)) < sparsity
        sparse = rng.standard_normal((p, q)) * mask
        return low + sparse
    if structure == "sparse":
        mask = rng.uniform(size=(p, q)) < sparsity
        return rng.standard_normal((p, q)) * mask
    if structure == "dense":
        return rng.standard_normal((p, q))
    raise ValueError("unknown beta_structure: " + str(structure))


def _make_residual_cov(q, error_correlation, rho_e):
    if error_correlation == "toeplitz":
        return _toeplitz_corr(q, rho_e)
    if error_correlation in ("identity", "iid", None):
        return np.eye(q)
    raise ValueError("unknown error_correlation: " + str(error_correlation))


def _scale_to_target_r2(M, Sigma_base, target_r2):
    Mc = M - M.mean(axis=0)
    v_signal = float(np.trace(np.dot(Mc.T, Mc)) / M.shape[0])
    v_noise = float(np.trace(Sigma_base))
    if v_signal <= 0 or target_r2 <= 0 or target_r2 >= 1:
        return Sigma_base
    c = v_signal * (1.0 - target_r2) / (target_r2 * v_noise)
    return c * Sigma_base


def _draw_residuals(n, q, Sigma, noise, rng):
    C = np.linalg.cholesky(Sigma)
    if noise == "normal":
        G = rng.standard_normal((n, q))
    elif noise == "t":
        df = 5.0
        G = rng.standard_t(df=df, size=(n, q)) / np.sqrt(df / (df - 2.0))
    else:
        raise ValueError("unknown noise: " + str(noise))
    return np.dot(G, C.T)


# ----------------------------------------------------------------------
# main DGP
# ----------------------------------------------------------------------

def generate_mmr_data(n=500, p=20, q=5, fit_intercept=True,
                      beta_structure="low_rank_plus_sparse", beta_rank=2,
                      beta_sparsity=0.2, x_correlation="toeplitz",
                      rho_x=0.5, error_correlation="toeplitz", rho_e=0.4,
                      target_r2=0.5, noise="normal", random_state=None,
                      return_dataframe=False):
    rng = np.random.default_rng(random_state)
    feature_names = ["x{}".format(i + 1) for i in range(p)]
    response_names = ["y{}".format(i + 1) for i in range(q)]
    term_names = (["Intercept"] + feature_names) if fit_intercept else feature_names

    X, Sigma_X = _make_x(n, p, x_correlation, rho_x, rng)
    B_slope = _make_beta_slope(p, q, beta_structure, beta_rank,
                               beta_sparsity, rng)
    if fit_intercept:
        intercept = rng.standard_normal(q)
        B = np.vstack([intercept[None, :], B_slope])
        X_design = np.column_stack([np.ones(n), X])
    else:
        intercept = np.zeros(q)
        B = B_slope
        X_design = X
    M = np.dot(X_design, B)

    Sigma_base = _make_residual_cov(q, error_correlation, rho_e)
    Sigma = _scale_to_target_r2(M, Sigma_base, target_r2)
    E = _draw_residuals(n, q, Sigma, noise, rng)
    Y = M + E

    Omega = np.linalg.inv(Sigma)
    L_omega = np.linalg.cholesky(Omega)
    packer = MMRParameterPacker(
        n_terms=B.shape[0], n_responses=q,
        term_names=term_names, response_names=response_names)
    theta_true = packer.pack(B, L_omega)

    out = {
        "X": X,
        "Y": Y,
        "X_design": X_design,
        "B": B,
        "coef": B_slope,
        "intercept": intercept,
        "Sigma": Sigma,
        "Omega": Omega,
        "precision_chol": L_omega,
        "theta_true": theta_true,
        "signal": M,
        "E": E,
        "Sigma_X": Sigma_X,
        "feature_names": feature_names,
        "term_names": term_names,
        "response_names": response_names,
        "packer": packer,
    }
    if return_dataframe:
        out["X"] = pd.DataFrame(X, columns=feature_names)
        out["Y"] = pd.DataFrame(Y, columns=response_names)
    return out
