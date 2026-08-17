import numpy as np
import pandas as pd
from ..output import get_param_table
from .repweights import replicate_variance


def _design_matrix(design, x):
    if isinstance(x, str):
        x = [x]
    if isinstance(x, (list, tuple)):
        names = list(x)
        X = design.df[names].to_numpy(dtype=np.float64)
    else:
        X = np.asarray(x, dtype=np.float64)
        X = X.reshape(-1, 1) if X.ndim == 1 else X
        names = [f"x{i}" for i in range(X.shape[1])]
    if X.shape[0] != design.n:
        raise ValueError(f"x has {X.shape[0]} rows, design has {design.n}")
    return X, names


def _replicates(design, vcov, n_rep, rng):
    if vcov == "jackknife":
        return design.jackknife_replicates()
    if vcov == "bootstrap":
        return design.bootstrap_replicates(n_rep=n_rep, rng=rng)
    raise ValueError(f"unknown vcov: {vcov!r}")


def _psu_totals(design, X):
    starts = design.layout.ind_psu[:-1]
    return np.add.reduceat(X * design.w[:, None], starts, axis=0)


def _resolve_center(center, est):
    if center is None:
        return None
    if isinstance(center, str) and center == "estimate":
        return est
    return np.asarray(center, dtype=np.float64)


def survey_total(design, x, vcov="linearized", n_rep=500, rng=None,
                 center=None):
    X, names = _design_matrix(design, x)
    U = X * design.w[:, None]
    est = U.sum(axis=0)
    if vcov == "linearized":
        V = design.meat(U)
    else:
        mult, rscales, scale = _replicates(design, vcov, n_rep, rng)
        tot_rep = mult.dot(_psu_totals(design, X))
        V = replicate_variance(tot_rep, rscales, scale,
                               center=_resolve_center(center, est))
    se = np.sqrt(np.diag(V))
    res = get_param_table(est, se, degfree=design.degf(), index=names,
                          parameter_label="total")
    return res, V


def survey_mean(design, x, vcov="linearized", n_rep=500, rng=None,
                center=None):
    X, names = _design_matrix(design, x)
    w = design.w
    n_hat = w.sum()
    est = np.dot(w, X) / n_hat
    if vcov == "linearized":
        Z = (X - est) * (w / n_hat)[:, None]
        V = design.meat(Z)
    else:
        mult, rscales, scale = _replicates(design, vcov, n_rep, rng)
        starts = design.layout.ind_psu[:-1]
        tx = np.add.reduceat(X * w[:, None], starts, axis=0)
        tw = np.add.reduceat(w, starts)
        mean_rep = mult.dot(tx) / mult.dot(tw)[:, None]
        V = replicate_variance(mean_rep, rscales, scale,
                               center=_resolve_center(center, est))
    se = np.sqrt(np.diag(V))
    res = get_param_table(est, se, degfree=design.degf(), index=names,
                          parameter_label="mean")
    return res, V


def _survey_by(stat, design, x, by, **kws):
    levels = np.sort(pd.unique(design.df[by]))
    values = design.df[by].to_numpy()
    tables = []
    for level in levels:
        res, _ = stat(design.subset(values == level), x, **kws)
        tables.append(res)
    return pd.concat(tables, keys=levels, names=[by])


def survey_mean_by(design, x, by, **kws):
    return _survey_by(survey_mean, design, x, by, **kws)


def survey_total_by(design, x, by, **kws):
    return _survey_by(survey_total, design, x, by, **kws)
