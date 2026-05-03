import numba
import numpy as np
import pandas as pd
from dataclasses import dataclass


@numba.jit(nopython=True)
def _grouped_cov(X, n_str, n_grp, ind_str, ind_psu, n_psu_per_str, ssf):
    p = X.shape[1]
    psu_sum = np.zeros((n_grp, p))
    out = np.zeros((p, p))
    for k in range(n_grp):
        psu_sum[k] = np.sum(X[ind_psu[k]:ind_psu[k + 1]], axis=0)
    for h in range(n_str):
        block = psu_sum[ind_str[h]:ind_str[h + 1]]
        block -= np.sum(block, axis=0) / n_psu_per_str[h]
        out += np.dot(block.T, block) * ssf[h]
    return out


@dataclass(frozen=True, slots=True)
class IndexLayout:
    n_str: int
    n_grp: int
    ind_str: np.ndarray         # (n_str+1,) cum pointer into PSU array
    ind_psu: np.ndarray         # (n_grp+1,) cum pointer into obs array
    n_psu_per_str: np.ndarray   # (n_str,) PSU count per stratum
    ssf: np.ndarray             # (n_str,) (1 - fpc_h) * n_h / (n_h - 1)


def _sizes_to_ptr(sizes):
    out = np.zeros(len(sizes) + 1, dtype=np.int64)
    out[1:] = np.cumsum(sizes)
    return out


def _ensure_columns(df, strata, psuind):
    df = df.copy(deep=False)
    if strata is None:
        strata = "_strata"
        df[strata] = 0
    if psuind is None:
        psuind = "_psuind"
        df[psuind] = np.arange(len(df))
    return df, strata, psuind


def _has_singletons(df, strata, psuind):
    sizes = df.groupby([strata, psuind]).size()
    return bool((sizes.groupby(strata).size() == 1).any())


def _aggregate_singletons(df, strata, psuind):
    # Each stratum that has exactly one PSU gets folded into a single new
    # stratum, with its sole PSU re-numbered 0..k-1 to guarantee no collision
    # with existing PSU ids in the new stratum.
    sizes = df.groupby([strata, psuind]).size()
    n_psu_per_str = sizes.groupby(strata).size()
    singletons = n_psu_per_str[n_psu_per_str == 1].index
    if len(singletons) == 0:
        return df
    df = df.copy()
    new_stratum = df[strata].max() + 1
    new_psu = {s: i for i, s in enumerate(singletons)}
    mask = df[strata].isin(singletons)
    df.loc[mask, psuind] = df.loc[mask, strata].map(new_psu).to_numpy()
    df.loc[mask, strata] = new_stratum
    return df


def _build_layout(df, strata, psuind, fpc):
    sizes = df.groupby([strata, psuind]).size().reset_index(name="n")
    n_psu_per_str = sizes.groupby(strata).size().to_numpy(dtype=np.int64)
    n_psu_obs = sizes["n"].to_numpy(dtype=np.int64)
    n_str = n_psu_per_str.size
    n_grp = n_psu_obs.size
    if fpc is None:
        fpc = np.zeros(n_str)
    elif np.shape(fpc) != (n_str,):
        raise ValueError(f"fpc must have shape ({n_str},), got {np.shape(fpc)}")
    # ssf = (1 - f_h) * n_h / (n_h - 1); singleton strata (n_h == 1) cannot
    # contribute within-stratum variation, so zero out their scaling. This
    # matches R survey's lonely.psu="remove" behavior.
    n = n_psu_per_str
    denom = np.where(n > 1, n - 1, 1)
    ssf = np.where(n > 1, (1.0 - fpc) * n / denom, 0.0)
    return IndexLayout(n_str, n_grp,
                       _sizes_to_ptr(n_psu_per_str),
                       _sizes_to_ptr(n_psu_obs),
                       n_psu_per_str, ssf)


class SampleDesign:
    """Stratified-cluster sample design.

    The design owns: a sorted dataframe, a per-row weight vector, and an
    IndexLayout describing the strata/PSU pointer arithmetic. It exposes
    `meat(U)` which returns the design covariance of `sum(U_i)` for any
    score-contribution matrix U whose rows align with `self.df`.
    """

    def __init__(self, df, strata=None, psuind=None, weight=None,
                 fpc=None, singleton="aggregate"):
        if singleton not in ("aggregate", "none", "error"):
            raise ValueError(f"unknown singleton policy: {singleton!r}")
        df, strata, psuind = _ensure_columns(df, strata, psuind)
        if _has_singletons(df, strata, psuind):
            if singleton == "aggregate":
                df = _aggregate_singletons(df, strata, psuind)
            elif singleton == "error":
                raise ValueError("design has singleton PSUs")
        df = df.sort_values([strata, psuind]).reset_index(drop=True)
        if weight is None:
            w = np.ones(len(df))
        else:
            w = df[weight].to_numpy(dtype=np.float64)
        self.df = df
        self.strata = strata
        self.psuind = psuind
        self.weight = weight
        self.layout = _build_layout(df, strata, psuind, fpc)
        self.w = w

    @property
    def n(self):
        return len(self.df)

    @property
    def n_str(self):
        return self.layout.n_str

    @property
    def n_grp(self):
        return self.layout.n_grp

    def meat(self, U):
        U = np.ascontiguousarray(U, dtype=np.float64)
        if U.ndim != 2 or U.shape[0] != self.n:
            raise ValueError(f"U must have shape ({self.n}, p), got {U.shape}")
        L = self.layout
        return _grouped_cov(U, L.n_str, L.n_grp, L.ind_str, L.ind_psu,
                            L.n_psu_per_str, L.ssf)

    def subset(self, mask):
        # Subpopulation: zero out weights outside `mask` while preserving the
        # full PSU/stratum structure (R survey::subset semantics).
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (self.n,):
            raise ValueError(f"mask must have shape ({self.n},)")
        out = self.__class__.__new__(self.__class__)
        out.df = self.df
        out.strata = self.strata
        out.psuind = self.psuind
        out.weight = self.weight
        out.layout = self.layout
        out.w = np.where(mask, self.w, 0.0)
        return out

    def __repr__(self):
        return (f"SampleDesign(n={self.n}, n_str={self.n_str}, "
                f"n_grp={self.n_grp}, weight={self.weight!r})")


def design_sandwich(A, U, design):
    # V = A M A where M = design.meat(U). A is the model bread (p, p);
    # U is the per-observation score contribution (n, p), already weighted.
    M = design.meat(U)
    return np.dot(A, np.dot(M, A))
