#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mixed-model simulation built on the same RandomEffects/RandomEffectTerm
machinery used by LMM2.  Z, G, and theta layout match the fitter by
construction, so recovery testing is just `theta_hat ≈ sim.theta_true`.
"""
import numpy as np
import scipy as sp
import pandas as pd
import patsy
from dataclasses import dataclass

from .re_mod import RandomEffects, LMM2
from ..utilities.linalg_operations import vech
from ..utilities.random import exact_rmvnorm
from ..utilities.cov_utils import _exact_cov
from ..utilities.formula import parse_random_effects


@dataclass
class RanefSpec:
    re_formula: str
    group_var: str
    G: np.ndarray
    membership: np.ndarray = None
    n_groups: int = None
    n_per: int = None


@dataclass
class CovariateSpec:
    cont_vars: list
    mean: np.ndarray
    cov: np.ndarray


@dataclass
class SimSpec:
    n_obs: int
    response: str
    fe_formula: str
    beta: np.ndarray
    ranef: list
    resid_var: float
    cov_spec: object = None

    @classmethod
    def from_formula(cls, formula, n_obs, beta, ranef_G, resid_var,
                     groupings=None, cov_spec=None):
        """Build a SimSpec from an lme4-style formula.

        ranef_G is keyed (or ordered) the same way as the random-effect terms
        appear in the formula: by group_var name (dict) or in left-to-right
        order (list/tuple). groupings is a dict {group_var: membership_array}
        (typically from build_groupings); if omitted, each RanefSpec falls
        back to its own (n_groups, n_per) — supply via the dict-of-tuples
        shortcut explained below.
        """
        info = parse_random_effects(formula)
        re_terms = info['re_terms']
        fe_form = info['fe_form'].strip()
        response = info['y_vars'][0]
        if isinstance(ranef_G, dict):
            Gs = [ranef_G[gr] for _, gr in re_terms]
        else:
            Gs = list(ranef_G)
        ranef = []
        for (fr, gr), G in zip(re_terms, Gs):
            kw = {}
            if groupings is not None and gr in groupings:
                kw['membership'] = groupings[gr]
            ranef.append(RanefSpec(re_formula=fr, group_var=gr,
                                   G=np.asarray(G), **kw))
        return cls(n_obs=n_obs, response=response, fe_formula=fe_form,
                   beta=np.asarray(beta), ranef=ranef, resid_var=resid_var,
                   cov_spec=cov_spec)


def balanced_membership(n_groups, n_per):
    return np.repeat(np.arange(n_groups), n_per)


@dataclass
class Grouping:
    """Top-level grouping factor for build_groupings.

    Two top-level Groupings are crossed automatically when one uses
    cycle='repeat' (slow, blocks of obs share the same level) and the other
    uses cycle='tile' (fast, level cycles every observation). For complex /
    unbalanced layouts, supply membership directly.
    """
    name: str
    n_levels: int = None
    membership: np.ndarray = None
    cycle: str = 'repeat'   # 'repeat' (slow-varying) or 'tile' (fast-varying)


@dataclass
class Nested:
    """Factor nested in another. Each parent level gets exactly n_per_parent
    distinct child levels; observations within a parent block split evenly
    across the child levels. Total child levels = parent.n_levels * n_per_parent.
    """
    name: str
    parent: str
    n_per_parent: int = None
    membership: np.ndarray = None


def build_groupings(n_obs, *factors):
    """Resolve a sequence of Grouping / Nested specs into a dict
    {name: membership_array}. Top-level Groupings are processed first so
    Nested can reference their parents."""
    out = {}
    for f in factors:
        if not isinstance(f, Grouping):
            continue
        if f.membership is not None:
            m = np.asarray(f.membership)
            if len(m) != n_obs:
                raise ValueError(f"{f.name}: membership length {len(m)} != n_obs {n_obs}")
            out[f.name] = m
            continue
        if f.n_levels is None:
            raise ValueError(f"{f.name}: need n_levels or membership")
        if n_obs % f.n_levels != 0:
            raise ValueError(f"{f.name}: n_obs ({n_obs}) not divisible by "
                             f"n_levels ({f.n_levels})")
        n_per = n_obs // f.n_levels
        if f.cycle == 'repeat':
            out[f.name] = np.repeat(np.arange(f.n_levels), n_per)
        elif f.cycle == 'tile':
            out[f.name] = np.tile(np.arange(f.n_levels), n_per)
        else:
            raise ValueError(f"{f.name}: cycle must be 'repeat' or 'tile'")
    for f in factors:
        if not isinstance(f, Nested):
            continue
        if f.membership is not None:
            m = np.asarray(f.membership)
            if len(m) != n_obs:
                raise ValueError(f"{f.name}: membership length {len(m)} != n_obs {n_obs}")
            out[f.name] = m
            continue
        if f.parent not in out:
            raise ValueError(f"{f.name}: parent '{f.parent}' must be a Grouping "
                             f"declared before it")
        if f.n_per_parent is None:
            raise ValueError(f"{f.name}: need n_per_parent or membership")
        parent = out[f.parent]
        n_parent_levels = int(parent.max()) + 1
        child = np.zeros(n_obs, dtype=np.int64)
        for p in range(n_parent_levels):
            mask = parent == p
            n_in_parent = int(mask.sum())
            if n_in_parent % f.n_per_parent != 0:
                raise ValueError(f"{f.name}: parent level {p} has {n_in_parent} "
                                 f"obs, not divisible by n_per_parent={f.n_per_parent}")
            child[mask] = (p * f.n_per_parent +
                           np.repeat(np.arange(f.n_per_parent),
                                     n_in_parent // f.n_per_parent))
        out[f.name] = child
    return out


def _membership(r, n_obs):
    if r.membership is not None:
        m = np.asarray(r.membership)
        if len(m) != n_obs:
            raise ValueError(f"membership length {len(m)} != n_obs {n_obs}")
        return m
    if r.n_groups is None or r.n_per is None:
        raise ValueError("ranef needs either membership or (n_groups, n_per)")
    if r.n_groups * r.n_per != n_obs:
        raise ValueError("n_groups * n_per must equal n_obs for balanced spec")
    return balanced_membership(r.n_groups, r.n_per)


def _n_groups(r, n_obs):
    if r.n_groups is not None:
        return r.n_groups
    return int(pd.unique(_membership(r, n_obs)).shape[0])


def to_lme4_formula(spec):
    parts = [f"{spec.response} ~ {spec.fe_formula}"]
    for r in spec.ranef:
        parts.append(f"({r.re_formula} | {r.group_var})")
    return " + ".join(parts)


def build_frame(spec, rng):
    df = pd.DataFrame(index=np.arange(spec.n_obs))
    for r in spec.ranef:
        df[r.group_var] = _membership(r, spec.n_obs)
    if spec.cov_spec is not None:
        c = spec.cov_spec
        x_cov = np.atleast_2d(np.asarray(c.cov))
        x_mean = np.asarray(c.mean)
        seed = int(rng.integers(0, 2**31 - 1))
        xvals = exact_rmvnorm(x_cov, spec.n_obs, mu=x_mean, seed=seed)
        df[list(c.cont_vars)] = xvals
    df[spec.response] = 0.0
    return df


def build_design(spec, df):
    re_terms = [(r.re_formula, r.group_var) for r in spec.ranef]
    re_mod = RandomEffects(re_terms, data=df)
    X = patsy.dmatrix(spec.fe_formula, data=df, return_type='dataframe').values
    return X, re_mod


def spec_to_theta(spec):
    parts = [vech(np.asarray(r.G)) for r in spec.ranef]
    parts.append(np.array([float(spec.resid_var)]))
    return np.concatenate(parts)


def draw_ranefs(spec, rng, exact=False, dist=None):
    blocks = []
    for r in spec.ranef:
        ng = _n_groups(r, spec.n_obs)
        G = np.asarray(r.G)
        nv = G.shape[0]
        mean = np.zeros(nv)
        if dist is None:
            U = rng.multivariate_normal(mean, G, size=ng)
        else:
            U = dist(mean, G, ng, rng)
        if exact:
            # Center then rescale so empirical mean is exactly 0 and empirical
            # covariance is exactly G. _exact_cov alone only enforces the cov.
            U = U - U.mean(axis=0)
            U = _exact_cov(U, mean=mean, cov=G)
        blocks.append(U.reshape(-1))
    return np.concatenate(blocks)


def draw_residuals(spec, rng, exact=False, dist=None):
    if dist is None:
        eps = rng.normal(size=spec.n_obs)
    else:
        eps = dist(spec.n_obs, rng)
    if exact:
        eps = (eps - eps.mean()) / eps.std()
    return eps * np.sqrt(float(spec.resid_var))


def variance_components(spec, X, re_mod):
    eta_fe = X.dot(np.asarray(spec.beta)).reshape(-1)
    v_fe = float(np.var(eta_fe))
    G = re_mod.update_gcov(spec_to_theta(spec))
    Z = re_mod.Z
    ZG = Z.dot(G)
    v_re = float(Z.multiply(ZG).sum() / Z.shape[0])
    return v_fe, v_re


def rescale_to_var_ratios(spec, X, re_mod, r_fe, r_re):
    if r_fe + r_re >= 1.0:
        raise ValueError("r_fe + r_re must be < 1 to leave room for residual variance")
    v_fe, v_re = variance_components(spec, X, re_mod)
    if v_re <= 0:
        raise ValueError("ranef variance is zero; cannot rescale")
    c = (v_fe / v_re) * (r_re / r_fe)
    new_ranef = [RanefSpec(re_formula=r.re_formula, group_var=r.group_var,
                           G=np.asarray(r.G) * c, membership=r.membership,
                           n_groups=r.n_groups, n_per=r.n_per)
                 for r in spec.ranef]
    rt = r_fe + r_re
    v_resid = (1.0 - rt) / rt * (v_fe + v_re * c)
    return SimSpec(n_obs=spec.n_obs, response=spec.response,
                   fe_formula=spec.fe_formula, beta=spec.beta,
                   ranef=new_ranef, resid_var=v_resid, cov_spec=spec.cov_spec)


class MixedModelSim:

    def __init__(self, spec, rng=None):
        self.spec = spec
        self.rng = np.random.default_rng() if rng is None else rng
        self.df = build_frame(spec, self.rng)
        self.X, self.re_mod = build_design(spec, self.df)
        self.Z = self.re_mod.Z
        self.theta_true = spec_to_theta(spec)
        self.beta_true = np.asarray(spec.beta).reshape(-1)
        self.eta_fe = self.X.dot(self.beta_true).reshape(-1)
        self.formula = to_lme4_formula(spec)
        self.re_mod.update_gcov(self.theta_true)

    def draw_ranefs(self, exact=False, dist=None):
        return draw_ranefs(self.spec, self.rng, exact=exact, dist=dist)

    def draw_residuals(self, exact=False, dist=None):
        return draw_residuals(self.spec, self.rng, exact=exact, dist=dist)

    def draw(self, exact_ranefs=False, exact_resids=False,
             ranef_dist=None, resid_dist=None):
        u = self.draw_ranefs(exact=exact_ranefs, dist=ranef_dist)
        zu = np.asarray(self.Z.dot(u)).reshape(-1)
        eps = self.draw_residuals(exact=exact_resids, dist=resid_dist)
        y = self.eta_fe + zu + eps
        return y, u

    def variance_components(self):
        return variance_components(self.spec, self.X, self.re_mod)

    def rescaled(self, r_fe, r_re):
        new_spec = rescale_to_var_ratios(self.spec, self.X, self.re_mod, r_fe, r_re)
        return MixedModelSim(new_spec, self.rng)

    def to_lmm(self, y):
        df = self.df.copy()
        df[self.spec.response] = np.asarray(y).reshape(-1)
        return LMM2(self.formula, df)


def fit_simulation(model, theta_init=None, reml=True, method='l-bfgs-b', opt_kws=None):
    re_mod = model.mme.re_mod
    if theta_init is None:
        theta_init = re_mod.theta
    eta_init = re_mod.reparam.fwd(theta_init)
    opt_kws = {} if opt_kws is None else opt_kws
    opt = sp.optimize.minimize(model.loglike_reparam, eta_init, args=(reml,),
                               jac=model.gradient_reparam, method=method, options=opt_kws)
    theta_hat = re_mod.reparam.rvs(opt.x)
    return theta_hat, opt
