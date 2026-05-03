"""LMM2 gradient/loglike scaling benchmark.

Sweeps configurations along three axes — number of levels per factor, number of
variates per term, and factor structure (single, crossed, nested) — and reports
wall time alongside structural sparsity (nnz of Z, C, L). The multipliers
ms/nnz(L) and ms/n_pars expose whether we're scaling as theory predicts.

Theory floor for the dominant cost: one solve_L on an n_ranef × n_ranef sparse
RHS is ≈ nnz(L) × n_ranef FLOPs. Wall_time / (nnz(L) × n_ranef) should be
roughly constant — a "GFLOPS"-equivalent we can compare across configs.

Run: PYTHONPATH=<repo> python lmm2_scaling.py
"""
import time
import numpy as np
import pandas as pd
import scipy as sp
import scipy.sparse

from pystatsm.pylmm.sim_lmm2 import (
    SimSpec, RanefSpec, CovariateSpec, MixedModelSim,
    Grouping, Nested, build_groupings,
)
from pystatsm.pylmm.re_mod import LMM2
from pystatsm.utilities.random import r_lkj


def _re_formula(n_v):
    """'1' for intercept-only or '1 + xa + xb + ...' for n_v > 1."""
    return '1' if n_v == 1 else '1 + ' + ' + '.join(f'x{i+3}' for i in range(n_v - 1))


def _make_G(n_v, rng):
    return r_lkj(eta=1, dim=n_v, rng=rng).squeeze() if n_v > 1 else np.array([[1.0]])


def build_single(n_obs, n_levels, n_v, seed=0):
    """One grouping factor — block-diagonal C, no fill-in baseline."""
    rng = np.random.default_rng(seed)
    Gid = _make_G(n_v, rng)
    g = build_groupings(n_obs, Grouping('id1', n_levels=n_levels, cycle='tile'))
    cv = ['x1', 'x2', 'x3'] + [f'x{i+3}' for i in range(max(0, n_v - 1))]
    spec = SimSpec(
        n_obs=n_obs, response='y', fe_formula='1 + x1 + x2 + x3',
        beta=np.array([0.0, 0.5, -0.3, 1.0]),
        ranef=[RanefSpec(_re_formula(n_v), 'id1', G=Gid, membership=g['id1'])],
        resid_var=0.5,
        cov_spec=CovariateSpec(cv, np.zeros(len(cv)), np.eye(len(cv))))
    sim = MixedModelSim(spec, rng)
    df = sim.df; df['y'], _ = sim.draw()
    return LMM2(f'y~1+x1+x2+x3+({_re_formula(n_v)}|id1)', data=df), sim


def build_crossed2(n_obs, n_l1, n_l2, n_v1, n_v2, seed=0):
    """Two crossed factors (id1 × id2) — introduces fill-in across the
    bipartite incidence structure."""
    rng = np.random.default_rng(seed)
    G1, G2 = _make_G(n_v1, rng), _make_G(n_v2, rng)
    g = build_groupings(n_obs,
                         Grouping('id1', n_levels=n_l1, cycle='tile'),
                         Grouping('id2', n_levels=n_l2, cycle='repeat'))
    n_v = max(n_v1, n_v2)
    cv = ['x1', 'x2', 'x3'] + [f'x{i+3}' for i in range(max(0, n_v - 1))]
    spec = SimSpec(
        n_obs=n_obs, response='y', fe_formula='1 + x1 + x2 + x3',
        beta=np.array([0.0, 0.5, -0.3, 1.0]),
        ranef=[
            RanefSpec(_re_formula(n_v1), 'id1', G=G1, membership=g['id1']),
            RanefSpec(_re_formula(n_v2), 'id2', G=G2, membership=g['id2']),
        ],
        resid_var=0.5,
        cov_spec=CovariateSpec(cv, np.zeros(len(cv)), np.eye(len(cv))))
    sim = MixedModelSim(spec, rng)
    df = sim.df; df['y'], _ = sim.draw()
    formula = (f'y~1+x1+x2+x3+({_re_formula(n_v1)}|id1)+({_re_formula(n_v2)}|id2)')
    return LMM2(formula, data=df), sim


def build_realistic(scale, seed=0):
    """User's three-term realistic spec, scaled. scale=1 → user's beefy config."""
    rng = np.random.default_rng(seed)
    n_obs = int(60000 * scale)
    n_l1 = max(50, int(1500 * scale))
    n_l3 = max(10, int(600 * scale))
    G1 = _make_G(3, rng); G2 = _make_G(2, rng); G3 = np.array([[1.0]])
    g = build_groupings(n_obs,
                         Grouping('id1', n_levels=n_l1, cycle='tile'),
                         Grouping('id3', n_levels=n_l3, cycle='repeat'),
                         Nested('id2', parent='id3', n_per_parent=5))
    spec = SimSpec.from_formula(
        'y ~ 1 + x1 + x2 + x3 + (1 + x3 + x4 | id1) + (1 + x5 | id2) + (1 | id3)',
        n_obs=n_obs, beta=np.array([0.0, 0.5, -0.3, 1.0]),
        ranef_G={'id1': G1, 'id2': G2, 'id3': G3},
        resid_var=0.5, groupings=g,
        cov_spec=CovariateSpec(['x1', 'x2', 'x3', 'x4', 'x5'],
                                np.zeros(5), np.eye(5)))
    sim = MixedModelSim(spec, rng)
    df = sim.df; df['y'], _ = sim.draw()
    return LMM2('y~1+x1+x2+x3+(1+x3+x4|id1)+(1+x5|id2)+(1|id3)', data=df), sim


def collect_structural(model):
    """Sparsity summary needed to interpret timings."""
    mme = model.mme
    Z = mme.Z; ZtRZ = mme.ZtRZ; G = mme.G
    C = ZtRZ + G
    fac = mme.chol_fac
    fac.cholesky_inplace(C)
    L = fac.L() if hasattr(fac, 'L') else None
    nnz_L = int(L.nnz) if L is not None else None
    return dict(
        n_obs=int(mme.n_obs), n_fixef=int(mme.n_fixef),
        n_pars=int(len(mme.re_mod.theta)),
        n_ranef=int(Z.shape[1]),
        nnz_Z=int(Z.nnz), nnz_C=int(C.nnz), nnz_L=nnz_L,
        fill_ratio=(nnz_L / C.nnz) if nnz_L else None,
    )


def time_call(fn, n_warm=1, n_runs=5):
    """Median-of-n_runs wall time in ms after n_warm warmup calls."""
    for _ in range(n_warm):
        fn()
    ts = []
    for _ in range(n_runs):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1000.0)
    return float(np.median(ts))


def benchmark_one(name, model, sim, reml=True, n_runs=5):
    theta = model.mme.re_mod.theta.copy()
    s = collect_structural(model)
    t_loglike = time_call(lambda: model.loglike(theta, reml=reml), n_runs=n_runs)
    t_grad = time_call(lambda: model.gradient(theta, reml=reml), n_runs=n_runs)
    return dict(
        name=name, **s,
        t_loglike_ms=t_loglike, t_gradient_ms=t_grad,
        ms_per_nnzL=t_grad / s['nnz_L'] * 1000.0 if s['nnz_L'] else None,
        ms_per_npar=t_grad / s['n_pars'],
        ratio_grad_loglike=t_grad / max(t_loglike, 1e-9),
        # Effective GFLOPs assuming dominant cost = solve_L on nnz_L × n_ranef
        eff_gflops=(s['nnz_L'] * s['n_ranef'] / max(t_grad / 1000.0, 1e-9) / 1e9
                    if s['nnz_L'] else None),
    )


CONFIGS = [
    # --- Single factor: vary n_levels at fixed n_v=3 (scaling in n_ranef) ---
    ('single_K=100_v=3',   lambda: build_single(n_obs=2000,  n_levels=100,   n_v=3)),
    ('single_K=500_v=3',   lambda: build_single(n_obs=5000,  n_levels=500,   n_v=3)),
    ('single_K=2000_v=3',  lambda: build_single(n_obs=20000, n_levels=2000,  n_v=3)),
    ('single_K=5000_v=3',  lambda: build_single(n_obs=50000, n_levels=5000,  n_v=3)),
    # --- Single factor: vary n_levels at fixed n_v=3 (scaling in n_ranef) ---
    ('single_K=100_v=3_nper=30',   lambda: build_single(n_obs=6000,  n_levels=100,   n_v=3)),
    ('single_K=500_v=3_nper=30',   lambda: build_single(n_obs=15000,  n_levels=500,   n_v=3)),
    ('single_K=2000_v=3_nper=30',  lambda: build_single(n_obs=60000, n_levels=2000,  n_v=3)),
    ('single_K=5000_v=3_nper=30',  lambda: build_single(n_obs=150000, n_levels=5000,  n_v=3)),
    # --- Single factor: vary n_levels at fixed n_v=4 (scaling in n_ranef) ---
    ('single_K=100_v=4',   lambda: build_single(n_obs=2000,  n_levels=100,   n_v=4)),
    ('single_K=500_v=4',   lambda: build_single(n_obs=5000,  n_levels=500,   n_v=4)),
    ('single_K=2000_v=4',  lambda: build_single(n_obs=20000, n_levels=2000,  n_v=4)),
    ('single_K=5000_v=4',  lambda: build_single(n_obs=50000, n_levels=5000,  n_v=4)),
    # --- Single factor: vary n_v at fixed K=1000 (scaling in n_pars) ---
    ('single_K=1000_v=1',  lambda: build_single(n_obs=10000, n_levels=1000, n_v=1)),
    ('single_K=1000_v=2',  lambda: build_single(n_obs=10000, n_levels=1000, n_v=2)),
    ('single_K=1000_v=3',  lambda: build_single(n_obs=10000, n_levels=1000, n_v=3)),
    ('single_K=1000_v=4',  lambda: build_single(n_obs=10000, n_levels=1000, n_v=4)),
    # --- Single factor: vary n_v at fixed K=1000 (scaling in n_pars) ---
    ('single_K=200_v=1',  lambda: build_single(n_obs=20000, n_levels=200, n_v=1)),
    ('single_K=200_v=2',  lambda: build_single(n_obs=20000, n_levels=200, n_v=2)),
    ('single_K=200_v=3',  lambda: build_single(n_obs=20000, n_levels=200, n_v=3)),
    ('single_K=200_v=4',  lambda: build_single(n_obs=20000, n_levels=200, n_v=4)),
    # --- Two crossed: vary one factor's K (introduces fill-in) ---
    ('crossed_1k_x_100',   lambda: build_crossed2(n_obs=10000, n_l1=1000, n_l2=100, n_v1=2, n_v2=2)),
    ('crossed_1k_x_500',   lambda: build_crossed2(n_obs=10000, n_l1=1000, n_l2=500, n_v1=2, n_v2=2)),
    ('crossed_2k_x_500',   lambda: build_crossed2(n_obs=20000, n_l1=2000, n_l2=500, n_v1=2, n_v2=2)),
    ('crossed_5k_x_1k',    lambda: build_crossed2(n_obs=50000, n_l1=5000, n_l2=1000, n_v1=2, n_v2=2)),
    # --- Realistic three-term, scaled ---
    ('realistic_0.25x',    lambda: build_realistic(scale=0.25)),
    ('realistic_0.5x',     lambda: build_realistic(scale=0.5)),
    ('realistic_1x',       lambda: build_realistic(scale=1.0)),
    ('realistic_2x',       lambda: build_realistic(scale=2.0)),
]



rows = []
for name, builder in CONFIGS:
    print(f'  {name}...', flush=True)
    try:
        model, sim = builder()
        rows.append(benchmark_one(name, model, sim, reml=True, n_runs=5))
    except Exception as e:
        print(f'    FAILED: {type(e).__name__}: {e}')
df = pd.DataFrame(rows)
pd.options.display.float_format = '{:.3f}'.format
pd.options.display.width = 220
pd.options.display.max_columns = 20
print()
cols = ['name', 'n_obs', 'n_pars', 'n_ranef', 'nnz_C', 'nnz_L', 'fill_ratio',
        't_loglike_ms', 't_gradient_ms', 'ms_per_nnzL', 'ratio_grad_loglike',
        'eff_gflops']
print(df[cols].to_string(index=False))


