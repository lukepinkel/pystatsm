"""In-depth profile of LMM2.loglike at realistic scale.

Runs three sweeps:
  1) Wall-clock per call, mean ± std over many trials.
  2) cProfile aggregate (cumulative + self-time).
  3) Direct timing of each primitive inside `_loglike` /
     `update_chol_diag` / `_chol_sparse_diag` so we know exactly where
     the wall-clock goes.

Run: PYTHONPATH=<repo> python pystatsm/tests/lmm2_profile_loglike.py
"""
import time
import cProfile
import pstats
import io
import numpy as np
import scipy as sp
import scipy.sparse

from ..pylmm.sim_lmm2 import (
    SimSpec, CovariateSpec, MixedModelSim,
    Grouping, Nested, build_groupings,
)
from ..pylmm.re_mod import LMM2
from ..utilities.random import r_lkj
from ..utilities.python_wrappers import cs_add_inplace


def build_realistic(scale=1.0, seed=123):
    rng = np.random.default_rng(seed)
    n_obs = int(60000 * scale)
    n_l1 = max(50, int(1500 * scale))
    n_l3 = max(10, int(600 * scale))
    G1 = r_lkj(eta=1, dim=3, rng=rng).squeeze()
    G2 = r_lkj(eta=1, dim=2, rng=rng).squeeze()
    G3 = np.array([[1.0]])
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
    return LMM2('y~1+x1+x2+x3+(1+x3+x4|id1)+(1+x5|id2)+(1|id3)', data=df)


def time_n(fn, n_warm=3, n_runs=50):
    for _ in range(n_warm):
        fn()
    ts = []
    for _ in range(n_runs):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1000.0)
    arr = np.asarray(ts)
    return arr.mean(), arr.std(), arr.min(), arr.max()


def primitive_timings(model, theta, n=30):
    """Time each primitive op inside `loglike` (which goes through
    `_loglike` → `update_chol_diag` → `_chol_sparse_diag`). Mimics the
    sequence exactly so warm-up effects on each call are realistic."""
    mme = model.mme
    fac = mme.chol_fac
    rows = []

    def bench(label, fn):
        for _ in range(3):
            fn()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        rows.append((label, (time.perf_counter() - t0) / n * 1000.0))

    bench('update_crossprods',
          lambda: mme.update_crossprods(theta))

    bench('update_gcov (Ginv into mme.G)',
          lambda: mme.re_mod.update_gcov(theta, inv=True, G=mme.G))

    # Need an up-to-date Ginv before the cs_add benchmark
    Ginv = mme.re_mod.update_gcov(theta, inv=True, G=mme.G)
    bench('cs_add_inplace (C = ZtRZ + Ginv)',
          lambda: cs_add_inplace(mme.ZtRZ, Ginv, mme.C))

    # Now factor C once before we benchmark sub-steps
    C = cs_add_inplace(mme.ZtRZ, Ginv, mme.C)
    bench('cholesky_inplace(C)',
          lambda: fac.cholesky_inplace(C))

    fac.cholesky_inplace(C)
    bench('chol_fac.L()',  lambda: fac.L())

    L11 = fac.L()
    bench('L11[p, p]  (diag extract w/ permutation)',
          lambda: L11[mme._p, mme._p])

    ZtRXy = mme.ZtRXy
    bench('apply_P(ZtRXy)',  lambda: fac.apply_P(ZtRXy))

    p_b = fac.apply_P(ZtRXy)
    bench('solve_L(applied) on dense (n_ranef, n_fixef+1)',
          lambda: fac.solve_L(p_b, False))

    s_b = fac.solve_L(p_b, False)
    bench('apply_Pt(solve_L output)',
          lambda: fac.apply_Pt(s_b))

    bench('whole _chol_sparse_diag(C)',
          lambda: mme._chol_sparse_diag(C))

    bench('whole update_chol_diag(theta)',
          lambda: mme.update_chol_diag(theta))

    bench('whole _loglike(theta, reml=True)',
          lambda: mme._loglike(theta, reml=True))

    return rows



    print('=== Build model (realistic 1x: n=60000, n_ranef=11100) ===')
    model = build_realistic(scale=1.0)
    theta = model.mme.re_mod.theta.copy()
    print(f'Z {model.mme.Z.shape}, n_pars {len(theta)}, '
          f'nnz(C) {model.mme.C.nnz}\n')

    # 1) Wall-clock summary
    print('=== Wall-clock (mean ± std, min/max over 50 runs) ===')
    m, s, lo, hi = time_n(lambda: model.loglike(theta, reml=True))
    print(f'loglike(reml=True):  mean {m:6.2f} ± {s:5.2f} ms  '
          f'(min {lo:.2f}, max {hi:.2f})')
    m, s, lo, hi = time_n(lambda: model.loglike(theta, reml=False))
    print(f'loglike(reml=False): mean {m:6.2f} ± {s:5.2f} ms  '
          f'(min {lo:.2f}, max {hi:.2f})\n')

    # 2) cProfile aggregate
    print('=== cProfile (30 calls, sorted) ===')
    N = 30
    for _ in range(3):
        model.loglike(theta, reml=True)
    prof = cProfile.Profile()
    prof.enable()
    for _ in range(N):
        model.loglike(theta, reml=True)
    prof.disable()

    buf = io.StringIO()
    pstats.Stats(prof, stream=buf).sort_stats('cumulative').print_stats(20)
    print('--- cumulative time ---')
    print(buf.getvalue())

    buf = io.StringIO()
    pstats.Stats(prof, stream=buf).sort_stats('tottime').print_stats(15)
    print('--- self time ---')
    print(buf.getvalue())

    # 3) Per-primitive timing
    print('=== Primitive breakdown (30 runs each) ===')
    rows = primitive_timings(model, theta, n=30)
    width = max(len(r[0]) for r in rows)
    for label, ms in rows:
        print(f'  {label:<{width}}  {ms:7.3f} ms')

