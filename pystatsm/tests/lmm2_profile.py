import cProfile
import io
import pstats
import time
import numpy as np
import scipy as sp

from pystatsm.pystatsm.pylmm.sim_lmm2 import (
    SimSpec, CovariateSpec, MixedModelSim, fit_simulation,
    Grouping, Nested, build_groupings,
)
from pystatsm.pystatsm.utilities.random import r_lkj


def make_realistic_sim(n_obs=30000, n_id1=5000, n_id3=1000, id2_per_id3=10,
                      seed=123):
    rng = np.random.default_rng(seed)
    G_id1 = r_lkj(eta=1, dim=3, rng=rng).squeeze()
    G_id2 = r_lkj(eta=1, dim=2, rng=rng).squeeze()
    G_id3 = np.array([[1.0]])
    groupings = build_groupings(
        n_obs,
        Grouping('id1', n_levels=n_id1, cycle='tile'),
        Grouping('id3', n_levels=n_id3, cycle='repeat'),
        Nested('id2', parent='id3', n_per_parent=id2_per_id3),
    )
    spec = SimSpec.from_formula(
        'y ~ 1 + x1 + x2 + x3 + (1 + x3 + x4 | id1) + (1 + x5 | id2) + (1 | id3)',
        n_obs=n_obs,
        beta=np.array([0.0, 0.5, -0.3, 1.0]),
        ranef_G={'id1': G_id1, 'id2': G_id2, 'id3': G_id3},
        resid_var=0.5,
        groupings=groupings,
        cov_spec=CovariateSpec(['x1', 'x2', 'x3', 'x4', 'x5'],
                               np.zeros(5), np.eye(5)),
    )
    return MixedModelSim(spec, rng)


def _show(prof, sort_by, n=20, label=''):
    buf = io.StringIO()
    pstats.Stats(prof, stream=buf).sort_stats(sort_by).print_stats(n)
    print(f"\n=== {label} (top {n} by {sort_by}) ===")
    print(buf.getvalue())


def profile_gradient_calls(sim, n_calls=50):
    model = sim.to_lmm(sim.draw()[0])
    theta = sim.theta_true.copy()
    # Warm up so first-call costs don't dominate.
    model.gradient(theta, reml=True)
    prof = cProfile.Profile()
    prof.enable()
    for _ in range(n_calls):
        model.gradient(theta, reml=True)
    prof.disable()
    return prof


def profile_full_fit(sim):
    model = sim.to_lmm(sim.draw()[0])
    fit_simulation(model, reml=True)  # warm-up: triggers chol_fac construction
    model = sim.to_lmm(sim.draw()[0])
    prof = cProfile.Profile()
    prof.enable()
    fit_simulation(model, reml=True)
    prof.disable()
    return prof

mult=None
M_sum=None
n_obs=4500
n_id1=1500
n_id3=150
id2_per_id3=5
n_grad_calls=10
n=50000;n_id1=1000;n_id3=50;id2_per_id3=10

t0 = time.perf_counter()
sim = make_realistic_sim(n_obs=n_obs, n_id1=n_id1, n_id3=n_id3, id2_per_id3=id2_per_id3)
print(f"  built in {time.perf_counter() - t0:.2f}s; "
  f"Z {sim.Z.shape}, n_pars {sim.theta_true.size}")

t0 = time.perf_counter()
prof_grad = profile_gradient_calls(sim, n_calls=n_grad_calls)
print(f"\n{n_grad_calls} gradient calls: {time.perf_counter() - t0:.2f}s "
  f"({(time.perf_counter() - t0) * 1000 / n_grad_calls:.1f} ms/call)")


_show(prof_grad, 'cumulative', n=300, label='gradient hot path (cumulative)')
_show(prof_grad, 'tottime',     n=300, label='gradient hot path (self-time)')

# t0 = time.perf_counter()
# prof_fit = profile_full_fit(sim)
# print(f"\nfull fit: {time.perf_counter() - t0:.2f}s")
# _show(prof_fit, 'cumulative', n=15, label='full fit (cumulative)')

