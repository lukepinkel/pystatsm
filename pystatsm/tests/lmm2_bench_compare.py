import time
import numpy as np
from pystatsm.pylmm import re_mod as new_mod
from pystatsm.pylmm import _re_mod_orig as orig_mod
from pystatsm.pylmm.sim_lmm2 import (SimSpec, RanefSpec, CovariateSpec, MixedModelSim,)


def _fmt(s):#Nah
    if s < 1e-6: return f"{s*1e9:8.1f} ns"
    if s < 1e-3: return f"{s*1e6:8.1f} us"
    if s < 1.0:  return f"{s*1e3:8.1f} ms"
    return f"{s:8.3f}  s"


def _time(fn, n_reps):#Nah
    fn()
    t = np.empty(n_reps)
    for i in range(n_reps):
        t0 = time.perf_counter_ns()
        fn()
        t[i] = time.perf_counter_ns() - t0
    return float(np.median(t)) * 1e-9


def _time_interleaved(fn_a, fn_b, n_reps):#Nah
    """Alternate calls so cache/JIT state is shared. Avoids the bias you get
    from running fn_a 30 times in a row (cache hot for fn_a) and then fn_b 30
    times (cache cold for fn_b's data)."""
    fn_a(); fn_b()
    ta = np.empty(n_reps)
    tb = np.empty(n_reps)
    for i in range(n_reps):
        t0 = time.perf_counter_ns(); fn_a(); ta[i] = time.perf_counter_ns() - t0
        t0 = time.perf_counter_ns(); fn_b(); tb[i] = time.perf_counter_ns() - t0
    return float(np.median(ta)) * 1e-9, float(np.median(tb)) * 1e-9


def _make_spec(n_groups, n_per, n_revars):
    cov_vars = [f"x{i+1}" for i in range(max(n_revars - 1, 1))]
    fe_form = "1" + "".join([f" + {v}" for v in cov_vars])
    re_form = "1" + "".join([f" + {v}" for v in cov_vars[: n_revars - 1]])
    G = np.eye(n_revars) * 0.5 + 0.1#Yikes
    G = (G + G.T) * 0.5#Yikes
    np.fill_diagonal(G, np.diag(G) + 0.5)#Yikes
    n_cov = len(cov_vars)
    return SimSpec(
        n_obs=n_groups * n_per,
        response='y',
        fe_formula=fe_form,
        beta=np.r_[0.5, np.full(n_cov, 0.5)],
        ranef=[RanefSpec(re_formula=re_form, group_var='g',
                         G=G, n_groups=n_groups, n_per=n_per)],
        resid_var=0.5,
        cov_spec=CovariateSpec(cont_vars=cov_vars,
                               mean=np.zeros(n_cov),
                               cov=np.eye(n_cov)),
    )#Yikes for this whole line


def _build_pair(spec, seed=0):
    """Build matched (orig, new) LMM2 instances on identical data."""
    rng = np.random.default_rng(seed)
    sim = MixedModelSim(spec, rng)
    y, _ = sim.draw()
    df = sim.df.copy()
    df[spec.response] = y
    model_orig = orig_mod.LMM2(sim.formula, df)
    model_new = new_mod.LMM2(sim.formula, df)
    return sim, model_orig, model_new


def _n_reps_for(n_obs):
    if n_obs < 1000:  return 200, 50
    if n_obs < 5000:  return 100, 25
    return 50, 15


def compare_one_case(n_groups, n_per, n_revars, seed=0):
    spec = _make_spec(n_groups, n_per, n_revars)
    sim, m_o, m_n = _build_pair(spec, seed)
    theta = sim.theta_true.copy()
    eta = m_n.mme.re_mod.reparam.fwd(theta)
    rf, rs = _n_reps_for(spec.n_obs)
    rows = []
    for label, fn_o, fn_n in [
        ("loglike (REML)",
         lambda: m_o.loglike(theta, True),  lambda: m_n.loglike(theta, True)),
        ("loglike (ML)",
         lambda: m_o.loglike(theta, False), lambda: m_n.loglike(theta, False)),
        ("gradient (REML)",
         lambda: m_o.gradient(theta, True), lambda: m_n.gradient(theta, True)),
        ("gradient (ML)",
         lambda: m_o.gradient(theta, False), lambda: m_n.gradient(theta, False)),
        ("loglike_reparam",
         lambda: m_o.loglike_reparam(eta, True),
         lambda: m_n.loglike_reparam(eta, True)),
        ("gradient_reparam",
         lambda: m_o.gradient_reparam(eta, True),
         lambda: m_n.gradient_reparam(eta, True)),
    ]:
        n = rs if "gradient" in label else rf
        t_o, t_n = _time_interleaved(fn_o, fn_n, n)
        rows.append((label, t_o, t_n, t_o / t_n))
    return rows


def correctness_consistency(theta, m_o, m_n):
    """Spot-check where the algorithms agree:
       - loglike (both REML & ML): identical algorithm
       - gradient REML: identical algorithm (only slicing changed)
       - gradient ML & gradient_reparam: original has bugs, expect mismatch
    """
    eta = m_n.mme.re_mod.reparam.fwd(theta)
    out = []
    out.append(("loglike (REML)",
                m_o.loglike(theta, True), m_n.loglike(theta, True)))
    out.append(("loglike (ML)",
                m_o.loglike(theta, False), m_n.loglike(theta, False)))
    out.append(("gradient (REML) max|diff|",
                None, float(np.max(np.abs(
                    m_o.gradient(theta, True) - m_n.gradient(theta, True))))))
    out.append(("gradient (ML) max|diff| (orig has known bug)",
                None, float(np.max(np.abs(
                    m_o.gradient(theta, False) - m_n.gradient(theta, False))))))
    out.append(("gradient_reparam max|diff| (orig has known bug)",
                None, float(np.max(np.abs(
                    m_o.gradient_reparam(eta, True) -
                    m_n.gradient_reparam(eta, True))))))
    return out



cases = [
    (50,  10, 2),
    (200, 10, 2),
    (200, 10, 3),
]
print("\n=== correctness consistency (small case) ===")
spec = _make_spec(50, 10, 2)
sim, m_o, m_n = _build_pair(spec)
for label, a, b in correctness_consistency(sim.theta_true, m_o, m_n):
    if a is None:
        print(f"  {label:<50} {b:.3e}")
    else:
        print(f"  {label:<50} orig={a:.6f} new={b:.6f} diff={abs(a-b):.3e}")

print("\n=== timing comparison (median; speedup = orig / new) ===")
header = f"\n  {'op':<22}  {'orig':>10}  {'new':>10}  {'speedup':>9}"
for ng, np_, nv in cases:
    spec = _make_spec(ng, np_, nv)
    n_pars = nv * (nv + 1) // 2 + 1
    print(f"\nn_groups={ng}, n_per={np_}, n_revars={nv}, "
          f"n_obs={spec.n_obs}, n_pars={n_pars}:")
    print(header)
    for label, t_o, t_n, ratio in compare_one_case(ng, np_, nv):
        print(f"  {label:<22}  {_fmt(t_o):>10}  {_fmt(t_n):>10}  {ratio:>8.2f}x")

