import time
import numpy as np

from pystatsm.pylmm.sim_lmm2 import SimSpec, RanefSpec, CovariateSpec, MixedModelSim, fit_simulation


def _fmt_secs(s):
    if s < 1e-3: return f"{s*1e6:6.1f} us"
    if s < 1.0:  return f"{s*1e3:6.1f} ms"
    return f"{s:6.3f}  s"


def _summarize(theta_hat, theta_true, names):
    bias = theta_hat.mean(axis=0) - theta_true
    rmse = np.sqrt(np.mean((theta_hat - theta_true) ** 2, axis=0))
    se = theta_hat.std(axis=0, ddof=1) / np.sqrt(theta_hat.shape[0])
    print(f"  {'param':<10} {'true':>10} {'mean(hat)':>10} {'bias':>10} {'rmse':>10} {'mc_se':>10}")
    for i, name in enumerate(names):
        print(f"  {name:<10} {theta_true[i]:10.4f} {theta_hat[:, i].mean():10.4f} "
              f"{bias[i]:10.4f} {rmse[i]:10.4f} {se[i]:10.4f}")


def _basic_spec(n_groups, n_per=10):
    return SimSpec(
        n_obs=n_groups * n_per,
        response='y',
        fe_formula='1 + x1',
        beta=np.array([0.5, 1.0]),
        ranef=[RanefSpec(re_formula='1 + x1', group_var='g',
                         G=np.array([[1.0, 0.2], [0.2, 0.5]]),
                         n_groups=n_groups, n_per=n_per)],
        resid_var=0.5,
        cov_spec=CovariateSpec(cont_vars=['x1'], mean=np.array([0.0]),
                               cov=np.array([[1.0]])),
    )


def _intercept_only_spec(n_groups, n_per=10):
    return SimSpec(
        n_obs=n_groups * n_per,
        response='y',
        fe_formula='1 + x1',
        beta=np.array([0.0, 1.0]),
        ranef=[RanefSpec(re_formula='1', group_var='g',
                         G=np.array([[1.5]]),
                         n_groups=n_groups, n_per=n_per)],
        resid_var=0.5,
        cov_spec=CovariateSpec(cont_vars=['x1'], mean=np.array([0.0]),
                               cov=np.array([[1.0]])),
    )


def run_replications(spec, n_reps=200, reml=True, seed=0):
    """Build a fresh sim per replicate (fresh covariates AND fresh u, eps);
    fit; collect theta_hat. Returns (theta_hat, fit_times, fail_count)."""
    base_rng = np.random.default_rng(seed)
    n_pars = len(MixedModelSim(spec, base_rng).theta_true)
    theta_hat = np.zeros((n_reps, n_pars))
    times = np.zeros(n_reps)
    fails = 0
    for r in range(n_reps):
        rng = np.random.default_rng(int(base_rng.integers(0, 2**31 - 1)))
        sim = MixedModelSim(spec, rng)
        y, _ = sim.draw()
        t0 = time.perf_counter()
        try:
            th, opt = fit_simulation(sim.to_lmm(y), reml=reml)
            if not opt.success:
                fails += 1
            theta_hat[r] = th
        except Exception:
            fails += 1
            theta_hat[r] = np.nan
        times[r] = time.perf_counter() - t0
    return theta_hat, times, fails


def study_recovery_vs_n_groups(n_reps=30):
    """REML, balanced 1+x|g. Default settings keep total wall time small;
    raise n_reps and/or extend the grid for tighter MC SE."""
    print(f"\n=== Recovery vs n_groups (REML, balanced 1+x|g, n_reps={n_reps}) ===")
    names = ['G[0,0]', 'G[1,0]', 'G[1,1]', 'sigma2']
    for ng in [20, 50, 100]:
        spec = _basic_spec(ng)
        theta_hat, times, fails = run_replications(spec, n_reps=n_reps, seed=ng)
        valid = ~np.isnan(theta_hat).any(axis=1)
        print(f"\nn_groups={ng}, n_obs={spec.n_obs}, "
              f"failures={fails}, mean_fit_time={_fmt_secs(times.mean())}")
        _summarize(theta_hat[valid], MixedModelSim(spec).theta_true, names)


def study_recovery_intercept_only(n_reps=50):
    print(f"\n=== Recovery for random-intercept model (REML, n_reps={n_reps}) ===")
    spec = _intercept_only_spec(n_groups=100)
    names = ['G[0,0]', 'sigma2']
    theta_hat, times, fails = run_replications(spec, n_reps=n_reps, seed=1)
    valid = ~np.isnan(theta_hat).any(axis=1)
    print(f"\nn_groups=100, n_obs={spec.n_obs}, "
          f"failures={fails}, mean_fit_time={_fmt_secs(times.mean())}")
    _summarize(theta_hat[valid], MixedModelSim(spec).theta_true, names)


def study_reml_vs_ml(n_reps=30):
    print(f"\n=== REML vs ML on the same simulations (n_groups=50, n_reps={n_reps}) ===")
    spec = _basic_spec(n_groups=50)
    names = ['G[0,0]', 'G[1,0]', 'G[1,1]', 'sigma2']
    for reml in (True, False):
        theta_hat, times, fails = run_replications(spec, n_reps=n_reps, reml=reml, seed=99)
        valid = ~np.isnan(theta_hat).any(axis=1)
        label = 'REML' if reml else 'ML'
        print(f"\n{label}: failures={fails}, mean_fit_time={_fmt_secs(times.mean())}")
        _summarize(theta_hat[valid], MixedModelSim(spec).theta_true, names)


def main():
    t_start = time.perf_counter()
    study_recovery_vs_n_groups()
    study_recovery_intercept_only()
    study_reml_vs_ml()
    print(f"\ntotal wall time: {_fmt_secs(time.perf_counter() - t_start)}")


if __name__ == '__main__':
    main()
