import time
import numpy as np

from pystatsm.pylmm.sim_lmm2 import SimSpec, RanefSpec, CovariateSpec, MixedModelSim, fit_simulation


def _fmt(s):
    if s < 1e-6: return f"{s*1e9:8.1f} ns"
    if s < 1e-3: return f"{s*1e6:8.1f} us"
    if s < 1.0:  return f"{s*1e3:8.1f} ms"
    return f"{s:8.3f}  s"


def _time(fn, n_reps):
    fn()
    t = np.empty(n_reps)
    for i in range(n_reps):
        t0 = time.perf_counter_ns()
        fn()
        t[i] = time.perf_counter_ns() - t0
    return float(np.median(t)) * 1e-9


def _make_spec(n_groups, n_per, n_revars):
    """A canonical spec with random intercept + (n_revars - 1) random slopes
    on covariates x1, x2, ..."""
    cov_vars = [f"x{i+1}" for i in range(max(n_revars - 1, 1))]
    fe_form = "1" + "".join([f" + {v}" for v in cov_vars])
    re_form = "1" + "".join([f" + {v}" for v in cov_vars[: n_revars - 1]])
    G = np.eye(n_revars) * 0.5 + np.full((n_revars, n_revars), 0.1)
    G = G + G.T
    G = G * 0.5  # keep PD
    np.fill_diagonal(G, np.diag(G) + 0.5)
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
    )


class LMM2Benchmark:

    DEFAULT_CASES = [
        # (n_groups, n_per, n_revars)
        (50,  10, 2),
        (200, 10, 2),
        (200, 10, 3),
        (500, 10, 2),
    ]

    def __init__(self, cases=None, seed=0):
        self.cases = self.DEFAULT_CASES if cases is None else cases
        self.seed = seed

    def n_reps_for(self, n_obs):
        if n_obs < 1000:  return 50, 10
        if n_obs < 5000:  return 20, 5
        return 10, 3

    def setup(self, n_groups, n_per, n_revars):
        spec = _make_spec(n_groups, n_per, n_revars)
        sim = MixedModelSim(spec, np.random.default_rng(self.seed))
        y, _ = sim.draw()
        model = sim.to_lmm(y)
        return sim, model

    def bench_objectives(self, n_groups, n_per, n_revars):
        sim, model = self.setup(n_groups, n_per, n_revars)
        theta = sim.theta_true.copy()
        eta = model.mme.re_mod.reparam.fwd(theta)
        rf, rs = self.n_reps_for(sim.spec.n_obs)
        return [
            ("loglike (REML)",        _time(lambda: model.loglike(theta, True), rf)),
            ("loglike (ML)",          _time(lambda: model.loglike(theta, False), rf)),
            ("gradient (REML)",       _time(lambda: model.gradient(theta, True), rs)),
            ("gradient (ML)",         _time(lambda: model.gradient(theta, False), rs)),
            ("loglike_reparam",       _time(lambda: model.loglike_reparam(eta, True), rf)),
            ("gradient_reparam",      _time(lambda: model.gradient_reparam(eta, True), rs)),
        ]

    def bench_fit(self, n_groups, n_per, n_revars, n_runs=3):
        sim, _ = self.setup(n_groups, n_per, n_revars)
        y, _ = sim.draw()
        # Warm-up
        fit_simulation(sim.to_lmm(y), reml=True)
        out = {}
        for label, reml in [("fit (REML)", True), ("fit (ML)", False)]:
            ts = []
            for _ in range(n_runs):
                model = sim.to_lmm(y)
                t0 = time.perf_counter_ns()
                _, opt = fit_simulation(model, reml=reml)
                ts.append(time.perf_counter_ns() - t0)
            out[label] = (float(np.median(ts)) * 1e-9, opt.nit)
        return out

    def run(self, sections=('objectives', 'fit')):
        cases = self.cases
        header = f"{'op':<24}" + "".join(
            f"  ng={ng:>3},np={np_:>2},nv={nv}" for ng, np_, nv in cases)

        if 'objectives' in sections:
            print("\n=== per-call timings (median) ===")
            print(header)
            rows = [self.bench_objectives(*c) for c in cases]
            self._print_aligned(rows)

        if 'fit' in sections:
            print("\n=== full fit (median of 3) ===")
            for c in cases:
                ng, np_, nv = c
                t = self.bench_fit(*c)
                print(f"\nng={ng}, np={np_}, nv={nv}, n_obs={ng*np_}, "
                      f"n_pars={nv*(nv+1)//2 + 1}:")
                for label, (sec, nit) in t.items():
                    print(f"  {label:<14} {_fmt(sec)}  ({nit} iters)")

    @staticmethod
    def _print_aligned(rows):
        names = [r[0] for r in rows[0]]
        for i, name in enumerate(names):
            line = f"{name:<24}"
            for case_rows in rows:
                line += f"  {_fmt(case_rows[i][1]):>16}"
            print(line)



LMM2Benchmark().run()

