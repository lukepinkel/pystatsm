"""Benchmark harness for pyfa2. Pyfa2Benchmark.run() times each primitive,
likelihood op, and end-to-end pipeline across a list of (p, m) cases and
prints a tab-separated table."""

import time
import numpy as np

from pystatsm.pyfa2.criterion import GCFCriterion
from pystatsm.pyfa2.rotation import OrthoRotation, ObliqueRotation
from pystatsm.pyfa2.solvers import GPA, CayleySolver
from pystatsm.pyfa2.likelihood import MLEstimator
from pystatsm.pyfa2.identification import RotationIdentification, CanonicalIdentification
from pystatsm.pyfa2.inference import (param_cov, sandwich_cov,
                                      empirical_cov_vech_S, se_from_cov)


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


class Pyfa2Benchmark:
    """Times pyfa2 primitives and pipelines across (p, m) cases. Override
    `cases`, `n_obs`, or `n_reps_for(case)` to adjust scope."""

    DEFAULT_CASES = [(20, 3), (50, 5), (80, 8)]

    def __init__(self, cases=None, n_obs=1000, seed=0):
        self.cases = self.DEFAULT_CASES if cases is None else cases
        self.n_obs = n_obs
        self.seed = seed

    def n_reps_for(self, p, m):
        # Heuristic: fast ops vs slow ops, scaled by problem size.
        if p * m < 100:   return 500, 50
        if p * m < 500:   return 100, 20
        return 30, 5

    def setup(self, p, m):
        rng = np.random.default_rng(self.seed)
        L_true = rng.standard_normal((p, m)) * 0.5
        psi_true = 0.4 + rng.random(p)
        Sigma = L_true @ L_true.T + np.diag(psi_true)
        X = rng.multivariate_normal(np.zeros(p), Sigma, size=self.n_obs)
        S = np.cov(X, rowvar=False, bias=True)
        return X, S, L_true, psi_true

    # ---------- per-section timing ----------

    def bench_primitives(self, p, m):
        X, S, L_true, psi_true = self.setup(p, m)
        rng = np.random.default_rng(self.seed)
        A = rng.standard_normal((p, m))
        crit = GCFCriterion('varimax', p, m)
        rot_o, rot_b = OrthoRotation(m), ObliqueRotation(m)
        T_o = rot_o.constraint_retract(rng.standard_normal((m, m)))
        T_b = rot_b.constraint_retract(rng.standard_normal((m, m)))
        L_o = rot_o.rotated_loadings(A, T_o)
        L_b = rot_b.rotated_loadings(A, T_b)
        Phi_o = rot_o.implied_corr(T_o)
        Phi_b = rot_b.implied_corr(T_b)
        dQL = crit.dQ(L_o)
        rf, rs = self.n_reps_for(p, m)
        return [
            ("crit.Q",                 _time(lambda: crit.Q(L_o), rf)),
            ("crit.dQ",                _time(lambda: crit.dQ(L_o), rf)),
            ("crit.d2Q",               _time(lambda: crit.d2Q(L_o), rs)),
            ("ortho.grad",             _time(lambda: rot_o.grad(A, T_o, dQL), rf)),
            ("ortho.constraint_retract", _time(lambda: rot_o.constraint_retract(T_o + 0.01), rf)),
            ("ortho.constraint",       _time(lambda: rot_o.constraint(L_o, Phi_o, crit), rf)),
            ("ortho.d_constraint",     _time(lambda: rot_o.d_constraint(L_o, Phi_o, crit), rs)),
            ("obl.grad",               _time(lambda: rot_b.grad(A, T_b, dQL), rf)),
            ("obl.d_constraint",       _time(lambda: rot_b.d_constraint(L_b, Phi_b, crit), rs)),
        ]

    def bench_likelihood(self, p, m):
        X, S, L_true, psi_true = self.setup(p, m)
        est = MLEstimator(S, m)
        theta = est.layout.pack(L_true, np.eye(m), psi_true)
        rf, rs = self.n_reps_for(p, m)
        return [
            ("loglike_psi",       _time(lambda: est.loglike_psi(psi_true), rf)),
            ("grad_psi",          _time(lambda: est.grad_psi(psi_true), rf)),
            ("hess_psi",          _time(lambda: est.hess_psi(psi_true), rs)),
            ("loadings_from_psi", _time(lambda: est.loadings_from_psi(psi_true), rf)),
            ("F",                 _time(lambda: est.F(theta), rf)),
            ("grad",              _time(lambda: est.grad(theta), rf)),
            ("dsigma",            _time(lambda: est.dsigma(theta), rs)),
            ("hessian",           _time(lambda: est.hessian(theta), rs)),
            ("score_jacobian",    _time(lambda: est.score_jacobian(theta), rs)),
            ("meat (analytical)", _time(lambda: est.meat(theta, n_obs=self.n_obs), rs)),
            ("empirical_cov_vech_S", _time(lambda: empirical_cov_vech_S(X), rs)),
        ]

    def bench_endtoend(self, p, m, n_runs=3):
        X, S, L_true, psi_true = self.setup(p, m)
        crit = GCFCriterion('varimax', p, m)
        n = self.n_obs

        def fit_canon():
            est = MLEstimator(S, m)
            out = est.fit_psi()
            ident = CanonicalIdentification(p, m)
            f = ident.fit(out['Lambda'], Psi=out['psi'])
            theta = est.layout.pack(f['L'], f['Phi'], out['psi'])
            ws = est.workspace(theta)
            H = ws.hessian()
            C = ident.d_constraint(theta)
            return param_cov(H, C, ident.free_mask(), n)

        def fit_rot(rot):
            est = MLEstimator(S, m)
            out = est.fit_psi()
            ident = RotationIdentification(rot, crit)
            f = ident.fit(out['Lambda'])
            theta = est.layout.pack(f['L'], f['Phi'], out['psi'])
            ws = est.workspace(theta)
            H = ws.hessian()
            C = ident.d_constraint(theta)
            return param_cov(H, C, ident.free_mask(), n)

        def fit_sandwich():
            est = MLEstimator(S, m)
            out = est.fit_psi()
            ident = RotationIdentification(OrthoRotation(m), crit)
            f = ident.fit(out['Lambda'])
            theta = est.layout.pack(f['L'], f['Phi'], out['psi'])
            ws = est.workspace(theta)
            H = ws.hessian()
            C = ident.d_constraint(theta)
            V_S = empirical_cov_vech_S(X)
            J = ws.meat(V_S=V_S)
            return sandwich_cov(H, C, J, ident.free_mask())

        labels = [
            ('canon',            fit_canon),
            ('ortho/varimax',    lambda: fit_rot(OrthoRotation(m))),
            ('oblique/varimax',  lambda: fit_rot(ObliqueRotation(m))),
            ('sandwich (ortho)', fit_sandwich),
        ]
        out = {}
        for label, fn in labels:
            ts = []
            fn()
            for _ in range(n_runs):
                t0 = time.perf_counter_ns()
                fn()
                ts.append(time.perf_counter_ns() - t0)
            out[label] = float(np.median(ts)) * 1e-9
        return out

    # ---------- runner ----------

    def run(self, sections=('primitives', 'likelihood', 'endtoend')):
        cases_data = [(p, m) for p, m in self.cases]
        header = f"{'op':<32}" + "".join(f"  p={p:>3},m={m:>2}" for p, m in cases_data)

        if 'primitives' in sections:
            print("\n=== primitives ===")
            print(header)
            rows = [self.bench_primitives(p, m) for p, m in cases_data]
            self._print_aligned(rows)

        if 'likelihood' in sections:
            print("\n=== likelihood ===")
            print(header)
            rows = [self.bench_likelihood(p, m) for p, m in cases_data]
            self._print_aligned(rows)

        if 'endtoend' in sections:
            print("\n=== end-to-end pipelines (median, 3 runs) ===")
            for p, m in cases_data:
                times = self.bench_endtoend(p, m)
                print(f"p={p}, m={m}:")
                for k, t in times.items():
                    print(f"  {k:<22} {_fmt(t)}")

    @staticmethod
    def _print_aligned(rows):
        names = [r[0] for r in rows[0]]
        for i, name in enumerate(names):
            line = f"{name:<32}"
            for case_rows in rows:
                line += f"  {_fmt(case_rows[i][1]):>12}"
            print(line)


def main():
    Pyfa2Benchmark().run()


if __name__ == "__main__":
    main()
