import numpy as np
import scipy as sp


class GPA:
    """Gradient projection algorithm. Works for ortho or oblique via rot
    primitives. Alpha propagates across outer iterations (Jennrich-style):
    doubles on success, halves during Armijo backtrack. Strict reject on LS
    exhaustion keeps the oblique case stable (where column-normalization can
    amplify an over-ambitious step)."""

    def __init__(self, crit, rot, alpha=1.0, tol=1e-6, n_iters=1000, ls_iters=30):
        self.crit = crit
        self.rot = rot
        self.alpha0 = alpha
        self.tol = tol
        self.n_iters = n_iters
        self.ls_iters = ls_iters

    def solve(self, A, T0=None):
        crit, rot = self.crit, self.rot
        m = A.shape[1]
        T = np.eye(m) if T0 is None else T0
        L = rot.rotated_loadings(A, T)
        f = crit.Q(L)
        G = rot.grad(A, T, crit.dQ(L))
        hist = []
        s = np.inf
        alpha = self.alpha0
        for _ in range(self.n_iters):
            Gp = rot.constraint_project(T, G)
            s = np.linalg.norm(Gp)
            hist.append((f, s))
            if s < self.tol:
                break
            alpha *= 2.0
            accepted = False
            for _ in range(self.ls_iters):
                Tt = rot.constraint_retract(T - alpha * Gp)
                Lt = rot.rotated_loadings(A, Tt)
                ft = crit.Q(Lt)
                if ft < f - 0.5 * s * s * alpha:
                    accepted = True
                    break
                alpha *= 0.5
            if not accepted:
                break
            T, L, f = Tt, Lt, ft
            G = rot.grad(A, T, crit.dQ(L))
        return T, {"f": f, "grad_norm": s, "n_iter": len(hist), "trace": hist}


class CayleySolver:
    """Unconstrained minimization of Q in the unconstrained_to_rotation
    coordinates. Only defined for rotators that implement
    unconstrained_to_rotation/d_rotation (currently Ortho). T0 is a
    rotation-matrix initial point (converted to Cayley coords internally)."""

    def __init__(self, crit, rot, method='trust-constr', **opt_kws):
        self.crit = crit
        self.rot = rot
        self.method = method
        self.opt_kws = opt_kws

    def _fg(self, theta, A):
        T = self.rot.unconstrained_to_rotation(theta)
        L = self.rot.rotated_loadings(A, T)
        f = self.crit.Q(L)
        g_T = self.rot.grad(A, T, self.crit.dQ(L))
        return f, self.rot.d_rotation(theta, g_T)

    def solve(self, A, T0=None):
        if T0 is None:
            theta0 = np.zeros(self.rot.constraint_dim)
        else:
            theta0 = self.rot.rotation_to_unconstrained(T0)
        opt = sp.optimize.minimize(lambda t: self._fg(t, A)[0], theta0,
                                   jac=lambda t: self._fg(t, A)[1],
                                   method=self.method, **self.opt_kws)
        T = self.rot.unconstrained_to_rotation(opt.x)
        return T, {"f": opt.fun, "opt": opt, "theta": opt.x}


# Backward-compat function form: build solver, call solve. Existing callers
# of solvers.gpa(crit, rot, A, ...) or solvers.cayley_solve(...) continue to work.

def gpa(crit, rot, A, T0=None, **kw):
    return GPA(crit, rot, **kw).solve(A, T0=T0)


def cayley_solve(crit, rot, A, T0=None, **kw):
    return CayleySolver(crit, rot, **kw).solve(A, T0=T0)
