import numpy as np
import scipy as sp

#TODO: Turn these into classes


def gpa(crit, rot, A, T0=None, alpha=1.0, tol=1e-6, n_iters=1000, ls_iters=30):
    m = A.shape[1]
    T = np.eye(m) if T0 is None else T0
    L = rot.rotated_loadings(A, T)
    f = crit.Q(L)
    G = rot.grad(A, T, crit.dQ(L))
    hist = []
    s = np.inf
    for _ in range(n_iters):
        Gp = rot.constraint_project(T, G)
        s = np.linalg.norm(Gp)
        hist.append((f, s))
        if s < tol:
            break
        alpha *= 2.0
        accepted = False
        for _ in range(ls_iters):
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


def cayley_solve(crit, rot, A, T0=None, method='trust-constr', **opt_kws):
    theta0 = np.zeros(rot.constraint_dim) if T0 is None else rot.rotation_to_unconstrained(T0)

    def fg(theta):
        T = rot.unconstrained_to_rotation(theta)
        L = rot.rotated_loadings(A, T)
        f = crit.Q(L)
        g_T = rot.grad(A, T, crit.dQ(L))
        return f, rot.d_rotation(theta, g_T)

    opt = sp.optimize.minimize(lambda t: fg(t)[0], theta0,
                               jac=lambda t: fg(t)[1],
                               method=method, **opt_kws)
    T = rot.unconstrained_to_rotation(opt.x)
    return T, {"f": opt.fun, "opt": opt, "theta": opt.x}
