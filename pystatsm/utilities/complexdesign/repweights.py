import numpy as np


def jackknife_multipliers(layout):
    n_grp = layout.n_grp
    mult = np.ones((n_grp, n_grp))
    rscales = np.zeros(n_grp)
    for h in range(layout.n_str):
        g0, g1 = layout.ind_str[h], layout.ind_str[h + 1]
        n_h = g1 - g0
        if n_h < 2:
            continue
        blk = np.full((n_h, n_h), n_h / (n_h - 1.0))
        blk[np.arange(n_h), np.arange(n_h)] = 0.0
        mult[g0:g1, g0:g1] = blk
        rscales[g0:g1] = layout.ssf[h] * ((n_h - 1.0) / n_h) ** 2
    return mult, rscales


def bootstrap_multipliers(layout, n_rep, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    mult = np.ones((n_rep, layout.n_grp))
    for h in range(layout.n_str):
        g0, g1 = layout.ind_str[h], layout.ind_str[h + 1]
        n_h = g1 - g0
        if n_h < 2:
            continue
        counts = rng.multinomial(n_h - 1, np.full(n_h, 1.0 / n_h), size=n_rep)
        mult[:, g0:g1] = counts * (n_h / (n_h - 1.0))
    return mult


def replicate_variance(theta_rep, rscales, scale, center=None):
    theta_rep = np.asarray(theta_rep, dtype=np.float64)
    if theta_rep.ndim == 1:
        theta_rep = theta_rep.reshape(-1, 1)
    if center is None:
        c = theta_rep.mean(axis=0)
    else:
        c = np.asarray(center, dtype=np.float64)
    d = theta_rep - c
    return scale * np.einsum("r,ri,rj->ij", rscales, d, d, optimize=True)
