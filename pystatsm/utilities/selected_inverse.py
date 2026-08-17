import numpy as np
import numba


@numba.njit(cache=True)
def _find_pos(Lp, Li, row, col):
    lo, hi = Lp[col], Lp[col + 1]
    while lo < hi:
        mid = (lo + hi) // 2
        v = Li[mid]
        if v == row:
            return mid
        if v < row:
            lo = mid + 1
        else:
            hi = mid
    return -1


@numba.njit(cache=True)
def lookup_positions(Lp, Li, rows, cols):
    m = rows.shape[0]
    pos = np.empty(m, dtype=np.int64)
    for t in range(m):
        pos[t] = _find_pos(Lp, Li, rows[t], cols[t])
    return pos


@numba.njit(cache=True)
def _takahashi(Lp, Li, Lu, d, Sx):
    n = Lp.shape[0] - 1
    flag = np.full(n, -1, dtype=np.int64)
    acc = np.zeros(n)
    lcoef = np.zeros(n)
    for c in range(n - 1, -1, -1):
        s0, s1 = Lp[c], Lp[c + 1]
        for t in range(s0 + 1, s1):
            k = Li[t]
            flag[k] = c
            lcoef[k] = Lu[t]
            acc[k] = 0.0
        for t in range(s0 + 1, s1):
            k = Li[t]
            lk = lcoef[k]
            for u in range(Lp[k], Lp[k + 1]):
                i = Li[u]
                if flag[i] == c:
                    sv = Sx[u]
                    acc[i] += lk * sv
                    if i != k:
                        acc[k] += lcoef[i] * sv
        diag_acc = 0.0
        for t in range(s0 + 1, s1):
            k = Li[t]
            Sx[t] = -acc[k]
            diag_acc += lcoef[k] * Sx[t]
            flag[k] = -1
        Sx[s0] = 1.0 / d[c] - diag_acc
    return Sx


def selected_inverse_data(Lp, Li, Lx):
    n = Lp.shape[0] - 1
    dvec = Lx[Lp[:-1]]
    cols = np.repeat(np.arange(n), np.diff(Lp))
    Lu = Lx / dvec[cols]
    d = dvec * dvec
    Sx = np.zeros_like(Lx)
    _takahashi(Lp, Li, Lu, d, Sx)
    return Sx


def selected_inverse(L_csc):
    L_csc.sort_indices()
    Lp = np.asarray(L_csc.indptr)
    Li = np.asarray(L_csc.indices)
    Sx = selected_inverse_data(Lp, Li, np.asarray(L_csc.data))
    return Sx, Lp, Li
