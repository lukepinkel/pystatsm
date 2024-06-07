cimport numpy as np
np.import_array()

cdef extern from "coo_to_csc.h":
    void coo_to_csc(int Anr, int Anc, int Anz, const int *Ai, const int *Aj, const double *Ax,
                    int *Bp, int *Bi, double *Bx)

def coo_to_csc_wrapper(int Anr, int Anc, int Anz,
                       np.ndarray[int, ndim=1, mode="c"] Ai not None,
                       np.ndarray[int, ndim=1, mode="c"] Aj not None,
                       np.ndarray[double, ndim=1, mode="c"] Ax not None,
                       np.ndarray[int, ndim=1, mode="c"] Bp not None,
                       np.ndarray[int, ndim=1, mode="c"] Bi not None,
                       np.ndarray[double, ndim=1, mode="c"] Bx not None):
    coo_to_csc(Anr, Anc, Anz, &Ai[0], &Aj[0], &Ax[0], &Bp[0], &Bi[0], &Bx[0])