cimport numpy as np
np.import_array()

cdef extern from "cs_kron.h":
    void cs_kron(const int *Ap, const int *Ai, const double *Ax, int Anr, int Anc,
                 const int *Bp, const int *Bi, const double *Bx, int Bnr, int Bnc,
                 int *Cp, int *Ci, double *Cx)

def cs_kron_wrapper(np.ndarray[int, ndim=1, mode="c"] Ap not None,
                    np.ndarray[int, ndim=1, mode="c"] Ai not None,
                    np.ndarray[double, ndim=1, mode="c"] Ax not None,
                    int Anr, int Anc,
                    np.ndarray[int, ndim=1, mode="c"] Bp not None,
                    np.ndarray[int, ndim=1, mode="c"] Bi not None,
                    np.ndarray[double, ndim=1, mode="c"] Bx not None,
                    int Bnr, int Bnc,
                    np.ndarray[int, ndim=1, mode="c"] Cp not None,
                    np.ndarray[int, ndim=1, mode="c"] Ci not None,
                    np.ndarray[double, ndim=1, mode="c"] Cx not None):
    cs_kron(&Ap[0], &Ai[0], &Ax[0], Anr, Anc,
            &Bp[0], &Bi[0], &Bx[0], Bnr, Bnc,
            &Cp[0], &Ci[0], &Cx[0])
    
  