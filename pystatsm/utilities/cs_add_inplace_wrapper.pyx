cimport numpy as np
np.import_array()

cdef extern from "cs_add_inplace.h":
    void cs_add_inplace(const int *Ap, const int *Ai, const double *Ax,
                const int *Bp, const int *Bi, const double *Bx,
                double alpha, double beta,
                int *Cp, int *Ci, double *Cx, int Cnr, int Cnc)

def cs_add_inplace_wrapper(np.ndarray[int, ndim=1, mode="c"] Ap not None,
                   np.ndarray[int, ndim=1, mode="c"] Ai not None,
                   np.ndarray[double, ndim=1, mode="c"] Ax not None,
                   np.ndarray[int, ndim=1, mode="c"] Bp not None,
                   np.ndarray[int, ndim=1, mode="c"] Bi not None,
                   np.ndarray[double, ndim=1, mode="c"] Bx not None,
                   double alpha, double beta,
                   np.ndarray[int, ndim=1, mode="c"] Cp not None,
                   np.ndarray[int, ndim=1, mode="c"] Ci not None,
                   np.ndarray[double, ndim=1, mode="c"] Cx not None,
                   int Cnr, int Cnc):
    cs_add_inplace(&Ap[0], &Ai[0], &Ax[0],
                   &Bp[0], &Bi[0], &Bx[0],
                   alpha, beta,
                   &Cp[0], &Ci[0], &Cx[0], Cnr, Cnc)