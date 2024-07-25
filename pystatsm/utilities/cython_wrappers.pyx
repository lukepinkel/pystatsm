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


cdef extern from "cs_kron_ss.h":
    void cs_kron_ss(const int *Ap, const int *Ai, const double *Ax, int Anr, int Anc,
                    const int *Bp, const int *Bi, const double *Bx, int Bnr, int Bnc,
                    int *Cp, int *Ci, double *Cx)
    void cs_kron_ss_inplace(const int *Ap, const int *Ai, const double *Ax, int Anr, int Anc,
                            const int *Bp, const int *Bi, const double *Bx, int Bnr, int Bnc,
                            double *Cx)

def cs_kron_ss_wrapper(np.ndarray[int, ndim=1, mode="c"] Ap not None,
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
    cs_kron_ss(&Ap[0], &Ai[0], &Ax[0], Anr, Anc,
               &Bp[0], &Bi[0], &Bx[0], Bnr, Bnc,
               &Cp[0], &Ci[0], &Cx[0])
    

def cs_kron_ss_inplace_wrapper(np.ndarray[int, ndim=1, mode="c"] Ap not None,
                               np.ndarray[int, ndim=1, mode="c"] Ai not None,
                               np.ndarray[double, ndim=1, mode="c"] Ax not None,
                               int Anr, int Anc,
                        np.ndarray[int, ndim=1, mode="c"] Bp not None,
                        np.ndarray[int, ndim=1, mode="c"] Bi not None,
                        np.ndarray[double, ndim=1, mode="c"] Bx not None,
                        int Bnr, int Bnc,
                        np.ndarray[double, ndim=1, mode="c"] Cx not None):
    cs_kron_ss_inplace(&Ap[0], &Ai[0], &Ax[0], Anr, Anc,
                       &Bp[0], &Bi[0], &Bx[0], Bnr, Bnc,
                       &Cx[0])

cdef extern from "cs_kron_ds.h":
    void cs_kron_ds(const double *A, int Anr, int Anc,
                    const int *Bp, const int *Bi, const double *Bx, int Bnr, int Bnc,
                    int *Cp, int *Ci, double *Cx)
    void cs_kron_ds_inplace(const double *A, int Anr, int Anc,
                            const int *Bp, const int *Bi, const double *Bx, int Bnr, int Bnc,
                            double *Cx)

def cs_kron_ds_wrapper(np.ndarray[double, ndim=2, mode="fortran"] A not None,
                       np.ndarray[int, ndim=1, mode="c"] Bp not None,
                       np.ndarray[int, ndim=1, mode="c"] Bi not None,
                       np.ndarray[double, ndim=1, mode="c"] Bx not None,
                       int Bnr, int Bnc,
                       np.ndarray[int, ndim=1, mode="c"] Cp not None,
                       np.ndarray[int, ndim=1, mode="c"] Ci not None,
                       np.ndarray[double, ndim=1, mode="c"] Cx not None):
    cs_kron_ds(&A[0,0], A.shape[0], A.shape[1],
               &Bp[0], &Bi[0], &Bx[0], Bnr, Bnc,
               &Cp[0], &Ci[0], &Cx[0])

def cs_kron_ds_inplace_wrapper(np.ndarray[double, ndim=2, mode="fortran"] A not None,
                               np.ndarray[int, ndim=1, mode="c"] Bp not None,
                               np.ndarray[int, ndim=1, mode="c"] Bi not None,
                               np.ndarray[double, ndim=1, mode="c"] Bx not None,
                               int Bnr, int Bnc,
                               np.ndarray[double, ndim=1, mode="c"] Cx not None):
    cs_kron_ds_inplace(&A[0,0], A.shape[0], A.shape[1],
                       &Bp[0], &Bi[0], &Bx[0], Bnr, Bnc,
                       &Cx[0])

cdef extern from "cs_matmul_inplace.h":
    void cs_matmul_inplace(const int *Ap, const int *Ai, const double *Ax, int Anr, int Anc,
                           const int *Bp, const int *Bi, const double *Bx, int Bnr, int Bnc,
                           int *Cp, int *Ci, double *Cx)

def cs_matmul_inplace_wrapper(np.ndarray[int, ndim=1, mode="c"] Ap not None,
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
    cs_matmul_inplace(&Ap[0], &Ai[0], &Ax[0], Anr, Anc,
                      &Bp[0], &Bi[0], &Bx[0], Bnr, Bnc,
                      &Cp[0], &Ci[0], &Cx[0])
    

cdef extern from "cs_kron_sd.h":
    void cs_kron_sd(const int *Ap, const int *Ai, const double *Ax, int Anr, int Anc,
                    const double *B, int Bnr, int Bnc,
                    int *Cp, int *Ci, double *Cx)
    void cs_kron_sd_inplace(const int *Ap, const int *Ai, const double *Ax, int Anr, int Anc,
                            const double *B, int Bnr, int Bnc,
                            double *Cx)

def cs_kron_sd_wrapper(np.ndarray[int, ndim=1, mode="c"] Ap not None,
                       np.ndarray[int, ndim=1, mode="c"] Ai not None,
                       np.ndarray[double, ndim=1, mode="c"] Ax not None,
                       int Anr, int Anc,
                       np.ndarray[double, ndim=2, mode="fortran"] B not None,
                       int Bnr, int Bnc,
                       np.ndarray[int, ndim=1, mode="c"] Cp not None,
                       np.ndarray[int, ndim=1, mode="c"] Ci not None,
                       np.ndarray[double, ndim=1, mode="c"] Cx not None):
    cs_kron_sd(&Ap[0], &Ai[0], &Ax[0], Anr, Anc,
               &B[0,0], Bnr, Bnc,
               &Cp[0], &Ci[0], &Cx[0])

def cs_kron_sd_inplace_wrapper(np.ndarray[int, ndim=1, mode="c"] Ap not None,
                               np.ndarray[int, ndim=1, mode="c"] Ai not None,
                               np.ndarray[double, ndim=1, mode="c"] Ax not None,
                               int Anr, int Anc,
                               np.ndarray[double, ndim=2, mode="fortran"] B not None,
                               int Bnr, int Bnc,
                               np.ndarray[double, ndim=1, mode="c"] Cx not None):
    cs_kron_sd_inplace(&Ap[0], &Ai[0], &Ax[0], Anr, Anc,
                       &B[0,0], Bnr, Bnc,
                       &Cx[0])
    
cdef extern from "cs_kron_id_sp.h":
    void cs_kron_id_sp(int m, const double *Bx, const int *Bi, const int *Bp, 
                       int Bnr, int Bnc, double *Cx, int *Ci, int *Cp)
    void cs_kron_id_sp_inplace(int m, const double *Bx, const int *Bi, const int *Bp, 
                               int Bnr, int Bnc, double *Cx, int *Ci, int *Cp)

def cs_kron_id_sp_wrapper(int m,
                          np.ndarray[double, ndim=1, mode="c"] Bx not None,
                          np.ndarray[int, ndim=1, mode="c"] Bi not None,
                          np.ndarray[int, ndim=1, mode="c"] Bp not None,
                          int Bnr, int Bnc,
                          np.ndarray[double, ndim=1, mode="c"] Cx not None,
                          np.ndarray[int, ndim=1, mode="c"] Ci not None,
                          np.ndarray[int, ndim=1, mode="c"] Cp not None):
    cs_kron_id_sp(m, &Bx[0], &Bi[0], &Bp[0], Bnr, Bnc, &Cx[0], &Ci[0], &Cp[0])

def cs_kron_id_sp_inplace_wrapper(int m,
                                  np.ndarray[double, ndim=1, mode="c"] Bx not None,
                                  np.ndarray[int, ndim=1, mode="c"] Bi not None,
                                  np.ndarray[int, ndim=1, mode="c"] Bp not None,
                                  int Bnr, int Bnc,
                                  np.ndarray[double, ndim=1, mode="c"] Cx not None,
                                  np.ndarray[int, ndim=1, mode="c"] Ci not None,
                                  np.ndarray[int, ndim=1, mode="c"] Cp not None):
    cs_kron_id_sp_inplace(m, &Bx[0], &Bi[0], &Bp[0], Bnr, Bnc, &Cx[0], &Ci[0], &Cp[0])


cdef extern from "cs_dot.h":
    void cs_dot(const double *ax, const int *ai, const int anz, 
                const double *bx, const int *bi, const int bnz, 
                double *cx)
    
def cs_dot_wrapper(np.ndarray[double, ndim=1, mode="c"] ax not None,
                   np.ndarray[int, ndim=1, mode="c"] ai not None,
                   int anz,
                   np.ndarray[double, ndim=1, mode="c"] bx not None,
                   np.ndarray[int, ndim=1, mode="c"] bi not None,
                   int bnz,
                   double cx):
    cs_dot(&ax[0], &ai[0], anz, &bx[0], &bi[0], bnz, &cx)
    return cx

cdef extern from "cs_pattern_trace.h":
    void cs_pattern_trace(const int *Bp, const int *Bi, const double *Bx, int Bnr, int Bnc, int Bnz,
                          const int *Cp, const int *Ci, const double *Cx, int Cnr, int Cnc, int Cnz,
                          double *trace)

def cs_pattern_trace_wrapper(np.ndarray[int, ndim=1, mode="c"] Bp not None,
                             np.ndarray[int, ndim=1, mode="c"] Bi not None,
                             np.ndarray[double, ndim=1, mode="c"] Bx not None,
                             int Bnr, int Bnc, int Bnz,
                             np.ndarray[int, ndim=1, mode="c"] Cp not None,
                             np.ndarray[int, ndim=1, mode="c"] Ci not None,
                             np.ndarray[double, ndim=1, mode="c"] Cx not None,
                             int Cnr, int Cnc, int Cnz,
                             double trace):
    cs_pattern_trace(&Bp[0], &Bi[0], &Bx[0], Bnr, Bnc, Bnz,
                     &Cp[0], &Ci[0], &Cx[0], Cnr, Cnc, Cnz,
                     &trace)
    return trace


cdef extern from "repeat_1d.h":
    void repeat_1d(const double *arr, int n, int reps, double *out)

def repeat_1d_wrapper(np.ndarray[double, ndim=1, mode="c"] arr not None,
                      int reps,
                      np.ndarray[double, ndim=1, mode="c"] out not None):
    cdef int n = arr.shape[0]
    repeat_1d(&arr[0], n, reps, &out[0])


cdef extern from "tile_1d.h":
    void tile_1d_nested(const double *arr, int n, int reps, double *out)


def tile_1d_nested_wrapper(np.ndarray[double, ndim=1, mode="c"] arr not None,
                           int reps,
                           np.ndarray[double, ndim=1, mode="c"] out not None):
    cdef int n = arr.shape[0]
    tile_1d_nested(&arr[0], n, reps, &out[0])


cdef extern from "tile_1d_complex.h":
    void tile_1d_nested_complex(const double complex *arr, int n, int reps, double complex *out)


def tile_1d_complex_wrapper(np.ndarray[np.complex128_t, ndim=1, mode="c"] arr not None,
                            int reps,
                            np.ndarray[np.complex128_t, ndim=1, mode="c"] out not None):
    cdef int n = arr.shape[0]
    tile_1d_nested_complex(<double complex*>&arr[0], n, reps, <double complex*>&out[0])

cdef extern from "naive_matmul_inplace.h":
    void naive_matmul_inplace(double *A, double *B, double *C, size_t N, size_t M, size_t K)
    void naive_matmul_inplace2(double *A, double *B, double *C, size_t N, size_t M, size_t K)

def naive_matmul_inplace_wrapper(np.ndarray[double, ndim=2, mode="c"] A not None,
                                 np.ndarray[double, ndim=2, mode="c"] B not None,
                                 np.ndarray[double, ndim=2, mode="c"] C not None):
    naive_matmul_inplace(&A[0,0], &B[0,0], &C[0,0], A.shape[0], A.shape[1], B.shape[1])


def naive_matmul_inplace2_wrapper(np.ndarray[double, ndim=2, mode="c"] A not None,
                                 np.ndarray[double, ndim=2, mode="c"] B not None,
                                 np.ndarray[double, ndim=2, mode="c"] C not None):
    naive_matmul_inplace2(&A[0,0], &B[0,0], &C[0,0], A.shape[0], A.shape[1], B.shape[1])


cdef extern from "cs_matmul_inplace_complex.h":
    void cs_matmul_inplace_complex(const int *Ap, const int *Ai, const double complex *Ax, int Anr, int Anc,
                                   const int *Bp, const int *Bi, const double complex *Bx, int Bnr, int Bnc,
                                   int *Cp, int *Ci, double complex *Cx)
    
def cs_matmul_inplace_complex_wrapper(np.ndarray[int, ndim=1, mode="c"] Ap not None,
                                      np.ndarray[int, ndim=1, mode="c"] Ai not None,
                                      np.ndarray[np.complex128_t, ndim=1, mode="c"] Ax not None,
                                      int Anr, int Anc,
                                      np.ndarray[int, ndim=1, mode="c"] Bp not None,
                                      np.ndarray[int, ndim=1, mode="c"] Bi not None,
                                      np.ndarray[np.complex128_t, ndim=1, mode="c"] Bx not None,
                                      int Bnr, int Bnc,
                                      np.ndarray[int, ndim=1, mode="c"] Cp not None,
                                      np.ndarray[int, ndim=1, mode="c"] Ci not None,
                                      np.ndarray[np.complex128_t, ndim=1, mode="c"] Cx not None):
    cs_matmul_inplace_complex(&Ap[0], &Ai[0], <double complex*>&Ax[0], Anr, Anc,
                              &Bp[0], &Bi[0], <double complex*>&Bx[0], Bnr, Bnc,
                              &Cp[0], &Ci[0], <double complex*>&Cx[0])
    
    
cdef extern from "cs_add_inplace_complex.h":
    void cs_add_inplace_complex(const int *Ap, const int *Ai, const double complex *Ax,
                                const int *Bp, const int *Bi, const double complex *Bx,
                                double complex alpha, double complex beta,
                                int *Cp, int *Ci, double complex *Cx, int Cnr, int Cnc)

def cs_add_inplace_complex_wrapper(np.ndarray[int, ndim=1, mode="c"] Ap not None,
                                   np.ndarray[int, ndim=1, mode="c"] Ai not None,
                                   np.ndarray[np.complex128_t, ndim=1, mode="c"] Ax not None,
                                   np.ndarray[int, ndim=1, mode="c"] Bp not None,
                                   np.ndarray[int, ndim=1, mode="c"] Bi not None,
                                   np.ndarray[np.complex128_t, ndim=1, mode="c"] Bx not None,
                                   np.complex128_t alpha,
                                   np.complex128_t beta,
                                   np.ndarray[int, ndim=1, mode="c"] Cp not None,
                                   np.ndarray[int, ndim=1, mode="c"] Ci not None,
                                   np.ndarray[np.complex128_t, ndim=1, mode="c"] Cx not None,
                                   int Cnr, int Cnc):
    cs_add_inplace_complex(&Ap[0], &Ai[0], <double complex*>&Ax[0],
                           &Bp[0], &Bi[0], <double complex*>&Bx[0],
                           <double complex>alpha, <double complex>beta,
                           &Cp[0], &Ci[0], <double complex*>&Cx[0], Cnr, Cnc)
    