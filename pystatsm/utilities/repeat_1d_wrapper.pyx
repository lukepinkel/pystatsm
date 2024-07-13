cimport numpy as np
np.import_array()

cdef extern from "repeat_1d.h":
    void repeat_1d(const double *arr, int n, int reps, double *out)

def repeat_1d_wrapper(np.ndarray[double, ndim=1, mode="c"] arr not None,
                      int reps,
                      np.ndarray[double, ndim=1, mode="c"] out not None):
    cdef int n = arr.shape[0]
    repeat_1d(&arr[0], n, reps, &out[0])