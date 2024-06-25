cimport numpy as np
np.import_array()

cdef extern from "tile_1d.h":
    void tile_1d_modulo(const double *arr, int n, int reps, double *out)
    void tile_1d_nested(const double *arr, int n, int reps, double *out)

def tile_1d_modulo_wrapper(np.ndarray[double, ndim=1, mode="c"] arr not None,
                           int reps,
                           np.ndarray[double, ndim=1, mode="c"] out not None):
    cdef int n = arr.shape[0]
    tile_1d_modulo(&arr[0], n, reps, &out[0])

def tile_1d_nested_wrapper(np.ndarray[double, ndim=1, mode="c"] arr not None,
                           int reps,
                           np.ndarray[double, ndim=1, mode="c"] out not None):
    cdef int n = arr.shape[0]
    tile_1d_nested(&arr[0], n, reps, &out[0])