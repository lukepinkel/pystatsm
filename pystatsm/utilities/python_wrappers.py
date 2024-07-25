import numpy as np
import scipy as sp
from . import utils
from . import cython_wrappers
from .cython_wrappers import (coo_to_csc_wrapper,
                              cs_add_inplace_wrapper,
                              cs_kron_ss_wrapper,
                              cs_kron_ss_inplace_wrapper,
                              cs_kron_ds_wrapper,
                              cs_kron_ds_inplace_wrapper,
                              cs_kron_sd_wrapper,
                              cs_kron_sd_inplace_wrapper,
                              cs_kron_id_sp_wrapper,
                              cs_kron_id_sp_inplace_wrapper,
                              cs_matmul_inplace_wrapper,
                              cs_dot_wrapper,
                              cs_pattern_trace_wrapper,
                              naive_matmul_inplace_wrapper,
                              naive_matmul_inplace2_wrapper,
                              repeat_1d_wrapper,
                              tile_1d_nested_wrapper,
                              tile_1d_complex_wrapper,
                              cs_matmul_inplace_complex_wrapper,
                              cs_add_inplace_complex_wrapper)
                              

import scipy as sp
import numpy as np


def fully_dense_to_csc(arr):
    n, m = arr.shape
    indptr = (np.arange(0, m+1)*n).astype(np.int32)#np.arange(0, (n+1)*m, n, dtype=np.int32) 
    indices = np.tile(np.arange(n, dtype=np.int32), m)
    data = arr.reshape(-1, order='F')
    nnz = n * m
    return (n, m), nnz, indptr, indices, data

def csc_eq(A, B):
    b = ((np.allclose(A.data, B.data)) & 
        (np.allclose(A.indices, B.indices)) &
        (np.allclose(A.indptr, B.indptr)))
    return b
    

def symbolic_csc_like(arr, zero=True):
    res = arr.copy()
    if zero:
        res.data = res.data * 0.0
    return res

def cs_add_inplace(A, B, C, alpha=1.0, beta=1.0):
    Ap, Ai, Ax = A.indptr, A.indices, A.data
    Bp, Bi, Bx = B.indptr, B.indices, B.data
    Cp, Ci, Cx = C.indptr, C.indices, C.data
    Cnr, Cnc = C.shape
    
    cs_add_inplace_wrapper(Ap, Ai, Ax, Bp, Bi, Bx, alpha, beta, Cp, Ci, Cx, Cnr, Cnc)
    C.eliminate_zeros()
    return C


def cs_add_inplace_complex(A, B, C, alpha=1.0+0j, beta=1.0+0j):
    Ap, Ai, Ax = A.indptr, A.indices, A.data
    Bp, Bi, Bx = B.indptr, B.indices, B.data
    Cp, Ci, Cx = C.indptr, C.indices, C.data
    Cnr, Cnc = C.shape
    
    cs_add_inplace_complex_wrapper(Ap, Ai, Ax, Bp, Bi, Bx, alpha, beta, Cp, Ci, Cx, Cnr, Cnc)
    C.eliminate_zeros()
    return C

def tile_1d(arr, reps, out=None):
    n = len(arr)
    out = np.zeros(n*reps, dtype=np.double) if out is None else out
    tile_1d_nested_wrapper(arr, reps, out)
    return out

def tile_1d_complex(arr, reps, out=None):
    n = len(arr)
    out = np.zeros(n*reps, dtype=np.complex128) if out is None else out
    tile_1d_complex_wrapper(arr, reps, out)
    return out


def repeat_1d(arr, reps, out=None):
    n = len(arr)
    out = np.zeros(n*reps, dtype=np.double) if out is None else out
    repeat_1d_wrapper(arr, reps, out)
    return out


def cs_matmul_inplace(A, B, C):
    Anr, Anc = A.shape
    Bnr, Bnc = B.shape
    
    Ap, Ai, Ax = A.indptr, A.indices, A.data
    Bp, Bi, Bx = B.indptr, B.indices, B.data
    Cp, Ci, Cx = C.indptr, C.indices, C.data
    
    cs_matmul_inplace_wrapper(Ap, Ai, Ax, Anr, Anc, Bp, Bi, Bx, Bnr, Bnc, Cp, Ci, Cx)
    C.sort_indices()
    return C


def cs_matmul_inplace_complex(A, B, C):
    Anr, Anc = A.shape
    Bnr, Bnc = B.shape
    
    Ap, Ai, Ax = A.indptr, A.indices, A.data
    Bp, Bi, Bx = B.indptr, B.indices, B.data
    Cp, Ci, Cx = C.indptr, C.indices, C.data
    
    cs_matmul_inplace_complex_wrapper(Ap, Ai, Ax, Anr, Anc, Bp, Bi, Bx, Bnr, Bnc, Cp, Ci, Cx)
    C.sort_indices()
    return C


def coo_to_csc(row_inds, col_inds, data, shape, return_array=True):
    Anr, Anc = shape
    Anz =  data.shape[0] 
    Ai, Aj = row_inds.astype(np.int32), col_inds.astype(np.int32)
    Ax = data.astype(np.double)
    Bp = np.zeros((Anc+1), dtype=np.int32)
    Bi = np.zeros(Anz, dtype=np.int32)
    Bx = np.zeros(Anz, dtype=np.double)
    coo_to_csc_wrapper(Anr, Anc, Anz, Ai, Aj, Ax, Bp, Bi, Bx)
    if return_array:
        ret = sp.sparse.csc_array((Bx, Bi, Bp), shape=(Anr, Anc))
    else:
        ret = (Anr, Anc) , Anz, Bp, Bi, Bx
    return ret
        
    
def dense_sparse_kron(A, B, out=None):
    Bnr, Bnc = B.shape
    Bnz, Bp, Bi, Bx = B.nnz, B.indptr, B.indices, B.data
    (Anr, Anc), Anz, Ap, Ai, Ax = fully_dense_to_csc(A)
    Cnr = Anr * Bnr
    Cnc = Anc * Bnc 
    Cnz = Anz * Bnz
    
    if out is None:
        Cx = np.zeros(Cnz, dtype=np.double)
        Ci = np.zeros(Cnz, dtype=np.int32)
        Cp = np.zeros(Cnc+1, dtype=np.int32)
    elif sp.sparse.issparse(out):
        Cx, Ci, Cp = out.data, out.indices, out.indptr
    else:
        Cx, Ci, Cp = out
    
    cs_kron_ss_wrapper(Ap, Ai, Ax, Anr, Anc, Bp, Bi, Bx, Bnr, Bnc, Cp, Ci, Cx)
    
    if out is None:
        return sp.sparse.csc_array((Cx, Ci, Cp), shape=(Cnr, Cnc))
    elif sp.sparse.issparse(out):
        return out
    else:
        return Cx, Ci, Cp
    


def sparse_kron(A, B, out=None):        
    Anr, Anc = A.shape
    Bnr, Bnc = B.shape
    Anz, Bnz = A.nnz, B.nnz
    
    Ap, Ai, Ax = A.indptr, A.indices, A.data
    Bp, Bi, Bx = B.indptr, B.indices, B.data
    
    Cnr = Anr * Bnr
    Cnc = Anc * Bnc 
    Cnz = Anz * Bnz
    if out is None:
        Cx = np.zeros(Cnz, dtype=np.double)
        Ci = np.zeros(Cnz, dtype=np.int32)
        Cp = np.zeros(Cnc+1, dtype=np.int32)
    elif sp.sparse.issparse(out):
        Cx, Ci, Cp = out.data, out.indices, out.indptr
    else:
        Cx, Ci, Cp = out
   
    
    cs_kron_ss_wrapper(Ap, Ai, Ax, Anr, Anc, Bp, Bi, Bx, Bnr, Bnc, Cp, Ci, Cx)
    if out is None:
        return sp.sparse.csc_array((Cx, Ci, Cp), shape=(Cnr, Cnc))
    elif sp.sparse.issparse(out):
        return out
    else:
        return Cx, Ci, Cp
    
def sparse_kron_inplace(A, B, C):
    Anr, Anc = A.shape
    Bnr, Bnc = B.shape
    
    Ap, Ai, Ax = A.indptr, A.indices, A.data
    Bp, Bi, Bx = B.indptr, B.indices, B.data
    Cx = C.data
    cs_kron_ss_inplace_wrapper(Ap, Ai, Ax, Anr, Anc, Bp, Bi, Bx, Bnr, Bnc, Cx)
    return C

def ds_kron(A, B, out=None):
    Anr, Anc = A.shape
    Bnr, Bnc = B.shape
    Bnz = B.nnz
    
    if out is None:
        Cx = np.zeros(Anr * Anc * Bnz, dtype=np.double)
        Ci = np.zeros(Anr * Anc * Bnz, dtype=np.int32)
        Cp = np.zeros(Anc * Bnc + 1, dtype=np.int32)
    elif sp.sparse.issparse(out):
        Cx, Ci, Cp = out.data, out.indices, out.indptr
    else:
        Cx, Ci, Cp = out
    
    cs_kron_ds_wrapper(A, B.indptr, B.indices, B.data, Bnr, Bnc, Cp, Ci, Cx)
    
    if out is None:
        return sp.sparse.csc_array((Cx, Ci, Cp), shape=(Anr * Bnr, Anc * Bnc))
    elif sp.sparse.issparse(out):
        return out
    else:
        return Cx, Ci, Cp

def ds_kron_inplace(A, B, C):
    Anr, Anc = A.shape
    Bnr, Bnc = B.shape
    
    cs_kron_ds_inplace_wrapper(A, B.indptr, B.indices, B.data, Bnr, Bnc, C.data)
    return C

def naive_matmul_inplace(A, B, C=None): 
    anr, bnc = A.shape[0], B.shape[1]
    C = np.zeros((anr, bnc), dtype=np.double, order='C') if C is None else C
    naive_matmul_inplace_wrapper(A, B, C)
    return C


def naive_matmul_inplace2(A, B, C=None): 
    anr, bnc = A.shape[0], B.shape[1]
    C = np.zeros((anr, bnc), dtype=np.double, order='C') if C is None else C
    naive_matmul_inplace2_wrapper(A, B, C)
    return C

def id_sp_kron(m, B):
    Bnr, Bnc = B.shape
    Bp, Bi, Bx = B.indptr, B.indices, B.data
    nnz = B.nnz
    
    Cp = np.zeros(m * Bnc + 1, dtype=np.int32)
    Ci = np.zeros(m * nnz, dtype=np.int32)
    Cx = np.zeros(m * nnz, dtype=np.float64)
    
    cs_kron_id_sp_wrapper(m, Bx, Bi, Bp, Bnc, Bnc, Cx, Ci, Cp)
    
    return sp.sparse.csc_matrix((Cx, Ci, Cp), shape=(m * Bnr, m * Bnc))


def id_sp_kron_inplace(m, B, C):
    Bnr, Bnc = B.shape
    Bp, Bi, Bx = B.indptr, B.indices, B.data
    Cp, Ci, Cx = C.indptr, C.indices, C.data
   
    cs_kron_id_sp_inplace_wrapper(m, Bx, Bi, Bp, Bnc, Bnc, Cx, Ci, Cp)
    
    return C


def sparse_dense_kron(A, B, out=None):
    Anr, Anc = A.shape
    Bnr, Bnc = B.shape
    Anz = A.nnz
    
    Ap, Ai, Ax = A.indptr, A.indices, A.data
    
    Cnr = Anr * Bnr
    Cnc = Anc * Bnc 
    Cnz = Anz * Bnr * Bnc
    
    if out is None:
        Cx = np.zeros(Cnz, dtype=np.double)
        Ci = np.zeros(Cnz, dtype=np.int32)
        Cp = np.zeros(Cnc+1, dtype=np.int32)
    elif sp.sparse.issparse(out):
        Cx, Ci, Cp = out.data, out.indices, out.indptr
    else:
        Cx, Ci, Cp = out
   
    cs_kron_sd_wrapper(Ap, Ai, Ax, Anr, Anc, B, Bnr, Bnc, Cp, Ci, Cx)
    
    if out is None:
        return sp.sparse.csc_array((Cx, Ci, Cp), shape=(Cnr, Cnc))
    elif sp.sparse.issparse(out):
        return out
    else:
        return Cx, Ci, Cp
    
def sparse_dot(a, b, c=0.0):
    c = cs_dot_wrapper(a.data, a.indices, a.nnz, b.data, b.indices, b.nnz, c)
    return c

def sparse_pattern_trace(B, C, out=0.0):
    out = cs_pattern_trace_wrapper(B.indptr, B.indices, B.data, B.shape[0], B.shape[1], B.nnz,
                                   C.indptr, C.indices, C.data, C.shape[0], C.shape[1], C.nnz,
                                   out)
    
    return out

def sparse_dense_kron_inplace(A, B, C):
    Anr, Anc = A.shape
    Bnr, Bnc = B.shape
    
    Ap, Ai, Ax = A.indptr, A.indices, A.data
    Cx = C.data
    cs_kron_sd_inplace_wrapper(Ap, Ai, Ax, Anr, Anc, B, Bnr, Bnc, Cx)
    return C