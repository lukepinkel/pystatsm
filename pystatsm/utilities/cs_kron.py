import scipy as sp
import numpy as np
import numba
from .cs_kron_wrapper import cs_kron_wrapper
from .indexing_utils import vech_inds_reverse



def fully_dense_to_csc(arr):
    n, m = arr.shape
    indptr = (np.arange(0, m+1)*n).astype(np.int32)#np.arange(0, (n+1)*m, n, dtype=np.int32) 
    indices = np.tile(np.arange(n, dtype=np.int32), m)
    data = arr.reshape(-1, order='F')
    nnz = n * m
    return (n, m), nnz, indptr, indices, data

def sparse_dense_kron(A, B):
    Anr, Anc = A.shape
    Anz, Ap, Ai, Ax = A.nnz, A.indptr, A.indices, A.data

    (Bnr, Bnc), Bnz, Bp, Bi, Bx = fully_dense_to_csc(B)
    Cnr = Anr * Bnr
    Cnc = Anc * Bnc 
    Cnz = Anz * Bnz
    Cp = np.zeros(Cnc+1, dtype=np.int32)
    Ci = np.zeros(Cnz, dtype=np.int32)
    Cx = np.zeros(Cnz, dtype=np.double)
    
    cs_kron_wrapper(Ap, Ai, Ax, Anr, Anc, Bp, Bi, Bx, Bnr, Bnc, Cp, Ci, Cx)
    return sp.sparse.csc_array((Cx, Ci, Cp), shape=(Cnr, Cnc))


def _id_kron_coo(q, B):
    #I_q and A (n, m) to (qn, qm)
    n, m = B.shape
    qm = q * m
    cols = np.repeat(np.arange(qm), n)
    row_tr = np.tile(np.arange(n), qm)
    row_bl = np.repeat(np.arange(q) * n, n * m)
    rows = row_tr + row_bl
    data = np.tile(B.reshape(-1, order='F'), q)
    nnz = q * n * m
    return (q*n, q*m), nnz, rows, cols, data

def id_kron_coo(q, B):
    (Cnr, Cnc), Cnz, Cr, Cc, Cx = _id_kron_coo(q, B)
    return sp.sparse.coo_array((Cx, (Cr, Cc)), shape=(Cnr, Cnc))


def _id_kron_csc(q, B):
    n, m = B.shape
    qm = q * m
    indptr = np.arange(0, qm+1) * n
    #row indices tiled 
    # [0,...,n] [0,...,n] (qm times)
    # m times for B and q times the blocks
    row_tile = np.tile(np.arange(n), qm) 
    # [0,n, 2n, ..., (q-1)n] are the block offsets 
    # [0,,...,0] [n,...,n] [2n,...,2n]...
    # nm times for each entry in each blocks B
    block_offset = np.repeat(np.arange(q) * n, n * m)
    indices =  row_tile + block_offset
    
    indptr = indptr.astype(np.int32)
    indices = indices.astype(np.int32)
    data = np.tile(B.reshape(-1, order='F'), q)
    nnz = q * n * m
    return (q*n, q*m), nnz, indptr, indices, data

def id_kron_csc(q, B):
    (Cnr, Cnc), Cnz, Cp, Ci, Cx = _id_kron_csc(q, B)
    return sp.sparse.csc_array((Cx, Ci, Cp), shape=(Cnr, Cnc))


def sparse_kron(A, B):        
    Anr, Anc = A.shape
    Bnr, Bnc = B.shape
    Anz, Bnz = A.nnz, B.nnz
    
    Ap, Ai, Ax = A.indptr, A.indices, A.data
    Bp, Bi, Bx = B.indptr, B.indices, B.data
    
    Cnr = Anr * Bnr
    Cnc = Anc * Bnc 
    Cnz = Anz * Bnz
    Cp = np.zeros(Cnc+1, dtype=np.int32)
    Ci = np.zeros(Cnz, dtype=np.int32)
    Cx = np.zeros(Cnz, dtype=np.double)
    
    cs_kron_wrapper(Ap, Ai, Ax, Anr, Anc, Bp, Bi, Bx, Bnr, Bnc, Cp, Ci, Cx)
    return sp.sparse.csc_array((Cx, Ci, Cp), shape=(Cnr, Cnc))

@numba.jit(
    numba.void(numba.int32[:], numba.int32[:], numba.float64[:], numba.int32, numba.int32,
               numba.int32[:], numba.int32[:], numba.float64[:], numba.int32, numba.int32,
               numba.int32[:], numba.int32[:], numba.float64[:]), nopython=True)
def _cs_kron_jit(Ap, Ai, Ax, Anr, Anc, Bp, Bi, Bx, Bnr, Bnc, Cp, Ci, Cx):
    cnt = 0
    for a_col in range(Anc):
        for b_col in range(Bnc):
            for a_i in range(Ap[a_col], Ap[a_col+1]):
                rout = Ai[a_i] * Bnr
                ax = Ax[a_i]
                for b_i in range(Bp[b_col], Bp[b_col+1]):
                    Cx[cnt] = ax * Bx[b_i]
                    Ci[cnt] = Bi[b_i] + rout
                    cnt += 1
            Cp[a_col * Bnc + b_col + 1] = cnt
    
            
def sparse_kron_jit(A, B):
    Anr, Anc = A.shape
    Bnr, Bnc = B.shape
    Anz, Bnz = A.nnz, B.nnz
    
    Ap, Ai, Ax = A.indptr, A.indices, A.data
    Bp, Bi, Bx = B.indptr, B.indices, B.data
    
    Cnr = Anr * Bnr
    Cnc = Anc * Bnc 
    Cnz = Anz * Bnz
    Cp = np.zeros(Cnc+1, dtype=np.int32)
    Ci = np.zeros(Cnz, dtype=np.int32)
    Cx = np.zeros(Cnz, dtype=np.double)
    
    _cs_kron_jit(Ap, Ai, Ax, Anr, Anc, Bp, Bi, Bx, Bnr, Bnc, Cp, Ci, Cx)
    return sp.sparse.csc_array((Cx, Ci, Cp), shape=(Cnr, Cnc))

def get_csc_eq(A, B):
    b = ((np.allclose(A.data, B.data)) & 
        (np.allclose(A.indices, B.indices)) &
        (np.allclose(A.indptr, B.indptr)))
    return b
    
        
def dkr_spsq_dnsq(A, p, r, return_array=True):
    Anr, Anc = A.shape
    Bnr, Bnc = p, p
    Anz = A.nnz
   
    Ap, Ai, Ax = A.indptr, A.indices, A.data
    row, col = vech_inds_reverse(r, p)
    diag = row==col
    
    Cnr = Anr * p
    Cnc = Anc * p
    
    Bp = np.zeros(p+1, dtype=np.int32)
    Bp[row+1:]+=1
    if diag:
        Bi=np.array([row], dtype=np.int32)
        Bx = np.array([1], dtype=np.double)
        Bnz = 1
    else:
        Bp[col+1:]+=1
        Bi = np.array([row, col], dtype=np.int32)
        Bx = np.array([1, 1], dtype=np.double)
        Bnz = 2
        
    Cnz = Anz * Bnz
   
    Cp = np.zeros(Cnc + 1, dtype=np.int32)
    Ci = np.zeros(Cnz, dtype=np.int32)
    Cx = np.zeros(Cnz, dtype=np.double)
    cs_kron_wrapper(Ap, Ai, Ax, Anr, Anc, Bp, Bi, Bx, Bnr, Bnc, Cp, Ci, Cx)
    if return_array:
        ret = sp.sparse.csc_array((Cx, Ci, Cp), shape=(Cnr, Cnc))
    else:
        ret = (Cnr, Cnc) , Cnz, Cp, Ci, Cx
    return ret
        
def dkr_spid_dnsq(q, p, r, return_array=True):
    Anr, Anc = q, q
    Bnr, Bnc = p, p
    Anz = q
   
    Ap = np.arange(q+1, dtype=np.int32)
    Ai = np.arange(q, dtype=np.int32)
    Ax = np.ones(q, dtype=np.double)
    row, col = vech_inds_reverse(r, p)
    diag = row==col
    
    Cnr = Anr * p
    Cnc = Anc * p
    
    Bp = np.zeros(p+1, dtype=np.int32)
    Bp[row+1:]+=1
    if diag:
        Bi=np.array([row], dtype=np.int32)
        Bx = np.array([1], dtype=np.double)
        Bnz = 1
    else:
        Bp[col+1:]+=1
        Bi = np.array([row, col], dtype=np.int32)
        Bx = np.array([1, 1], dtype=np.double)
        Bnz = 2
        
    Cnz = Anz * Bnz
   
    Cp = np.zeros(Cnc + 1, dtype=np.int32)
    Ci = np.zeros(Cnz, dtype=np.int32)
    Cx = np.zeros(Cnz, dtype=np.double)
    cs_kron_wrapper(Ap, Ai, Ax, Anr, Anc, Bp, Bi, Bx, Bnr, Bnc, Cp, Ci, Cx)
    if return_array:
        ret = sp.sparse.csc_array((Cx, Ci, Cp), shape=(Cnr, Cnc))
    else:
        ret = (Cnr, Cnc) , Cnz, Cp, Ci, Cx
    return ret
        

