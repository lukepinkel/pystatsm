import timeit
import numpy as np
import scipy as sp
from pystatsm.utilities.cs_kron import sparse_kron, sparse_kron_jit, get_csc_eq
rng = np.random.default_rng(123)



A = sp.sparse.random(100, 100, density=0.1, format="csc")
B = sp.sparse.random(100, 100, density=0.1, format="csc")

C1 = sparse_kron(A, B)
C2 = sp.sparse.kron(A, B, format='csc')
C3 = sparse_kron_jit(A, B)
assert(get_csc_eq(C1, C2))
assert(get_csc_eq(C2, C3))
assert(get_csc_eq(C1, C3))
