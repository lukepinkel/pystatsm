import timeit
import numpy as np
import scipy as sp
from pystatsm.utilities.python_wrappers import sparse_kron, csc_eq
rng = np.random.default_rng(123)



A = sp.sparse.random(100, 100, density=0.1, format="csc")
B = sp.sparse.random(100, 100, density=0.1, format="csc")

C1 = sparse_kron(A, B)
C2 = sp.sparse.kron(A, B, format='csc')

assert(csc_eq(C1, C2))
