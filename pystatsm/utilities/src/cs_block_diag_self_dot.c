#include "cs_block_diag_self_dot.h"

/* For a CSR matrix L of shape (ng*nv, n_cols) (rows ordered (level, intra-row)
   so row j*nv + a is the a-th row of level j), compute the (nv, nv) summary

       out[a*nv + b] = sum_{j=0}^{ng-1} sum_p L[j*nv+a, p] * L[j*nv+b, p]

   in row-major. Each (a, b) inner sum is a sorted-merge dot product over the
   shared column-index pattern of two CSR rows, identical to what cs_dot does
   for two sparse vectors. The diagonal (a == b) collapses to L[r,:] dotted
   with itself, which is just a sum of squares — no merge needed. */

void cs_block_diag_self_dot(const int *Lp, const int *Li, const double *Lx,
                            int n_rows, int n_cols,
                            int ng, int nv,
                            double *out) {
    int total = nv * nv;
    for (int i = 0; i < total; i++) out[i] = 0.0;

    for (int j = 0; j < ng; j++) {
        for (int a = 0; a < nv; a++) {
            int r1 = j * nv + a;
            int p1s = Lp[r1], p1e = Lp[r1 + 1];

            double d = 0.0;
            for (int p = p1s; p < p1e; p++) d += Lx[p] * Lx[p];
            out[a * nv + a] += d;

            for (int b = a + 1; b < nv; b++) {
                int r2 = j * nv + b;
                int p2s = Lp[r2], p2e = Lp[r2 + 1];

                double s = 0.0;
                int p1 = p1s, p2 = p2s;
                while (p1 < p1e && p2 < p2e) {
                    int i1 = Li[p1], i2 = Li[p2];
                    if (i1 < i2) p1++;
                    else if (i1 > i2) p2++;
                    else { s += Lx[p1] * Lx[p2]; p1++; p2++; }
                }
                out[a * nv + b] += s;
                out[b * nv + a] += s;
            }
        }
    }
}
