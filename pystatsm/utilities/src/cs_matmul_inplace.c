#include "cs_matmul_inplace.h"
#include <stdlib.h>

void cs_matmul_inplace(const int *Ap, const int *Ai, const double *Ax, int Anr, int Anc,
                       const int *Bp, const int *Bi, const double *Bx, int Bnr, int Bnc,
                       int *Cp, int *Ci, double *Cx) {
    int *w = (int *)malloc(Anr * sizeof(int));
    double *x = (double *)malloc(Anr * sizeof(double));
    int nz = 0;

    for (int i = 0; i < Anr; i++) {
        w[i] = -1;
        x[i] = 0.0;
    }

    for (int j = 0; j < Bnc; j++) {
        Cp[j] = nz;
        for (int i = Bp[j]; i < Bp[j+1]; i++) {
            double b = Bx[i];
            int jj = Bi[i];
            for (int k = Ap[jj]; k < Ap[jj+1]; k++) {
                int ii = Ai[k];
                if (w[ii] < j) {
                    w[ii] = j;
                    Ci[nz] = ii;
                    x[ii] = b * Ax[k];
                    nz++;
                } else {
                    x[ii] += b * Ax[k];
                }
            }
        }
        for (int i = Cp[j]; i < nz; i++) {
            Cx[i] = x[Ci[i]];
        }
    }
    Cp[Bnc] = nz;

    free(w);
    free(x);
}