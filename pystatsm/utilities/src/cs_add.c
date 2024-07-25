#include "cs_add.h"
#include <stdlib.h>

void cs_add(const int *Ap, const int *Ai, const double *Ax,
            const int *Bp, const int *Bi, const double *Bx,
            double alpha, double beta,
            int *Cp, int *Ci, double *Cx, int Cnr, int Cnc) {
    double *x = (double *)calloc(Cnr, sizeof(double));
    
    for (int j = 0; j < Cnc; j++) {
        for (int p = Ap[j]; p < Ap[j+1]; p++) {
            int i = Ai[p];
            x[i] += alpha * Ax[p];
        }
        for (int p = Bp[j]; p < Bp[j+1]; p++) {
            int i = Bi[p];
            x[i] += beta * Bx[p];
        }
        for (int p = Cp[j]; p < Cp[j+1]; p++) {
            Cx[p] = x[Ci[p]];
            x[Ci[p]] = 0.0;
        }
    }
    
    free(x);
}