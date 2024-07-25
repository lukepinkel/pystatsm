#include "cs_add_inplace_complex.h"
#include <stdlib.h>
#include <complex.h>

void cs_add_inplace_complex(const int *Ap, const int *Ai, const double _Complex *Ax,
                            const int *Bp, const int *Bi, const double _Complex *Bx,
                            double _Complex alpha, double _Complex beta,
                            int *Cp, int *Ci, double _Complex *Cx, int Cnr, int Cnc) {
    double _Complex *x = (double _Complex *)calloc(Cnr, sizeof(double _Complex));
    
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
            x[Ci[p]] = 0.0 + 0.0 * I;  // Set to complex zero
        }
    }
    
    free(x);
}