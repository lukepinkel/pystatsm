#include "cs_pattern_trace.h"

void cs_pattern_trace(const int *Bp, const int *Bi, const double *Bx, int Bnr, int Bnc, int Bnz,
                      const int *Cp, const int *Ci, const double *Cx, int Cnr, int Cnc, int Cnz,
                      double *trace) {

    for (int i = 0; i < Cnc; i++) {
        for (int k = Cp[i]; k < Cp[i+1]; k++) {
            int j = Ci[k];
            double dot_product = 0.0;

            int p1 = Bp[i], p2 = Bp[j];
            while (p1 < Bp[i+1] && p2 < Bp[j+1]) {
                if (Bi[p1] < Bi[p2])
                    p1++;
                else if (Bi[p1] > Bi[p2])
                    p2++;
                else {
                    dot_product += Bx[p1] * Bx[p2];
                    p1++;
                    p2++;
                }
            }

            *trace += Cx[k] * dot_product;
        }
    }
}