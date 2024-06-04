#include "cs_kron.h"

void cs_kron(const int *Ap, const int *Ai, const double *Ax, int Anr, int Anc,
             const int *Bp, const int *Bi, const double *Bx, int Bnr, int Bnc,
             int *Cp, int *Ci, double *Cx) {
    int a_col, b_col, a_i, b_i, rout, cnt = 0;
    
    for (a_col = 0; a_col < Anc; a_col++) {
        for (b_col = 0; b_col < Bnc; b_col++) {
            for (a_i = Ap[a_col]; a_i < Ap[a_col + 1]; a_i++) {
                rout = Ai[a_i] * Bnr;
                double ax = Ax[a_i];
                for (b_i = Bp[b_col]; b_i < Bp[b_col + 1]; b_i++) {
                    Cx[cnt] = ax * Bx[b_i];
                    Ci[cnt] = Bi[b_i] + rout;
                    cnt++;
                }
            }
            Cp[a_col * Bnc + b_col + 1] = cnt;
        }
    }
}