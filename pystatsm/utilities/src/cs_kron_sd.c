#include "cs_kron_sd.h"

void cs_kron_sd(const int *Ap, const int *Ai, const double *Ax, int Anr, int Anc,
                const double *B, int Bnr, int Bnc,
                int *Cp, int *Ci, double *Cx) {
    int a_col, b_col, a_i, b_row, cnt = 0;
    
    for (a_col = 0; a_col < Anc; a_col++) {
        for (b_col = 0; b_col < Bnc; b_col++) {
            for (a_i = Ap[a_col]; a_i < Ap[a_col + 1]; a_i++) {
                int a_row = Ai[a_i];
                double ax = Ax[a_i];
                for (b_row = 0; b_row < Bnr; b_row++) {
                    Cx[cnt] = ax * B[b_row + b_col * Bnr];
                    Ci[cnt] = b_row + a_row * Bnr;
                    cnt++;
                }
            }
            Cp[a_col * Bnc + b_col + 1] = cnt;
        }
    }
}

void cs_kron_sd_inplace(const int *Ap, const int *Ai, const double *Ax, int Anr, int Anc,
                        const double *B, int Bnr, int Bnc,
                        double *Cx) {
    int a_col, b_col, a_i, b_row, cnt = 0;
    
    for (a_col = 0; a_col < Anc; a_col++) {
        for (b_col = 0; b_col < Bnc; b_col++) {
            for (a_i = Ap[a_col]; a_i < Ap[a_col + 1]; a_i++) {
                double ax = Ax[a_i];
                for (b_row = 0; b_row < Bnr; b_row++) {
                    Cx[cnt] = ax * B[b_row + b_col * Bnr];
                    cnt++;
                }
            }
        }
    }
}