#include "cs_kron_ds.h"

void cs_kron_ds(const double *A, int Anr, int Anc,
                const int *Bp, const int *Bi, const double *Bx, int Bnr, int Bnc,
                int *Cp, int *Ci, double *Cx) {
    int a_row, a_col, b_col, b_i;
    int cnt = 0;
    
    for (a_col = 0; a_col < Anc; a_col++) {
        for (b_col = 0; b_col < Bnc; b_col++) {
            for (a_row = 0; a_row < Anr; a_row++) {
                double a_val = A[a_row + a_col * Anr];
                for (b_i = Bp[b_col]; b_i < Bp[b_col + 1]; b_i++) {
                    Cx[cnt] = a_val * Bx[b_i];
                    Ci[cnt] = Bi[b_i] + a_row * Bnr;
                    cnt++;
                }
            }
            Cp[a_col * Bnc + b_col + 1] = cnt;
        }
    }
}

void cs_kron_ds_inplace(const double *A, int Anr, int Anc,
                        const int *Bp, const int *Bi, const double *Bx, int Bnr, int Bnc,
                        double *Cx) {
    int a_row, a_col, b_col, b_i;
    int cnt = 0;
    
    for (a_col = 0; a_col < Anc; a_col++) {
        for (b_col = 0; b_col < Bnc; b_col++) {
            for (a_row = 0; a_row < Anr; a_row++) {
                double a_val = A[a_row + a_col * Anr];
                for (b_i = Bp[b_col]; b_i < Bp[b_col + 1]; b_i++) {
                    Cx[cnt] = a_val * Bx[b_i];
                    cnt++;
                }
            }
        }
    }
}