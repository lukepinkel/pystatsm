#include "cs_kron_id_sp.h"

void cs_kron_id_sp(int m, const double *Bx, const int *Bi, const int *Bp,
                   int Bnr, int Bnc, double *Cx, int *Ci, int *Cp) {
    int cnt = 0;
    Cp[0] = 0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < Bnc; j++) {
            for (int k = Bp[j]; k < Bp[j+1]; k++) {
                Cx[cnt] = Bx[k];
                Ci[cnt] = Bi[k] + i * Bnr;
                cnt++;
            }
            Cp[i * Bnc + j + 1] = cnt;
        }
    }
}

void cs_kron_id_sp_inplace(int m, const double *Bx, const int *Bi, const int *Bp,
                           int Bnr, int Bnc, double *Cx, int *Ci, int *Cp) {
    int cnt = 0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < Bnc; j++) {
            for (int k = Bp[j]; k < Bp[j+1]; k++) {
                Cx[cnt] = Bx[k];
                cnt++;
            }
        }
    }
}