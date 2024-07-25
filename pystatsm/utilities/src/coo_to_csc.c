#include "coo_to_csc.h"

void coo_to_csc(int Anr, int Anc, int Anz, const int *Ai, const int *Aj, const double *Ax,
                int *Bp, int *Bi, double *Bx) {
    int i, j, k, c, r, bpj;

    for (i = 0; i < Anz; i++) {
        Bp[Aj[i]]++;
    }

    c = 0;
    for (j = 0; j < Anc; j++) {
        bpj = Bp[j];
        Bp[j] = c;
        c += bpj;
    }
    Bp[Anc] = c;

    for (k = 0; k < Anz; k++) {
        c = Aj[k];
        Bi[Bp[c]] = Ai[k];
        Bx[Bp[c]] = Ax[k];
        Bp[c]++;
    }

    r = 0;
    for (j = 0; j < Anc + 1; j++) {
        bpj = Bp[j];
        Bp[j] = r;
        r = bpj;
    }
}