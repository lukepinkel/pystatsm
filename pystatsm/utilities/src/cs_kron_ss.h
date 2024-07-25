void cs_kron_ss(const int *Ap, const int *Ai, const double *Ax, int Anr, int Anc,
                const int *Bp, const int *Bi, const double *Bx, int Bnr, int Bnc,
                int *Cp, int *Ci, double *Cx);
void cs_kron_ss_inplace(const int *Ap, const int *Ai, const double *Ax, int Anr, int Anc,
                        const int *Bp, const int *Bi, const double *Bx, int Bnr, int Bnc,
                        double *Cx);