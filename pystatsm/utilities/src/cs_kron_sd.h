void cs_kron_sd(const int *Ap, const int *Ai, const double *Ax, int Anr, int Anc,
                const double *B, int Bnr, int Bnc,
                int *Cp, int *Ci, double *Cx);

void cs_kron_sd_inplace(const int *Ap, const int *Ai, const double *Ax, int Anr, int Anc,
                        const double *B, int Bnr, int Bnc,
                        double *Cx);
