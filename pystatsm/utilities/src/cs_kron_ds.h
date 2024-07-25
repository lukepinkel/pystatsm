void cs_kron_ds(const double *A, int Anr, int Anc,
                const int *Bp, const int *Bi, const double *Bx, int Bnr, int Bnc,
                int *Cp, int *Ci, double *Cx);
void cs_kron_ds_inplace(const double *A, int Anr, int Anc,
                        const int *Bp, const int *Bi, const double *Bx, int Bnr, int Bnc,
                        double *Cx);
            
            