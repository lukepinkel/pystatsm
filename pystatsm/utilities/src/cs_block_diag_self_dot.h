#ifndef CS_BLOCK_DIAG_SELF_DOT_H
#define CS_BLOCK_DIAG_SELF_DOT_H

void cs_block_diag_self_dot(const int *Lp, const int *Li, const double *Lx,
                            int n_rows, int n_cols,
                            int ng, int nv,
                            double *out);

#endif
