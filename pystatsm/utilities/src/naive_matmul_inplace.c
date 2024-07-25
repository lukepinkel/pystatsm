#include <stddef.h>
#include <stdlib.h>
#include <omp.h>

void naive_matmul_inplace(double *A, double *B, double *C, size_t N, size_t M, size_t K)
 #pragma omp parallel
 {
    #pragma omp for
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < K; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < M; k++) {
                sum += A[i * M + k] * B[k * K + j];
            }
            C[i * K + j] = sum;
        }
    }
}

void naive_matmul_inplace2(double *A, double *B, double *C, size_t N, size_t M, size_t K) 
    #pragma omp parallel
   {
    double *Aptr, *Bptr, *Cptr;
    double *Arow, *Bcol;
    double sum;
    #pragma omp for
    for (size_t i = 0; i < N; i++) {
        Arow = A + i * M;
        Cptr = C + i * K;
        for (size_t j = 0; j < K; j++) {
            sum = 0.0;
            Aptr = Arow;
            Bcol = B + j;
            for (size_t k = 0; k < M; k++) {
                sum += (*Aptr) * (*Bcol);
                Aptr++;
                Bcol += K;
            }
            *Cptr = sum;
            Cptr++;
        }
    }
}
