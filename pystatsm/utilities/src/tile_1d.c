#include "tile_1d.h"

void tile_1d_modulo(const double *arr, int n, int reps, double *out) {
    int total = n * reps;
    for (int i = 0; i < total; i++) {
        out[i] = arr[i % n];
    }
}

void tile_1d_nested(const double *arr, int n, int reps, double *out) {
    for (int i = 0; i < reps; i++) {
        for (int j = 0; j < n; j++) {
            out[i * n + j] = arr[j];
        }
    }
}
