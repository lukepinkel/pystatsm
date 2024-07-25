#include "tile_1d_complex.h"
#include <complex.h>

void tile_1d_nested_complex(const double _Complex *arr, int n, int reps, double _Complex *out) {
    for (int i = 0; i < reps; i++) {
        for (int j = 0; j < n; j++) {
            out[i * n + j] = arr[j];
        }
    }
}