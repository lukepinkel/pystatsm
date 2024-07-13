#include "repeat_1d.h"

void repeat_1d(const double *arr, int n, int reps, double *out) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < reps; j++) {
            out[i * reps + j] = arr[i];
        }
    }
}