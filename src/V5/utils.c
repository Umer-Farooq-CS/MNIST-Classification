#include "utils.h"
#include <stdlib.h>
#include <stdio.h>

double* allocateMatrix(int rows, int cols) {
    if (VERBOSE) printf("Allocating flattened matrix of size %d x %d\n", rows, cols);
    
    // Using acc_malloc for device-side allocation
    double* mat = (double*)acc_malloc(rows * cols * sizeof(double));
    if (!mat) {
        if (VERBOSE) printf("Failed to allocate matrix\n");
        exit(1);
    }

    if (VERBOSE) printf("Flattened matrix allocation successful\n");
    return mat;
}

void freeMatrix(double* mat, int rows, int cols) {
    if (VERBOSE) printf("Freeing matrix with %d rows\n", rows);

    // Use acc_free for device memory
    acc_free(mat);

    if (VERBOSE) printf("Matrix freed successfully\n");
}