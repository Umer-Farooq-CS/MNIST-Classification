#include "utils.h"
#include <stdlib.h>
#include <stdio.h>

double* allocateMatrix(int rows, int cols) {
    if (VERBOSE) printf("Allocating flattened matrix of size %d x %d\n", rows, cols);
    
    double* mat = (double*)malloc(rows * cols * sizeof(double));
    if (!mat) {
        if (VERBOSE) printf("Failed to allocate matrix\n");
        exit(1);
    }

    #pragma acc enter data copyin(mat[0:rows * cols])

    if (VERBOSE) printf("Flattened matrix allocation successful\n");
    return mat;
}

void freeMatrix(double* mat, int rows, int cols) {  // Added cols parameter
    if (VERBOSE) printf("Freeing matrix with %d rows\n", rows);

    #pragma acc exit data delete(mat[0:rows * cols])  // Use cols instead of assuming square
    free(mat);

    if (VERBOSE) printf("Matrix freed successfully\n");
}