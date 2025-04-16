#include "utils.h"
#include <stdlib.h>

double* allocateMatrix(int rows, int cols) {
    if (VERBOSE) printf("Allocating flattened matrix of size %d x %d\n", rows, cols);
    
    // Allocate single contiguous block
    double* mat = (double*)malloc(rows * cols * sizeof(double));
    if (!mat) {
        if (VERBOSE) printf("Failed to allocate matrix\n");
        exit(1);
    }
    
    if (VERBOSE) printf("Flattened matrix allocation successful\n");
    return mat;
}

void freeMatrix(double* mat, int rows) {
    if (VERBOSE) printf("Freeing matrix with %d rows\n", rows);
    free(mat);
    if (VERBOSE) printf("Matrix freed successfully\n");
}