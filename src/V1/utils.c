#include "utils.h"
#include <stdlib.h>

// Implementation of utility functions

double get_time(clock_t start) {
    return (double)(clock() - start) / CLOCKS_PER_SEC;
}

// Allocate memory for a matrix
double** allocateMatrix(int rows, int cols) {
    if (VERBOSE) printf("Allocating matrix of size %d x %d\n", rows, cols);
    double** mat = (double**)malloc(rows * sizeof(double*));
    if (!mat) {
        if (VERBOSE) printf("Failed to allocate matrix rows\n");
        exit(1);
    }
    
    for (int i = 0; i < rows; i++) {
        mat[i] = (double*)malloc(cols * sizeof(double));
        if (!mat[i]) {
            if (VERBOSE) printf("Failed to allocate matrix columns for row %d\n", i);
            exit(1);
        }
    }
    if (VERBOSE) printf("Matrix allocation successful\n");
    return mat;
}

// Free allocated matrix memory
void freeMatrix(double** mat, int rows) {
    if (VERBOSE) printf("Freeing matrix with %d rows\n", rows);
    for (int i = 0; i < rows; i++) {
        free(mat[i]);
    }
    free(mat);
    if (VERBOSE) printf("Matrix freed successfully\n");
}

