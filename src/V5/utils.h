#ifndef UTILS_H
#define UTILS_H

#include "nn.h"
#include <openacc.h>  // Add OpenACC header

#define CHECK_BOUNDS(index, max) \
    if (index >= max) { \
        printf("Index %d out of bounds (max %d)\n", index, max); \
        exit(1); \
    }

    
    // Remove CUDA-specific functions
    // Keep matrix operations
    
    double* allocateMatrix(int rows, int cols);
    void freeMatrix(double* mat, int rows);
    
#endif