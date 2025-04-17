#ifndef UTILS_H
#define UTILS_H

#include "nn.h"
#include <openacc.h>

#define CHECK_BOUNDS(index, max) \
    if (index >= max) { \
        printf("Index %d out of bounds (max %d)\n", index, max); \
        exit(1); \
    }

double* allocateMatrix(int rows, int cols);
void freeMatrix(double* mat, int rows, int cols);  // Fixed to match implementation
    
#endif