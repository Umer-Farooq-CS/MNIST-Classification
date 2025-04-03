#ifndef UTILS_H
#define UTILS_H

#include "nn.h"
#include <cuda_runtime.h>

#define CHECK_BOUNDS(index, max) \
    if (index >= max) { \
        printf("Index %d out of bounds (max %d)\n", index, max); \
        exit(1); \
    }

// Timer functions using CUDA events
void create_timer(cudaEvent_t* start, cudaEvent_t* stop);
void start_timer(cudaEvent_t start);
float stop_timer(cudaEvent_t start, cudaEvent_t stop);

// Matrix operations
double* allocateMatrix(int rows, int cols);
void freeMatrix(double* mat, int rows);

#endif