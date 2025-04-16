#ifndef UTILS_H
#define UTILS_H

#include "nn.h"
#include <cuda_runtime.h>

// ... existing code ...
// CUDA error checking
inline void checkCudaError(cudaError_t err, const char* context) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", context, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

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


// Profiling macros
#define PROFILE_START(name) \
    cudaEvent_t start_##name, stop_##name; \
    float milliseconds_##name = 0; \
    cudaEventCreate(&start_##name); \
    cudaEventCreate(&stop_##name); \
    cudaEventRecord(start_##name);

#define PROFILE_STOP(name) \
    cudaEventRecord(stop_##name); \
    cudaEventSynchronize(stop_##name); \
    cudaEventElapsedTime(&milliseconds_##name, start_##name, stop_##name); \
    cudaEventDestroy(start_##name); \
    cudaEventDestroy(stop_##name); \
    printf("[PROFILE] %s: %.3f ms\n", #name, milliseconds_##name);

#endif