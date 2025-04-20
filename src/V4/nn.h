#ifndef NN_H
#define NN_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.01f  // Changed to float
#define EPOCHS 3
#define BATCH_SIZE 64        // Should be multiple of 8 for Tensor Cores
#define NUM_CLASSES 10
#define VERBOSE 0

// Tensor core tile sizes
#define TC_TILE_M 16
#define TC_TILE_N 16
#define TC_TILE_K 16

#endif