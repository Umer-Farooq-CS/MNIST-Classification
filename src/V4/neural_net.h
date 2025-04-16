#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include "nn.h"
#include <cublas_v2.h>
#include <cuda_fp16.h>

#define BLOCK_SIZE 128
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

typedef struct {
    half* W1;   // FP16 weights
    half* W2;
    float* b1;  // FP32 biases
    float* b2;
    // Device copies
    half* d_W1;
    half* d_W2;
    float* d_b1;
    float* d_b2;
    half* d_input;  // FP16 input
    float* d_hidden;
    float* d_output;
} NeuralNetwork;

// Tensor Core-optimized forward pass
void forward_batch(NeuralNetwork* net, half* d_input_batch, int batch_size);

// WMMA matrix multiplication kernel
__global__ void wmma_matmul(half* A, half* B, float* C, int M, int N, int K);

// Neural network functions
NeuralNetwork* createNetwork();
void freeNetwork(NeuralNetwork* net);
void forward(NeuralNetwork* net, float* input, float* hidden, float* output);
// GPU backward pass: note that here the "input" and "target" are expected to be device pointers.
void backward(NeuralNetwork* net, float* d_input, float* d_target);
void train(NeuralNetwork* net, float* images, float* labels, int numImages);
void evaluate(NeuralNetwork* net, float* images, float* labels, int numImages);

// CUDA kernel declarations.

// Optimized matrix–vector multiplication kernel that uses shared memory to perform in–block reduction.
// Each block computes one output row.
__global__ void matrixVectorMultiplySM(const float* __restrict__ W, 
                                         const float* __restrict__ x, 
                                         const float* __restrict__ b, 
                                         float* __restrict__ y, 
                                         int rows, int cols);

// ReLU activation kernel (unchanged).
__global__ void relu_kernel(float* x, int size);

// Optimized softmax kernel using shared memory for reduction and numerical stability.
__global__ void softmaxKernelOpt(float* x, int size);

// Kernels used in the GPU backward pass.
__global__ void computeDOutputKernel(float* d_output, const float* d_target, int outputSize);
__global__ void computeDHiddenKernel(const float* d_W2, const float* d_output, 
                                       const float* d_hidden_forward, float* d_hidden_grad,
                                       int hiddenSize, int outputSize);
__global__ void updateW2Kernel(float* d_W2, const float* d_output, 
                               const float* d_hidden_forward, int hiddenSize, 
                               int outputSize, float learning_rate);
__global__ void updateW1Kernel(float* d_W1, const float* d_hidden_grad, 
                               const float* d_input, int inputSize, int hiddenSize, 
                               float learning_rate);
__global__ void updateBiasesKernel(float* d_bias, const float* d_grad, int size, float learning_rate);

#endif
