#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include "nn.h"   // This file must define HIDDEN_SIZE, INPUT_SIZE, OUTPUT_SIZE, LEARNING_RATE, EPOCHS, etc.
#include <stdlib.h>
#include <time.h>
#include <curand_kernel.h>

// Define a block size for shared memory kernels.
#define BLOCK_SIZE 128

// Structure holding host and device copies of weights and biases.
typedef struct {
    double* W1;  // Flattened weight matrix for layer 1 [HIDDEN_SIZE x INPUT_SIZE]
    double* W2;  // Flattened weight matrix for layer 2 [OUTPUT_SIZE x HIDDEN_SIZE]
    double* b1;  // Biases for layer 1 [HIDDEN_SIZE]
    double* b2;  // Biases for layer 2 [OUTPUT_SIZE]
    // Device (GPU) copies.
    double* d_W1;
    double* d_W2;
    double* d_b1;
    double* d_b2;
    double* d_input;
    double* d_hidden;
    double* d_output;
} NeuralNetwork;

// Neural network functions
NeuralNetwork* createNetwork();
void freeNetwork(NeuralNetwork* net);
void forward(NeuralNetwork* net, double* input, double* hidden, double* output);
// GPU backward pass: note that here the "input" and "target" are expected to be device pointers.
void backward(NeuralNetwork* net, double* d_input, double* d_target);
void train(NeuralNetwork* net, double* images, double* labels, int numImages);
void evaluate(NeuralNetwork* net, double* images, double* labels, int numImages);

// CUDA kernel declarations.

// Optimized matrix–vector multiplication kernel that uses shared memory to perform in–block reduction.
// Each block computes one output row.
__global__ void matrixVectorMultiplySM(const double* __restrict__ W, 
                                         const double* __restrict__ x, 
                                         const double* __restrict__ b, 
                                         double* __restrict__ y, 
                                         int rows, int cols);

// ReLU activation kernel (unchanged).
__global__ void relu_kernel(double* x, int size);

// Optimized softmax kernel using shared memory for reduction and numerical stability.
__global__ void softmaxKernelOpt(double* x, int size);

// Kernels used in the GPU backward pass.
__global__ void computeDOutputKernel(double* d_output, const double* d_target, int outputSize);
__global__ void computeDHiddenKernel(const double* d_W2, const double* d_output, 
                                       const double* d_hidden_forward, double* d_hidden_grad,
                                       int hiddenSize, int outputSize);
__global__ void updateW2Kernel(double* d_W2, const double* d_output, 
                               const double* d_hidden_forward, int hiddenSize, 
                               int outputSize, double learning_rate);
__global__ void updateW1Kernel(double* d_W1, const double* d_hidden_grad, 
                               const double* d_input, int inputSize, int hiddenSize, 
                               double learning_rate);
__global__ void updateBiasesKernel(double* d_bias, const double* d_grad, int size, double learning_rate);

#endif