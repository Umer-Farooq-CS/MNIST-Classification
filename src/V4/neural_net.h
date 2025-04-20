#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include "nn.h"
#include <stdlib.h>
#include <time.h>
#include <curand_kernel.h>

// Define block size for shared memory kernels
#define BLOCK_SIZE 128

// Structure holding host and device copies of weights and biases
typedef struct {
    // Master weights (FP32)
    float* W1;  // Flattened weight matrix for layer 1 [HIDDEN_SIZE x INPUT_SIZE]
    float* W2;  // Flattened weight matrix for layer 2 [OUTPUT_SIZE x HIDDEN_SIZE]
    float* b1;  // Biases for layer 1 [HIDDEN_SIZE]
    float* b2;  // Biases for layer 2 [OUTPUT_SIZE]
    
    // Device (GPU) copies (FP16 for computation, FP32 for storage)
    float* d_W1;
    float* d_W2;
    float* d_b1;
    float* d_b2;
    half* d_W1_half;   // FP16 copy for Tensor Core ops
    half* d_W2_half;   // FP16 copy for Tensor Core ops
    half* d_b1_half;   // FP16 copy for Tensor Core ops
    half* d_b2_half;   // FP16 copy for Tensor Core ops
    half* d_input;
    half* d_hidden;
    half* d_output;
} NeuralNetwork;

// Neural network functions
NeuralNetwork* createNetwork();
void freeNetwork(NeuralNetwork* net);
void forward(NeuralNetwork* net, float* input, float* hidden, float* output);
void backward(NeuralNetwork* net, half* d_input, half* d_target);
void train(NeuralNetwork* net, float* images, float* labels, int numImages);
void evaluate(NeuralNetwork* net, float* images, float* labels, int numImages);

// Tensor Core optimized kernels
__global__ void tc_matrixVectorMultiply(const half* W, const half* x, const half* b, float* y, int rows, int cols);
__global__ void relu_kernel(half* x, int size);
__global__ void softmaxKernelOpt(float* x, int size);

// Backward pass kernels
__global__ void computeDOutputKernel(float* d_output, const half* d_target, int outputSize);
__global__ void computeDHiddenKernel(const half* d_W2, const float* d_output, 
                                   const half* d_hidden_forward, float* d_hidden_grad,
                                   int hiddenSize, int outputSize);
__global__ void updateW2Kernel(half* d_W2, const float* d_output, 
                             const half* d_hidden_forward, int hiddenSize, 
                             int outputSize, float learning_rate);
__global__ void updateW1Kernel(half* d_W1, const float* d_hidden_grad, 
                             const half* d_input, int inputSize, int hiddenSize, 
                             float learning_rate);
__global__ void updateBiasesKernel(float* d_bias, const float* d_grad, int size, float learning_rate);

// Conversion kernels
__global__ void floatToHalfKernel(const float* in, half* out, int size);
__global__ void halfToFloatKernel(const half* in, float* out, int size);

#endif