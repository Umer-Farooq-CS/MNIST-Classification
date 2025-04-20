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
    // Host-side weights and biases (original float32 or float16)
    half* W1;    // Host weights FP16
    half* W2;
    float* b1;   // Host biases FP32
    float* b2;

    // Device-side weights and biases (FP32 for CPU-trained versions)
    half* d_W1;  // FP16
    half* d_W2;
    float* d_b1;
    float* d_b2;

    // Tensor Core-optimized copies (required in many kernels)
    half* d_W1_half;
    half* d_W2_half;
    half* d_b1_half;
    half* d_b2_half;

    // Input/output buffers
    half* d_input;       // FP16 input batch
    float* d_hidden;     // FP32 hidden layer
    float* d_output;     // FP32 output layer

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