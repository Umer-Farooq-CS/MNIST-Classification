#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include "nn.h"

typedef struct {
    double* W1;  // Flattened [HIDDEN_SIZE * INPUT_SIZE]
    double* W2;  // Flattened [OUTPUT_SIZE * HIDDEN_SIZE]
    double* b1;
    double* b2;
    // GPU copies
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
void backward(NeuralNetwork* net, double* input, double* hidden, double* output, double* target);
void train(NeuralNetwork* net, double* images, double* labels, int numImages);
void evaluate(NeuralNetwork* net, double* images, double* labels, int numImages);

// Activation functions (CUDA versions)
__global__ void matrixVectorMultiply(double* W, double* x, double* b, double* result, int rows, int cols);
__global__ void relu_kernel(double* x, int size);
__global__ void softmax_kernel(double* x, int size);

#endif