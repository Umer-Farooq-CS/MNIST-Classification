#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include "nn.h"

// Neural network structure
typedef struct {
    // Host (CPU) memory
    double* W1;      // Flattened weights input->hidden [HIDDEN_SIZE * INPUT_SIZE]
    double* W2;      // Flattened weights hidden->output [OUTPUT_SIZE * HIDDEN_SIZE]
    double* b1;      // Biases for hidden layer [HIDDEN_SIZE]
    double* b2;      // Biases for output layer [OUTPUT_SIZE]
    
    // Device (GPU) memory
    double* d_W1;    // Device copy of W1
    double* d_W2;    // Device copy of W2
    double* d_b1;    // Device copy of b1
    double* d_b2;    // Device copy of b2
    
    // Temporary buffers (device)
    double* d_input;   // [INPUT_SIZE]
    double* d_hidden;  // [HIDDEN_SIZE]
    double* d_output;  // [OUTPUT_SIZE]
    
    // Additional buffers for training
    double* d_d_output;  // Output gradients [OUTPUT_SIZE]
    double* d_d_hidden;  // Hidden gradients [HIDDEN_SIZE]
} NeuralNetwork;

// Neural network functions
NeuralNetwork* createNetwork();
void freeNetwork(NeuralNetwork* net);
void forwardGPU(NeuralNetwork* net, double* input, double* output);
void backwardGPU(NeuralNetwork* net, double* input, double* target);
void train(NeuralNetwork* net, double* images, double* labels, int numImages);
void evaluate(NeuralNetwork* net, double* images, double* labels, int numImages);

// Activation functions
__global__ void reluKernel(double* x, int size);
__global__ void softmaxKernel(double* x, int size);

#endif