#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include "nn.h"
#include <stdlib.h>
#include <time.h>
#include <openacc.h>  // Add OpenACC header

// Remove CUDA-specific kernel declarations
// Keep the NeuralNetwork structure but remove device pointers
typedef struct {
    double* W1;  // Flattened weight matrix for layer 1 [HIDDEN_SIZE x INPUT_SIZE]
    double* W2;  // Flattened weight matrix for layer 2 [OUTPUT_SIZE x HIDDEN_SIZE]
    double* b1;  // Biases for layer 1 [HIDDEN_SIZE]
    double* b2;  // Biases for layer 2 [OUTPUT_SIZE]
} NeuralNetwork;

// Neural network functions
NeuralNetwork* createNetwork();
void freeNetwork(NeuralNetwork* net);
void forward(NeuralNetwork* net, double* input, double* hidden, double* output);
void backward(NeuralNetwork* net, double* input, double* hidden, double* output, double* target);
void train(NeuralNetwork* net, double* images, double* labels, int numImages);
void evaluate(NeuralNetwork* net, double* images, double* labels, int numImages);

#endif