#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include "nn.h"

// Neural network structure
typedef struct {
    double** W1;
    double** W2;
    double* b1;
    double* b2;
} NeuralNetwork;

// Neural network functions
NeuralNetwork* createNetwork();
void freeNetwork(NeuralNetwork* net);
void forward(NeuralNetwork* net, double* input, double* hidden, double* output);
void backward(NeuralNetwork* net, double* input, double* hidden, double* output, double* target);
void train(NeuralNetwork* net, double** images, double** labels, int numImages);
void evaluate(NeuralNetwork* net, double** images, double** labels, int numImages);

// Activation functions
void relu(double* x, int size);
void softmax(double* x, int size);

#endif