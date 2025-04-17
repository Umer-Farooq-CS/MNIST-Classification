#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include "nn.h"  // Include nn.h first to get all constants

typedef struct {
    double *W1, *b1, *W2, *b2;
} NeuralNetwork;

// Activation functions
double sigmoid(double x);
double sigmoid_derivative(double x);
void relu(double* x, int size);
void softmax(double* x, int size);

// Neural network operations
void initWeights(double* W, int n, double scale, unsigned long seed);
void matrixVectorMultiply(double* W, double* x, double* b, double* y, int rows, int cols);
void forwardPass(NeuralNetwork* net, double* input, double* hidden, double* output);
void backward(NeuralNetwork* net, double* input, double* hidden, double* output, double* target);
void train(NeuralNetwork* net, double* images, double* labels, int numImages);
void evaluate(NeuralNetwork* net, double* images, double* labels, int numImages);
void initNetwork(NeuralNetwork* net);
void freeNetwork(NeuralNetwork* net);

#endif