#ifndef MNIST_H
#define MNIST_H

#include "nn.h"

// MNIST data loading functions
double* loadMNISTImages(const char* filename, int numImages);
double* loadMNISTLabels(const char* filename, int numLabels);

#endif