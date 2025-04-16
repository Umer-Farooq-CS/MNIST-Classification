#ifndef MNIST_H
#define MNIST_H

#include "nn.h"

// MNIST data loading functions
float* loadMNISTImages(const char* filename, int numImages);
float* loadMNISTLabels(const char* filename, int numLabels);

#endif