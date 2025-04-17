#ifndef NN_H
#define NN_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <openacc.h>

// Network architecture constants
#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10
#define NUM_CLASSES 10

// Training parameters
#define LEARNING_RATE 0.01
#define EPOCHS 3          // Consolidated definition (using 5 as the final value)
#define BATCH_SIZE 64
#define TRAINING_SIZE 60000

// Debugging
#define VERBOSE 0

// Function prototypes
double* loadMNISTImages(const char* filename, int num_images);
double* loadMNISTLabels(const char* filename, int num_labels);

#endif