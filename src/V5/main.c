#include "neural_net.h"
#include "mnist.h"
#include "utils.h"
#include "nn.h"  // Added this include
#include <stdio.h>

int main() {
    printf("\nMNIST Neural Network with OpenACC\n\n");

    if (VERBOSE) printf("Loading datasets...\n");

    // Load the MNIST dataset
    double* train_images = loadMNISTImages("../../data/train-images.idx3-ubyte", 60000);
    double* train_labels = loadMNISTLabels("../../data/train-labels.idx1-ubyte", 60000);
    double* test_images = loadMNISTImages("../../data/t10k-images.idx3-ubyte", 10000);
    double* test_labels = loadMNISTLabels("../../data/t10k-labels.idx1-ubyte", 10000);

    if (VERBOSE) printf("Creating neural network...\n");

    NeuralNetwork net;
    initNetwork(&net);

    if (VERBOSE) printf("Starting training...\n");

    // Train the network
    train(&net, train_images, train_labels, 60000);

    if (VERBOSE) printf("Starting evaluation...\n");

    // Evaluate the network
    evaluate(&net, test_images, test_labels, 10000);

    if (VERBOSE) printf("Cleaning up...\n");

    // Clean up network resources
    freeNetwork(&net);
    freeMatrix(train_images, 60000, INPUT_SIZE);
    freeMatrix(train_labels, 60000, OUTPUT_SIZE);
    freeMatrix(test_images, 10000, INPUT_SIZE);
    freeMatrix(test_labels, 10000, OUTPUT_SIZE);

    printf("Program completed successfully\n\n");
    return 0;
}
