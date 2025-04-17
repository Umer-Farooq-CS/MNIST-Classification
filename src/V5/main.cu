#include "neural_net.h"
#include "mnist.h"
#include "utils.h"
#include "nn.h"
#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    printf("\nMNIST Neural Network\n\n");

    if (VERBOSE) printf("Loading datasets...\n");

    // Load the MNIST dataset (host memory)
    double* train_images = loadMNISTImages("../../data/train-images.idx3-ubyte", 60000);
    double* train_labels = loadMNISTLabels("../../data/train-labels.idx1-ubyte", 60000);
    double* test_images = loadMNISTImages("../../data/t10k-images.idx3-ubyte", 10000);
    double* test_labels = loadMNISTLabels("../../data/t10k-labels.idx1-ubyte", 10000);

    // Allocate device memory for images and labels
    double* d_train_images;
    double* d_train_labels;
    double* d_test_images;
    double* d_test_labels;

    cudaMalloc((void**)&d_train_images, sizeof(double) * 60000 * INPUT_SIZE);
    cudaMalloc((void**)&d_train_labels, sizeof(double) * 60000);
    cudaMalloc((void**)&d_test_images, sizeof(double) * 10000 * INPUT_SIZE);
    cudaMalloc((void**)&d_test_labels, sizeof(double) * 10000);

    // Copy data from host to device
    cudaMemcpy(d_train_images, train_images, sizeof(double) * 60000 * INPUT_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_train_labels, train_labels, sizeof(double) * 60000, cudaMemcpyHostToDevice);
    cudaMemcpy(d_test_images, test_images, sizeof(double) * 10000 * INPUT_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_test_labels, test_labels, sizeof(double) * 10000, cudaMemcpyHostToDevice);

    if (VERBOSE) printf("Creating neural network...\n");

    NeuralNetwork* net = createNetwork();

    if (VERBOSE) printf("Starting training...\n");

    // Train the network (using device memory for images and labels)
    train(net, d_train_images, d_train_labels, 60000);

    if (VERBOSE) printf("Starting evaluation...\n");

    // Evaluate the network
    evaluate(net, d_test_images, d_test_labels, 10000);

    if (VERBOSE) printf("Cleaning up...\n");

    // Clean up network resources
    freeNetwork(net);

    printf("Program completed successfully\n\n");
    return 0;
}
