#include "neural_net.h"
#include "mnist.h"
#include "utils.h"
#include "nn.h"

int main() {
    printf("\nMNIST Neural Network\n\n");

    if (VERBOSE) printf("Loading datasets...\n");
    double** train_images = loadMNISTImages("../../data/train-images.idx3-ubyte", 60000);
    double** train_labels = loadMNISTLabels("../../data/train-labels.idx1-ubyte", 60000);
    double** test_images = loadMNISTImages("../../data/t10k-images.idx3-ubyte", 10000);
    double** test_labels = loadMNISTLabels("../../data/t10k-labels.idx1-ubyte", 10000);

    if (VERBOSE) printf("Creating neural network...\n");
    NeuralNetwork* net = createNetwork();
    
    if (VERBOSE) printf("Starting training...\n");
    train(net, train_images, train_labels, 60000);
    
    if (VERBOSE) printf("Starting evaluation...\n");
    evaluate(net, test_images, test_labels, 10000);

    if (VERBOSE) printf("Cleaning up...\n");
    freeNetwork(net);
    
    printf("Program completed successfully\n\n");
    return 0;
}