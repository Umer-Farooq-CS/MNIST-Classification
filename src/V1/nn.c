#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.01
#define EPOCHS 3
#define BATCH_SIZE 64
#define NUM_CLASSES 10  // Digits 0-9
#define VERBOSE 0     // Set to 1 to enable debugging prints

// Timer function
double get_time(clock_t start) {
    return (double)(clock() - start) / CLOCKS_PER_SEC;
}

// Allocate memory for a matrix
double** allocateMatrix(int rows, int cols) {
    if (VERBOSE) printf("Allocating matrix of size %d x %d\n", rows, cols);
    double** mat = (double**)malloc(rows * sizeof(double*));
    if (!mat) {
        if (VERBOSE) printf("Failed to allocate matrix rows\n");
        exit(1);
    }
    
    for (int i = 0; i < rows; i++) {
        mat[i] = (double*)malloc(cols * sizeof(double));
        if (!mat[i]) {
            if (VERBOSE) printf("Failed to allocate matrix columns for row %d\n", i);
            exit(1);
        }
    }
    if (VERBOSE) printf("Matrix allocation successful\n");
    return mat;
}

// Free allocated matrix memory
void freeMatrix(double** mat, int rows) {
    if (VERBOSE) printf("Freeing matrix with %d rows\n", rows);
    for (int i = 0; i < rows; i++) {
        free(mat[i]);
    }
    free(mat);
    if (VERBOSE) printf("Matrix freed successfully\n");
}

// Activation functions
void relu(double* x, int size) {
    if (VERBOSE) printf("Applying ReLU to vector of size %d\n", size);
    for (int i = 0; i < size; i++) {
        x[i] = (x[i] > 0) ? x[i] : 0;
    }
    if (VERBOSE) {
        printf("ReLU output (first 5 elements): ");
        for (int i = 0; i < 5 && i < size; i++) printf("%.4f ", x[i]);
        printf("\n");
    }
}

void softmax(double* x, int size) {
    if (VERBOSE) printf("Applying softmax to vector of size %d\n", size);
    double sum = 0;
    for (int i = 0; i < size; i++) {
        x[i] = exp(x[i]);
        sum += x[i];
    }
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
    if (VERBOSE) {
        printf("Softmax output: ");
        for (int i = 0; i < size; i++) printf("%.4f ", x[i]);
        printf("\n");
    }
}

// Neural network structure
typedef struct {
    double** W1;
    double** W2;
    double* b1;
    double* b2;
} NeuralNetwork;

// Initialize neural network
NeuralNetwork* createNetwork() {
    if (VERBOSE) printf("Creating neural network...\n");
    NeuralNetwork* net = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    if (!net) {
        if (VERBOSE) printf("Failed to allocate neural network\n");
        exit(1);
    }
    
    net->W1 = allocateMatrix(HIDDEN_SIZE, INPUT_SIZE);
    net->W2 = allocateMatrix(OUTPUT_SIZE, HIDDEN_SIZE);
    net->b1 = (double*)calloc(HIDDEN_SIZE, sizeof(double));
    net->b2 = (double*)calloc(OUTPUT_SIZE, sizeof(double));

    if (!net->b1 || !net->b2) {
        if (VERBOSE) printf("Failed to allocate biases\n");
        exit(1);
    }

    srand(time(NULL));
    if (VERBOSE) printf("Initializing weights...\n");
    for (int i = 0; i < HIDDEN_SIZE; i++)
        for (int j = 0; j < INPUT_SIZE; j++)
            net->W1[i][j] = ((double)rand() / RAND_MAX) * 0.01;

    for (int i = 0; i < OUTPUT_SIZE; i++)
        for (int j = 0; j < HIDDEN_SIZE; j++)
            net->W2[i][j] = ((double)rand() / RAND_MAX) * 0.01;

    if (VERBOSE) {
        printf("Weight initialization complete\n");
        printf("W1[0][0]: %.6f\n", net->W1[0][0]);
        printf("W2[0][0]: %.6f\n", net->W2[0][0]);
        printf("b1[0]: %.6f\n", net->b1[0]);
        printf("b2[0]: %.6f\n", net->b2[0]);
    }
    
    if (VERBOSE) printf("Neural network created successfully\n");
    return net;
}

// Forward pass
void forward(NeuralNetwork* net, double* input, double* hidden, double* output) {
    if (VERBOSE) printf("\nStarting forward pass...\n");
    
    // Hidden layer computation
    if (VERBOSE) printf("Computing hidden layer...\n");
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        hidden[i] = net->b1[i];
        for (int j = 0; j < INPUT_SIZE; j++)
            hidden[i] += net->W1[i][j] * input[j];
    }
    
    if (VERBOSE) {
        printf("Pre-activation hidden (first 5): ");
        for (int i = 0; i < 5; i++) printf("%.4f ", hidden[i]);
        printf("\n");
    }
    
    relu(hidden, HIDDEN_SIZE);
    
    if (VERBOSE) {
        printf("Post-ReLU hidden (first 5): ");
        for (int i = 0; i < 5; i++) printf("%.4f ", hidden[i]);
        printf("\n");
    }

    // Output layer computation
    if (VERBOSE) printf("Computing output layer...\n");
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output[i] = net->b2[i];
        for (int j = 0; j < HIDDEN_SIZE; j++)
            output[i] += net->W2[i][j] * hidden[j];
    }
    
    if (VERBOSE) {
        printf("Pre-softmax output: ");
        for (int i = 0; i < OUTPUT_SIZE; i++) printf("%.4f ", output[i]);
        printf("\n");
    }
    
    softmax(output, OUTPUT_SIZE);
    
    if (VERBOSE) {
        printf("Post-softmax output: ");
        for (int i = 0; i < OUTPUT_SIZE; i++) printf("%.4f ", output[i]);
        printf("\n");
        printf("Forward pass completed\n");
    }
}

// Backpropagation
void backward(NeuralNetwork* net, double* input, double* hidden, double* output, double* target) {
    if (VERBOSE) printf("\nStarting backward pass...\n");
    double d_output[OUTPUT_SIZE], d_hidden[HIDDEN_SIZE];

    // Compute output layer gradient
    if (VERBOSE) printf("Computing output gradients...\n");
    for (int i = 0; i < OUTPUT_SIZE; i++)
        d_output[i] = output[i] - target[i];
    
    if (VERBOSE) {
        printf("Output gradients: ");
        for (int i = 0; i < OUTPUT_SIZE; i++) printf("%.4f ", d_output[i]);
        printf("\n");
    }

    // Compute hidden layer gradient
    if (VERBOSE) printf("Computing hidden gradients...\n");
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        d_hidden[i] = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++)
            d_hidden[i] += net->W2[j][i] * d_output[j];
        d_hidden[i] *= (hidden[i] > 0);
    }
    
    if (VERBOSE) {
        printf("Hidden gradients (first 5): ");
        for (int i = 0; i < 5; i++) printf("%.4f ", d_hidden[i]);
        printf("\n");
    }

    // Update weights (gradient descent)
    if (VERBOSE) printf("Updating weights...\n");
    for (int i = 0; i < OUTPUT_SIZE; i++)
        for (int j = 0; j < HIDDEN_SIZE; j++)
            net->W2[i][j] -= LEARNING_RATE * d_output[i] * hidden[j];

    for (int i = 0; i < HIDDEN_SIZE; i++)
        for (int j = 0; j < INPUT_SIZE; j++)
            net->W1[i][j] -= LEARNING_RATE * d_hidden[i] * input[j];

    // Update biases
    if (VERBOSE) printf("Updating biases...\n");
    for (int i = 0; i < OUTPUT_SIZE; i++)
        net->b2[i] -= LEARNING_RATE * d_output[i];

    for (int i = 0; i < HIDDEN_SIZE; i++)
        net->b1[i] -= LEARNING_RATE * d_hidden[i];
    
    if (VERBOSE) {
        printf("Updated W2[0][0]: %.6f\n", net->W2[0][0]);
        printf("Updated W1[0][0]: %.6f\n", net->W1[0][0]);
        printf("Updated b2[0]: %.6f\n", net->b2[0]);
        printf("Updated b1[0]: %.6f\n", net->b1[0]);
        printf("Backward pass completed\n");
    }
}

// Train network
void train(NeuralNetwork* net, double** images, double** labels, int numImages) {
    if (VERBOSE) printf("\nStarting training...\n");
    clock_t total_start = clock();
    
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        clock_t epoch_start = clock();
        double loss = 0.0;
        int correct = 0;

        if (VERBOSE) printf("\nEpoch %d/%d\n", epoch+1, EPOCHS);
        
        for (int i = 0; i < numImages; i++) {
            if (VERBOSE && i % 1000 == 0) printf("Processing sample %d\n", i);
            
            double hidden[HIDDEN_SIZE], output[OUTPUT_SIZE];
            forward(net, images[i], hidden, output);
            backward(net, images[i], hidden, output, labels[i]);

            // Compute loss & accuracy
            for (int k = 0; k < OUTPUT_SIZE; k++) loss -= labels[i][k] * log(output[k]);
            int pred = 0, actual = 0;
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                if (output[j] > output[pred]) pred = j;
                if (labels[i][j] > labels[i][actual]) actual = j;
            }
            if (pred == actual) correct++;
        }

        printf("Epoch %d - Loss: %.4f - Train Accuracy: %.2f%% - Time: %.3fs\n",
               epoch + 1, loss / numImages, (correct / (double)numImages) * 100, get_time(epoch_start));
    }
    printf("Total training time: %.3fs\n", get_time(total_start));
    if (VERBOSE) printf("Training completed\n");
}

// Evaluate accuracy on test data
void evaluate(NeuralNetwork* net, double** images, double** labels, int numImages) {
    if (VERBOSE) printf("\nStarting evaluation...\n");
    int correct = 0;
    
    for (int i = 0; i < numImages; i++) {
        if (VERBOSE && i % 1000 == 0) printf("Evaluating sample %d\n", i);
        
        double hidden[HIDDEN_SIZE], output[OUTPUT_SIZE];
        forward(net, images[i], hidden, output);
        
        int pred = 0, actual = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            if (output[j] > output[pred]) pred = j;
            if (labels[i][j] > labels[i][actual]) actual = j;
        }
        if (pred == actual) correct++;
        
        if (VERBOSE && i < 3) {
            printf("Sample %d - Predicted: %d, Actual: %d\n", i, pred, actual);
            printf("Output probabilities: ");
            for (int j = 0; j < OUTPUT_SIZE; j++) printf("%.2f ", output[j]);
            printf("\n");
        }
    }
    
    printf("Test Accuracy: %.2f%%\n", (correct / (double)numImages) * 100);
    if (VERBOSE) printf("Evaluation completed\n");
}

// Read MNIST dataset
double** loadMNISTImages(const char* filename, int numImages) {
    if (VERBOSE) printf("Loading MNIST images from %s...\n", filename);
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening %s\n", filename);
        exit(1);
    }
    fseek(file, 16, SEEK_SET);
    double** images = allocateMatrix(numImages, INPUT_SIZE);
    for (int i = 0; i < numImages; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            unsigned char pixel;

            if (fread(&pixel, sizeof(unsigned char), 1, file) != 1) {
                fprintf(stderr, "Error: Failed to read pixel\n");
                fclose(file);
                exit(EXIT_FAILURE);
            }

            images[i][j] = pixel / 255.0;
        }
        if (VERBOSE && i % 10000 == 0) printf("Loaded %d images\n", i);
    }
    fclose(file);
    
    if (VERBOSE) {
        printf("First pixel of first image: %.4f\n", images[0][0]);
        printf("Last pixel of first image: %.4f\n", images[0][INPUT_SIZE-1]);
        printf("Image loading complete\n");
    }
    return images;
}

double** loadMNISTLabels(const char* filename, int numLabels) {
    if (VERBOSE) printf("Loading MNIST labels from %s...\n", filename);
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening %s\n", filename);
        exit(1);
    }
    fseek(file, 8, SEEK_SET);
    double** labels = allocateMatrix(numLabels, OUTPUT_SIZE);
    for (int i = 0; i < numLabels; i++) {
        unsigned char label;
        if (fread(&label, sizeof(unsigned char), 1, file) != 1) {
            fprintf(stderr, "Error: Failed to read label\n");
            fclose(file);
            exit(EXIT_FAILURE);
        }

        for (int j = 0; j < OUTPUT_SIZE; j++) {
            labels[i][j] = (j == label) ? 1.0 : 0.0;
        }
        
        if (VERBOSE && i % 10000 == 0) printf("Loaded %d labels\n", i);
    }
    fclose(file);
    
    if (VERBOSE) {
        printf("First label: ");
        for (int j = 0; j < OUTPUT_SIZE; j++) printf("%.1f ", labels[0][j]);
        printf("\n");
        printf("Label loading complete\n");
    }
    return labels;
}

// Free network memory
void freeNetwork(NeuralNetwork* net) {
    if (VERBOSE) printf("Freeing neural network...\n");
    freeMatrix(net->W1, HIDDEN_SIZE);
    freeMatrix(net->W2, OUTPUT_SIZE);
    free(net->b1);
    free(net->b2);
    free(net);
    if (VERBOSE) printf("Neural network freed\n");
}

// Main function
int main() {
    printf("MNIST Neural Network\n\n");

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
    
    if (VERBOSE) printf("Program completed successfully\n");
    return 0;
}