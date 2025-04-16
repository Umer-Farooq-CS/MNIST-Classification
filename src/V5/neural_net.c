#include "neural_net.h"
#include "utils.h"
#include <math.h>
#include <openacc.h>

// Initialize weights
NeuralNetwork* createNetwork() {
    if (VERBOSE) printf("Creating neural network...\n");
    NeuralNetwork* net = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    if (!net) {
        if (VERBOSE) printf("Failed to allocate neural network\n");
        exit(1);
    }
    
    // Allocate memory
    net->W1 = allocateMatrix(HIDDEN_SIZE, INPUT_SIZE);
    net->W2 = allocateMatrix(OUTPUT_SIZE, HIDDEN_SIZE);
    net->b1 = (double*)calloc(HIDDEN_SIZE, sizeof(double));
    net->b2 = (double*)calloc(OUTPUT_SIZE, sizeof(double));

    // Initialize weights
    srand(time(NULL));
    
    #pragma acc enter data create(net[0:1], net->W1[0:HIDDEN_SIZE*INPUT_SIZE], \
                              net->W2[0:OUTPUT_SIZE*HIDDEN_SIZE], \
                              net->b1[0:HIDDEN_SIZE], net->b2[0:OUTPUT_SIZE])
    
    #pragma acc parallel loop present(net->W1[0:HIDDEN_SIZE*INPUT_SIZE])
    for (int i = 0; i < HIDDEN_SIZE * INPUT_SIZE; i++) {
        net->W1[i] = ((double)rand() / RAND_MAX) * 0.01;
    }
    
    #pragma acc parallel loop present(net->W2[0:OUTPUT_SIZE*HIDDEN_SIZE])
    for (int i = 0; i < OUTPUT_SIZE * HIDDEN_SIZE; i++) {
        net->W2[i] = ((double)rand() / RAND_MAX) * 0.01;
    }

    if (VERBOSE) {
        printf("Weight initialization complete\n");
        printf("W1[0][0]: %.6f\n", net->W1[0]);
        printf("W2[0][0]: %.6f\n", net->W2[0]);
        printf("b1[0]: %.6f\n", net->b1[0]);
        printf("b2[0]: %.6f\n", net->b2[0]);
    }
    
    if (VERBOSE) printf("Neural network created successfully\n");
    return net;
}

// ReLU activation
void relu(double* x, int size) {
    #pragma acc parallel loop present(x[0:size])
    for (int i = 0; i < size; i++) {
        x[i] = (x[i] > 0) ? x[i] : 0;
    }
}

// Softmax activation
void softmax(double* x, int size) {
    double max_val = x[0];
    
    // Find max for numerical stability
    #pragma acc parallel loop reduction(max:max_val) present(x[0:size])
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    
    double sum = 0.0;
    #pragma acc parallel loop reduction(+:sum) present(x[0:size])
    for (int i = 0; i < size; i++) {
        x[i] = exp(x[i] - max_val);
        sum += x[i];
    }
    
    #pragma acc parallel loop present(x[0:size])
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

// Forward pass with OpenACC
void forward(NeuralNetwork* net, double* input, double* hidden, double* output) {
    if (VERBOSE) printf("\nStarting forward pass...\n");
    
    // Copy input to device if not already there
    #pragma acc enter data copyin(input[0:INPUT_SIZE]) \
                     create(hidden[0:HIDDEN_SIZE], output[0:OUTPUT_SIZE])
    
    // Hidden layer computation
    #pragma acc parallel loop present(net->W1[0:HIDDEN_SIZE*INPUT_SIZE], \
                                    net->b1[0:HIDDEN_SIZE], input[0:INPUT_SIZE], \
                                    hidden[0:HIDDEN_SIZE])
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        double sum = net->b1[i];
        for (int j = 0; j < INPUT_SIZE; j++) {
            sum += net->W1[i * INPUT_SIZE + j] * input[j];
        }
        hidden[i] = sum;
    }
    
    // Apply ReLU
    relu(hidden, HIDDEN_SIZE);
    
    // Output layer computation
    #pragma acc parallel loop present(net->W2[0:OUTPUT_SIZE*HIDDEN_SIZE], \
                                    net->b2[0:OUTPUT_SIZE], hidden[0:HIDDEN_SIZE], \
                                    output[0:OUTPUT_SIZE])
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        double sum = net->b2[i];
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            sum += net->W2[i * HIDDEN_SIZE + j] * hidden[j];
        }
        output[i] = sum;
    }
    
    // Apply softmax
    softmax(output, OUTPUT_SIZE);
    
    // Ensure results are on host
    #pragma acc update self(hidden[0:HIDDEN_SIZE], output[0:OUTPUT_SIZE])
    
    if (VERBOSE) {
        printf("Post-ReLU hidden (first 5): ");
        for (int i = 0; i < 5; i++) printf("%.4f ", hidden[i]);
        printf("\n");
        printf("Post-softmax output: ");
        for (int i = 0; i < OUTPUT_SIZE; i++) printf("%.4f ", output[i]);
        printf("\n");
        printf("Forward pass completed\n");
    }
    
    // Clean up temporary device data
    #pragma acc exit data delete(input[0:INPUT_SIZE])
}

// Backward pass with OpenACC
void backward(NeuralNetwork* net, double* input, double* hidden, double* output, double* target) {
    if (VERBOSE) printf("\nStarting backward pass...\n");
    
    double d_output[OUTPUT_SIZE], d_hidden[HIDDEN_SIZE];
    
    // Copy target to device
    #pragma acc enter data copyin(target[0:OUTPUT_SIZE]) \
                     create(d_output[0:OUTPUT_SIZE], d_hidden[0:HIDDEN_SIZE])
    
    // Compute output layer gradient
    #pragma acc parallel loop present(output[0:OUTPUT_SIZE], target[0:OUTPUT_SIZE], \
                                    d_output[0:OUTPUT_SIZE])
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        d_output[i] = output[i] - target[i];
    }
    
    // Compute hidden layer gradient
    #pragma acc parallel loop present(net->W2[0:OUTPUT_SIZE*HIDDEN_SIZE], \
                                    d_output[0:OUTPUT_SIZE], hidden[0:HIDDEN_SIZE], \
                                    d_hidden[0:HIDDEN_SIZE])
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        double sum = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            sum += net->W2[j * HIDDEN_SIZE + i] * d_output[j];
        }
        d_hidden[i] = sum * (hidden[i] > 0);
    }
    
    // Update weights (gradient descent)
    #pragma acc parallel loop present(net->W2[0:OUTPUT_SIZE*HIDDEN_SIZE], \
                                    d_output[0:OUTPUT_SIZE], hidden[0:HIDDEN_SIZE])
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            net->W2[i * HIDDEN_SIZE + j] -= LEARNING_RATE * d_output[i] * hidden[j];
        }
    }
    
    #pragma acc parallel loop present(net->W1[0:HIDDEN_SIZE*INPUT_SIZE], \
                                    d_hidden[0:HIDDEN_SIZE], input[0:INPUT_SIZE])
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            net->W1[i * INPUT_SIZE + j] -= LEARNING_RATE * d_hidden[i] * input[j];
        }
    }
    
    // Update biases
    #pragma acc parallel loop present(net->b2[0:OUTPUT_SIZE], d_output[0:OUTPUT_SIZE])
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        net->b2[i] -= LEARNING_RATE * d_output[i];
    }
    
    #pragma acc parallel loop present(net->b1[0:HIDDEN_SIZE], d_hidden[0:HIDDEN_SIZE])
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        net->b1[i] -= LEARNING_RATE * d_hidden[i];
    }
    
    if (VERBOSE) {
        printf("Updated W2[0][0]: %.6f\n", net->W2[0]);
        printf("Updated W1[0][0]: %.6f\n", net->W1[0]);
        printf("Updated b2[0]: %.6f\n", net->b2[0]);
        printf("Updated b1[0]: %.6f\n", net->b1[0]);
        printf("Backward pass completed\n");
    }
    
    // Clean up temporary device data
    #pragma acc exit data delete(target[0:OUTPUT_SIZE], d_output[0:OUTPUT_SIZE], \
                               d_hidden[0:HIDDEN_SIZE])
}

// Training function
void train(NeuralNetwork* net, double* images, double* labels, int numImages) {
    if (VERBOSE) printf("\nStarting training...\n");
    
    // Copy all data to device at once
    #pragma acc enter data copyin(images[0:numImages*INPUT_SIZE], \
                                labels[0:numImages*OUTPUT_SIZE])
    
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        double loss = 0.0;
        int correct = 0;

        if (VERBOSE) printf("\nEpoch %d/%d\n", epoch+1, EPOCHS);
        
        #pragma acc parallel loop reduction(+:loss, correct) \
                     present(images[0:numImages*INPUT_SIZE], labels[0:numImages*OUTPUT_SIZE])
        for (int i = 0; i < numImages; i++) {
            double hidden[HIDDEN_SIZE], output[OUTPUT_SIZE];
            
            // Forward pass for this sample
            #pragma acc loop seq
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                double sum = net->b1[j];
                for (int k = 0; k < INPUT_SIZE; k++) {
                    sum += net->W1[j * INPUT_SIZE + k] * images[i * INPUT_SIZE + k];
                }
                hidden[j] = (sum > 0) ? sum : 0;  // ReLU
            }
            
            #pragma acc loop seq
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                double sum = net->b2[j];
                for (int k = 0; k < HIDDEN_SIZE; k++) {
                    sum += net->W2[j * HIDDEN_SIZE + k] * hidden[k];
                }
                output[j] = sum;
            }
            
            // Softmax
            double max_val = output[0];
            #pragma acc loop reduction(max:max_val)
            for (int j = 1; j < OUTPUT_SIZE; j++) {
                if (output[j] > max_val) max_val = output[j];
            }
            
            double sum_exp = 0.0;
            #pragma acc loop reduction(+:sum_exp)
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                output[j] = exp(output[j] - max_val);
                sum_exp += output[j];
            }
            
            #pragma acc loop
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                output[j] /= sum_exp;
            }
            
            // Compute loss & accuracy
            #pragma acc loop reduction(+:loss)
            for (int k = 0; k < OUTPUT_SIZE; k++) {
                loss -= labels[i * OUTPUT_SIZE + k] * log(output[k]);
            }
            
            int pred = 0, actual = 0;
            #pragma acc loop seq
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                if (output[j] > output[pred]) pred = j;
                if (labels[i * OUTPUT_SIZE + j] > labels[i * OUTPUT_SIZE + actual]) actual = j;
            }
            if (pred == actual) correct++;
        }

        printf("Epoch %d - Loss: %.4f - Train Accuracy: %.2f%%\n",
               epoch + 1, loss / numImages, (correct / (double)numImages) * 100);
    }
    
    // Clean up data from device
    #pragma acc exit data delete(images[0:numImages*INPUT_SIZE], \
                               labels[0:numImages*OUTPUT_SIZE])
    
    if (VERBOSE) printf("Training completed\n");
}

// Evaluation function
void evaluate(NeuralNetwork* net, double* images, double* labels, int numImages) {
    if (VERBOSE) printf("\nStarting evaluation...\n");
    int correct = 0;
    
    // Copy data to device
    #pragma acc enter data copyin(images[0:numImages*INPUT_SIZE], \
                                labels[0:numImages*OUTPUT_SIZE])
    
    #pragma acc parallel loop reduction(+:correct) \
                 present(images[0:numImages*INPUT_SIZE], labels[0:numImages*OUTPUT_SIZE])
    for (int i = 0; i < numImages; i++) {
        double hidden[HIDDEN_SIZE], output[OUTPUT_SIZE];
        
        // Forward pass
        #pragma acc loop seq
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            double sum = net->b1[j];
            for (int k = 0; k < INPUT_SIZE; k++) {
                sum += net->W1[j * INPUT_SIZE + k] * images[i * INPUT_SIZE + k];
            }
            hidden[j] = (sum > 0) ? sum : 0;  // ReLU
        }
        
        #pragma acc loop seq
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            double sum = net->b2[j];
            for (int k = 0; k < HIDDEN_SIZE; k++) {
                sum += net->W2[j * HIDDEN_SIZE + k] * hidden[k];
            }
            output[j] = sum;
        }
        
        // Softmax
        double max_val = output[0];
        #pragma acc loop reduction(max:max_val)
        for (int j = 1; j < OUTPUT_SIZE; j++) {
            if (output[j] > max_val) max_val = output[j];
        }
        
        double sum_exp = 0.0;
        #pragma acc loop reduction(+:sum_exp)
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            output[j] = exp(output[j] - max_val);
            sum_exp += output[j];
        }
        
        #pragma acc loop
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            output[j] /= sum_exp;
        }
        
        // Compute accuracy
        int pred = 0, actual = 0;
        #pragma acc loop seq
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            if (output[j] > output[pred]) pred = j;
            if (labels[i * OUTPUT_SIZE + j] > labels[i * OUTPUT_SIZE + actual]) actual = j;
        }
        if (pred == actual) correct++;
    }
    
    printf("Test Accuracy: %.2f%%\n", (correct / (double)numImages) * 100);
    
    // Clean up data from device
    #pragma acc exit data delete(images[0:numImages*INPUT_SIZE], \
                               labels[0:numImages*OUTPUT_SIZE])
    
    if (VERBOSE) printf("Evaluation completed\n");
}

void freeNetwork(NeuralNetwork* net) {
    if (VERBOSE) printf("Freeing neural network...\n");
    
    // Remove data from device
    #pragma acc exit data delete(net->W1[0:HIDDEN_SIZE*INPUT_SIZE], \
                               net->W2[0:OUTPUT_SIZE*HIDDEN_SIZE], \
                               net->b1[0:HIDDEN_SIZE], net->b2[0:OUTPUT_SIZE], \
                               net[0:1])
    
    // Free host memory
    free(net->W1);
    free(net->W2);
    free(net->b1);
    free(net->b2);
    free(net);
    
    if (VERBOSE) printf("Neural network freed\n");
}