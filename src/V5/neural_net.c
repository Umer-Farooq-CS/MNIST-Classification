#include "neural_net.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

// Add this new function implementation
void initWeights(double* W, int n, double scale, unsigned long seed) {
    srand(seed);
    #pragma acc parallel loop present(W[0:n])
    for (int i = 0; i < n; ++i) {
        W[i] = scale * (2.0 * rand() / RAND_MAX - 1.0); // Random values between -scale and +scale
    }
}

// Activation functions (unchanged)
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double x) {
    return x * (1.0 - x);
}

void relu(double* x, int size) {
    #pragma acc parallel loop present(x[0:size])
    for (int i = 0; i < size; ++i) {
        x[i] = (x[i] > 0) ? x[i] : 0;
    }
}

void softmax(double* x, int size) {
    double max_val = -1e30;
    #pragma acc parallel loop reduction(max:max_val) present(x[0:size])
    for (int i = 0; i < size; ++i) {
        if (x[i] > max_val) max_val = x[i];
    }

    double sum = 0.0;
    #pragma acc parallel loop reduction(+:sum) present(x[0:size])
    for (int i = 0; i < size; ++i) {
        x[i] = exp(x[i] - max_val);
        sum += x[i];
    }

    #pragma acc parallel loop present(x[0:size])
    for (int i = 0; i < size; ++i) {
        x[i] /= sum;
    }
}

// Matrix-vector multiplication with bias addition (unchanged)
void matrixVectorMultiply(double* W, double* x, double* b, double* y, int rows, int cols) {
    #pragma acc parallel loop present(W[0:rows*cols], x[0:cols], b[0:rows], y[0:rows])
    for (int i = 0; i < rows; ++i) {
        double sum = 0.0;
        for (int j = 0; j < cols; ++j) {
            sum += W[i * cols + j] * x[j];
        }
        y[i] = sum + b[i];
    }
}

// Forward propagation through the network
void forwardPass(NeuralNetwork* net, double* input, double* hidden, double* output) {
    matrixVectorMultiply(net->W1, input, net->b1, hidden, HIDDEN_SIZE, INPUT_SIZE);
    relu(hidden, HIDDEN_SIZE);
    matrixVectorMultiply(net->W2, hidden, net->b2, output, OUTPUT_SIZE, HIDDEN_SIZE);
    softmax(output, OUTPUT_SIZE);
}

// Backward propagation (with OpenACC support)
void backward(NeuralNetwork* net, double* input, double* hidden, double* output, double* target) {
    double output_error[OUTPUT_SIZE];
    double hidden_error[HIDDEN_SIZE];

    // Compute output layer error
    #pragma acc parallel loop present(output[0:OUTPUT_SIZE], target[0:OUTPUT_SIZE], output_error[0:OUTPUT_SIZE])
    for (int i = 0; i < OUTPUT_SIZE; ++i) {
        output_error[i] = output[i] - target[i];  // Cross-entropy derivative
    }

    // Compute hidden layer error
    #pragma acc parallel loop present(output_error[0:OUTPUT_SIZE], net->W2[0:OUTPUT_SIZE*HIDDEN_SIZE], hidden_error[0:HIDDEN_SIZE])
    for (int i = 0; i < HIDDEN_SIZE; ++i) {
        hidden_error[i] = 0.0;
        for (int j = 0; j < OUTPUT_SIZE; ++j) {
            hidden_error[i] += output_error[j] * net->W2[j * HIDDEN_SIZE + i];
        }
        // Apply the ReLU derivative
        hidden_error[i] *= (hidden[i] > 0) ? 1.0 : 0.0;
    }

    // Update weights and biases for W2 and b2
    #pragma acc parallel loop present(net->W2[0:OUTPUT_SIZE*HIDDEN_SIZE], hidden[0:HIDDEN_SIZE], output_error[0:OUTPUT_SIZE])
    for (int i = 0; i < OUTPUT_SIZE; ++i) {
        for (int j = 0; j < HIDDEN_SIZE; ++j) {
            net->W2[i * HIDDEN_SIZE + j] -= LEARNING_RATE * output_error[i] * hidden[j];
        }
        net->b2[i] -= LEARNING_RATE * output_error[i];
    }

    // Update weights and biases for W1 and b1
    #pragma acc parallel loop present(net->W1[0:HIDDEN_SIZE*INPUT_SIZE], input[0:INPUT_SIZE], hidden_error[0:HIDDEN_SIZE])
    for (int i = 0; i < HIDDEN_SIZE; ++i) {
        for (int j = 0; j < INPUT_SIZE; ++j) {
            net->W1[i * INPUT_SIZE + j] -= LEARNING_RATE * hidden_error[i] * input[j];
        }
        net->b1[i] -= LEARNING_RATE * hidden_error[i];
    }
}

// Training the network (with OpenACC support)
void train(NeuralNetwork* net, double* images, double* labels, int numImages) {
    double input[INPUT_SIZE];
    double hidden[HIDDEN_SIZE];
    double output[OUTPUT_SIZE];
    
    struct timeval start, end;
    double total_time = 0;
    
    #pragma acc data copyin(images[0:numImages*INPUT_SIZE], labels[0:numImages*OUTPUT_SIZE])
    {
        for (int epoch = 0; epoch < EPOCHS; ++epoch) {
            gettimeofday(&start, NULL);
            
            int correct = 0;
            double total_loss = 0.0;
            
            #pragma acc parallel loop reduction(+:correct, total_loss)
            for (int i = 0; i < numImages; ++i) {
                // Copy the image to input layer
                #pragma acc loop
                for (int j = 0; j < INPUT_SIZE; ++j) {
                    input[j] = images[i * INPUT_SIZE + j];
                }

                // Forward pass
                forwardPass(net, input, hidden, output);
                
                // Calculate loss (cross-entropy)
                double loss = 0.0;
                #pragma acc loop reduction(+:loss)
                for (int j = 0; j < OUTPUT_SIZE; ++j) {
                    if (labels[i * OUTPUT_SIZE + j] == 1.0) {
                        loss += -log(output[j] + 1e-10);  // Small epsilon to avoid log(0)
                    }
                }
                total_loss += loss;
                
                // Check prediction
                int pred = 0;
                double max_val = output[0];
                #pragma acc loop reduction(max:max_val)
                for (int j = 1; j < OUTPUT_SIZE; ++j) {
                    if (output[j] > max_val) {
                        max_val = output[j];
                        pred = j;
                    }
                }
                
                // Check target
                int target = 0;
                #pragma acc loop
                for (int j = 0; j < OUTPUT_SIZE; ++j) {
                    if (labels[i * OUTPUT_SIZE + j] == 1.0) {
                        target = j;
                    }
                }
                
                if (pred == target) correct++;
                
                // Backward pass
                backward(net, input, hidden, output, &labels[i * OUTPUT_SIZE]);
            }
            
            gettimeofday(&end, NULL);
            double epoch_time = (end.tv_sec - start.tv_sec) + 
                               (end.tv_usec - start.tv_usec) / 1000000.0;
            total_time += epoch_time;
            
            double avg_loss = total_loss / numImages;
            double accuracy = (double)correct / numImages * 100.0;
            
            printf("Epoch %d - Loss: %.4f - Train Accuracy: %.2f%% - Time: %.3fs\n",
                   epoch + 1, avg_loss, accuracy, epoch_time);
        }
    }
    
    printf("Total training time: %.3fs\n", total_time);
}

void initNetwork(NeuralNetwork* net) {
    unsigned long seed = time(NULL);
    net->W1 = allocateMatrix(HIDDEN_SIZE, INPUT_SIZE);
    net->b1 = allocateMatrix(HIDDEN_SIZE, 1);
    net->W2 = allocateMatrix(OUTPUT_SIZE, HIDDEN_SIZE);
    net->b2 = allocateMatrix(OUTPUT_SIZE, 1);

    initWeights(net->W1, HIDDEN_SIZE * INPUT_SIZE, 0.1, seed);
    initWeights(net->b1, HIDDEN_SIZE, 0.1, seed + 1);
    initWeights(net->W2, OUTPUT_SIZE * HIDDEN_SIZE, 0.1, seed + 2);
    initWeights(net->b2, OUTPUT_SIZE, 0.1, seed + 3);
}

void freeNetwork(NeuralNetwork* net) {
    freeMatrix(net->W1, HIDDEN_SIZE, INPUT_SIZE);
    freeMatrix(net->b1, HIDDEN_SIZE, 1);
    freeMatrix(net->W2, OUTPUT_SIZE, HIDDEN_SIZE);
    freeMatrix(net->b2, OUTPUT_SIZE, 1);
}

void evaluate(NeuralNetwork* net, double* images, double* labels, int numImages) {
    int correct = 0;
    double input[INPUT_SIZE];
    double hidden[HIDDEN_SIZE];
    double output[OUTPUT_SIZE];
    double total_loss = 0.0;
    
    struct timeval start, end;
    gettimeofday(&start, NULL);
    
    #pragma acc data copyin(images[0:numImages*INPUT_SIZE], labels[0:numImages*OUTPUT_SIZE])
    {
        #pragma acc parallel loop reduction(+:correct, total_loss)
        for (int i = 0; i < numImages; ++i) {
            #pragma acc loop
            for (int j = 0; j < INPUT_SIZE; ++j) {
                input[j] = images[i * INPUT_SIZE + j];
            }

            forwardPass(net, input, hidden, output);
            
            // Calculate loss
            double loss = 0.0;
            #pragma acc loop reduction(+:loss)
            for (int j = 0; j < OUTPUT_SIZE; ++j) {
                if (labels[i * OUTPUT_SIZE + j] == 1.0) {
                    loss += -log(output[j] + 1e-10);
                }
            }
            total_loss += loss;
            
            // Check prediction
            int pred = 0;
            double max_val = output[0];
            #pragma acc loop reduction(max:max_val)
            for (int j = 1; j < OUTPUT_SIZE; ++j) {
                if (output[j] > max_val) {
                    max_val = output[j];
                    pred = j;
                }
            }
            
            // Check target
            int target = 0;
            #pragma acc loop
            for (int j = 0; j < OUTPUT_SIZE; ++j) {
                if (labels[i * OUTPUT_SIZE + j] == 1.0) {
                    target = j;
                }
            }
            
            if (pred == target) correct++;
        }
    }
    
    gettimeofday(&end, NULL);
    double eval_time = (end.tv_sec - start.tv_sec) + 
                      (end.tv_usec - start.tv_usec) / 1000000.0;
    
    double avg_loss = total_loss / numImages;
    double accuracy = (double)correct / numImages * 100.0;
    
    printf("\nTest Loss: %.4f\n", avg_loss);
    printf("Test Accuracy: %.2f%%\n", accuracy);
    printf("Evaluation time: %.3fs\n", eval_time);
}