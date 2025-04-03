#include "neural_net.h"
#include "utils.h"
#include <stdlib.h>
#include <time.h>

NeuralNetwork* createNetwork() {
    if (VERBOSE) printf("Creating neural network...\n");
    NeuralNetwork* net = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    if (!net) {
        if (VERBOSE) printf("Failed to allocate neural network\n");
        exit(1);
    }
    
    // Allocate flattened matrices
    net->W1 = (double*)malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(double));
    net->W2 = (double*)malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double));
    net->b1 = (double*)calloc(HIDDEN_SIZE, sizeof(double));
    net->b2 = (double*)calloc(OUTPUT_SIZE, sizeof(double));

    if (!net->W1 || !net->W2 || !net->b1 || !net->b2) {
        if (VERBOSE) printf("Failed to allocate weights/biases\n");
        exit(1);
    }

    srand(time(NULL));
    if (VERBOSE) printf("Initializing weights...\n");
    
    // Initialize W1 (flattened)
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            net->W1[i * INPUT_SIZE + j] = ((double)rand() / RAND_MAX) * 0.01;
        }
    }

    // Initialize W2 (flattened)
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            net->W2[i * HIDDEN_SIZE + j] = ((double)rand() / RAND_MAX) * 0.01;
        }
    }

    // Allocate and copy device memory
    checkCudaError(cudaMalloc(&net->d_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(double)), "cudaMalloc d_W1");
    checkCudaError(cudaMalloc(&net->d_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double)), "cudaMalloc d_W2");
    checkCudaError(cudaMalloc(&net->d_b1, HIDDEN_SIZE * sizeof(double)), "cudaMalloc d_b1");
    checkCudaError(cudaMalloc(&net->d_b2, OUTPUT_SIZE * sizeof(double)), "cudaMalloc d_b2");
    checkCudaError(cudaMalloc(&net->d_input, INPUT_SIZE * sizeof(double)), "cudaMalloc d_input");
    checkCudaError(cudaMalloc(&net->d_hidden, HIDDEN_SIZE * sizeof(double)), "cudaMalloc d_hidden");
    checkCudaError(cudaMalloc(&net->d_output, OUTPUT_SIZE * sizeof(double)), "cudaMalloc d_output");

    // Copy initial values to device
    checkCudaError(cudaMemcpy(net->d_W1, net->W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy d_W1");
    checkCudaError(cudaMemcpy(net->d_W2, net->W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy d_W2");
    checkCudaError(cudaMemcpy(net->d_b1, net->b1, HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy d_b1");
    checkCudaError(cudaMemcpy(net->d_b2, net->b2, OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy d_b2");

    if (VERBOSE) {
        printf("Weight initialization complete\n");
        printf("W1[0][0]: %.6f\n", net->W1[0]);  // First element of flattened W1
        printf("W2[0][0]: %.6f\n", net->W2[0]);  // First element of flattened W2
        printf("b1[0]: %.6f\n", net->b1[0]);
        printf("b2[0]: %.6f\n", net->b2[0]);
    }
    
    if (VERBOSE) printf("Neural network created successfully\n");
    return net;
}

// Forward pass
void forward(NeuralNetwork* net, double* input, double* hidden, double* output) {
    if (VERBOSE) printf("\nStarting forward pass...\n");
    
    // Copy input to device
    cudaMemcpy(net->d_input, input, INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    
    // Compute hidden layer
    if (VERBOSE) printf("Computing hidden layer...\n");
    int blockSize = 256;
    int numBlocks = (HIDDEN_SIZE + blockSize - 1) / blockSize;
    
    matrixVectorMultiply<<<numBlocks, blockSize>>>(
        net->d_W1, net->d_input, net->d_b1, net->d_hidden, HIDDEN_SIZE, INPUT_SIZE
    );
    
    // Apply ReLU
    relu_kernel<<<numBlocks, blockSize>>>(net->d_hidden, HIDDEN_SIZE);
    
    // Compute output layer
    if (VERBOSE) printf("Computing output layer...\n");
    numBlocks = (OUTPUT_SIZE + blockSize - 1) / blockSize;
    
    matrixVectorMultiply<<<numBlocks, blockSize>>>(
        net->d_W2, net->d_hidden, net->d_b2, net->d_output, OUTPUT_SIZE, HIDDEN_SIZE
    );
    
    // Apply softmax
    softmax_kernel<<<numBlocks, blockSize>>>(net->d_output, OUTPUT_SIZE);
    
    // Copy results back to host
    cudaMemcpy(hidden, net->d_hidden, HIDDEN_SIZE * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(output, net->d_output, OUTPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost);
    
    if (VERBOSE) {
        printf("Post-ReLU hidden (first 5): ");
        for (int i = 0; i < 5; i++) printf("%.4f ", hidden[i]);
        printf("\n");
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
            d_hidden[i] += net->W2[j * HIDDEN_SIZE + i] * d_output[j];  // Note the flattened access
        d_hidden[i] *= (hidden[i] > 0);
    }
    
    if (VERBOSE) {
        printf("Hidden gradients (first 5): ");
        for (int i = 0; i < 5; i++) printf("%.4f ", d_hidden[i]);
        printf("\n");
    }

    // Update weights (gradient descent)
    if (VERBOSE) printf("Updating weights...\n");
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            net->W2[i * HIDDEN_SIZE + j] -= LEARNING_RATE * d_output[i] * hidden[j];
        }
    }

    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            net->W1[i * INPUT_SIZE + j] -= LEARNING_RATE * d_hidden[i] * input[j];
        }
    }

    // Update biases
    if (VERBOSE) printf("Updating biases...\n");
    for (int i = 0; i < OUTPUT_SIZE; i++)
        net->b2[i] -= LEARNING_RATE * d_output[i];

    for (int i = 0; i < HIDDEN_SIZE; i++)
        net->b1[i] -= LEARNING_RATE * d_hidden[i];
    
    // Update device weights
    checkCudaError(cudaMemcpy(net->d_W1, net->W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy update d_W1");
    checkCudaError(cudaMemcpy(net->d_W2, net->W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy update d_W2");
    checkCudaError(cudaMemcpy(net->d_b1, net->b1, HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy update d_b1");
    checkCudaError(cudaMemcpy(net->d_b2, net->b2, OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy update d_b2");
    
    if (VERBOSE) {
        printf("Updated W2[0][0]: %.6f\n", net->W2[0]);
        printf("Updated W1[0][0]: %.6f\n", net->W1[0]);
        printf("Updated b2[0]: %.6f\n", net->b2[0]);
        printf("Updated b1[0]: %.6f\n", net->b1[0]);
        printf("Backward pass completed\n");
    }
}

// Train network
void train(NeuralNetwork* net, double** images, double** labels, int numImages) {
    if (VERBOSE) printf("\nStarting training...\n");
    
    cudaEvent_t total_start, total_stop;
    create_timer(&total_start, &total_stop);
    start_timer(total_start);
    
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        cudaEvent_t epoch_start, epoch_stop;
        create_timer(&epoch_start, &epoch_stop);
        start_timer(epoch_start);

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
               epoch + 1, loss / numImages, (correct / (double)numImages) * 100, 
               stop_timer(epoch_start, epoch_stop));
    }
    
    printf("Total training time: %.3fs\n", stop_timer(total_start, total_stop));

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

void freeNetwork(NeuralNetwork* net) {
    if (VERBOSE) printf("Freeing neural network...\n");
    
    // Free host memory
    free(net->W1);
    free(net->W2);
    free(net->b1);
    free(net->b2);
    
    // Free device memory
    cudaFree(net->d_W1);
    cudaFree(net->d_W2);
    cudaFree(net->d_b1);
    cudaFree(net->d_b2);
    cudaFree(net->d_input);
    cudaFree(net->d_hidden);
    cudaFree(net->d_output);
    
    free(net);
    if (VERBOSE) printf("Neural network freed\n");
}

// Matrix-vector multiplication kernel
__global__ void matrixVectorMultiply(double* W, double* x, double* b, double* result, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        double sum = b[row];
        for (int col = 0; col < cols; col++) {
            sum += W[row * cols + col] * x[col];
        }
        result[row] = sum;
    }
}

// ReLU activation kernel
__global__ void relu_kernel(double* x, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        x[idx] = (x[idx] > 0) ? x[idx] : 0;
    }
}

// Softmax activation kernel
__global__ void softmax_kernel(double* x, int size) {
    __shared__ double sum;
    double val = 0;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        val = exp(x[idx]);
    }
    
    // Thread 0 in block initializes sum
    if (threadIdx.x == 0) sum = 0;
    __syncthreads();
    
    // Atomic add to sum
    atomicAdd(&sum, val);
    __syncthreads();
    
    if (idx < size) {
        x[idx] = val / sum;
    }
}

// Host wrapper for ReLU
void relu(double* x, int size) {
    if (VERBOSE) printf("Applying ReLU to vector of size %d\n", size);
    
    double* d_x;
    cudaMalloc(&d_x, size * sizeof(double));
    cudaMemcpy(d_x, x, size * sizeof(double), cudaMemcpyHostToDevice);
    
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    relu_kernel<<<numBlocks, blockSize>>>(d_x, size);
    
    cudaMemcpy(x, d_x, size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_x);
    
    if (VERBOSE) {
        printf("ReLU output (first 5 elements): ");
        for (int i = 0; i < 5 && i < size; i++) printf("%.4f ", x[i]);
        printf("\n");
    }
}

// Host wrapper for softmax
void softmax(double* x, int size) {
    if (VERBOSE) printf("Applying softmax to vector of size %d\n", size);
    
    double* d_x;
    cudaMalloc(&d_x, size * sizeof(double));
    cudaMemcpy(d_x, x, size * sizeof(double), cudaMemcpyHostToDevice);
    
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    softmax_kernel<<<numBlocks, blockSize>>>(d_x, size);
    
    cudaMemcpy(x, d_x, size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_x);
    
    if (VERBOSE) {
        printf("Softmax output: ");
        for (int i = 0; i < size; i++) printf("%.4f ", x[i]);
        printf("\n");
    }
}