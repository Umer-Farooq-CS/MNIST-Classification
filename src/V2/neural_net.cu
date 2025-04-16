#include "neural_net.h"
#include "utils.h"
#include <stdlib.h>
#include <time.h>

// Kernel for computing hidden layer with ReLU activation
__global__ void computeHiddenLayer(double* d_input, double* d_W1, double* d_b1, double* d_hidden, int input_size, int hidden_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < hidden_size) {
        double sum = d_b1[i];
        for (int j = 0; j < input_size; j++) {
            sum += d_W1[i * input_size + j] * d_input[j];
        }
        // ReLU activation
        d_hidden[i] = (sum > 0) ? sum : 0;
    }
}

// Kernel for computing output layer with softmax activation
__global__ void computeOutputLayer(double* d_hidden, double* d_W2, double* d_b2, double* d_output, int hidden_size, int output_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < output_size) {
        double sum = d_b2[i];
        for (int j = 0; j < hidden_size; j++) {
            sum += d_W2[i * hidden_size + j] * d_hidden[j];
        }
        d_output[i] = sum;
    }
}

// Kernel for softmax activation
__global__ void softmaxKernel(double* x, int size) {
    __shared__ double sum;
    double val = exp(x[threadIdx.x]);
    
    if (threadIdx.x == 0) sum = 0;
    __syncthreads();
    
    atomicAdd(&sum, val);
    __syncthreads();
    
    x[threadIdx.x] = val / sum;
}

// Kernel for computing output gradients
__global__ void computeOutputGradients(double* d_output, double* d_target, double* d_d_output, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        d_d_output[i] = d_output[i] - d_target[i];
    }
}

// Kernel for computing hidden gradients
__global__ void computeHiddenGradients(double* d_hidden, double* d_W2, double* d_d_output, 
                                      double* d_d_hidden, int hidden_size, int output_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < hidden_size) {
        double sum = 0;
        for (int j = 0; j < output_size; j++) {
            sum += d_W2[j * hidden_size + i] * d_d_output[j];
        }
        d_d_hidden[i] = sum * (d_hidden[i] > 0);  // ReLU derivative
    }
}

// Kernel for updating weights
__global__ void updateWeights(double* d_W, double* d_delta, double* d_activation, 
                             int rows, int cols, double learning_rate) {
    int row = blockIdx.x;
    int col = threadIdx.x;
    
    if (row < rows && col < cols) {
        atomicAdd(&d_W[row * cols + col], -learning_rate * d_delta[row] * d_activation[col]);
    }
}

// Kernel for updating biases
__global__ void updateBiases(double* d_b, double* d_delta, int size, double learning_rate) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        d_b[i] -= learning_rate * d_delta[i];
    }
}

// GPU Forward pass
void forwardGPU(NeuralNetwork* net, double* input, double* output) {
    // Copy input to device (only necessary if input is on host)
    cudaMemcpy(net->d_input, input, INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    
    // Launch kernels with optimal block sizes
    dim3 blockSize(256);
    dim3 gridSize((HIDDEN_SIZE + blockSize.x - 1) / blockSize.x);
    
    // Compute hidden layer with ReLU
    computeHiddenLayer<<<gridSize, blockSize>>>(net->d_input, net->d_W1, net->d_b1, 
                                              net->d_hidden, INPUT_SIZE, HIDDEN_SIZE);
    
    // Compute output layer
    gridSize = dim3((OUTPUT_SIZE + blockSize.x - 1) / blockSize.x);
    computeOutputLayer<<<gridSize, blockSize>>>(net->d_hidden, net->d_W2, net->d_b2, 
                                              net->d_output, HIDDEN_SIZE, OUTPUT_SIZE);
    
    // Apply softmax
    softmaxKernel<<<1, OUTPUT_SIZE>>>(net->d_output, OUTPUT_SIZE);
    
    // Copy result back to host (only if needed)
    cudaMemcpy(output, net->d_output, OUTPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost);
}

// GPU Backward pass
void backwardGPU(NeuralNetwork* net, double* input, double* target) {
    // Copy input and target to device (only necessary if they're on host)
    cudaMemcpy(net->d_input, input, INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(net->d_d_output, target, OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    
    dim3 blockSize(256);
    dim3 gridSize;
    
    // Compute output gradients
    gridSize = dim3((OUTPUT_SIZE + blockSize.x - 1) / blockSize.x);
    computeOutputGradients<<<gridSize, blockSize>>>(net->d_output, net->d_d_output, 
                                                  net->d_d_output, OUTPUT_SIZE);
    
    // Compute hidden gradients
    gridSize = dim3((HIDDEN_SIZE + blockSize.x - 1) / blockSize.x);
    computeHiddenGradients<<<gridSize, blockSize>>>(net->d_hidden, net->d_W2, net->d_d_output,
                                                  net->d_d_hidden, HIDDEN_SIZE, OUTPUT_SIZE);
    
    // Update weights and biases
    // W2 update
    gridSize = dim3(OUTPUT_SIZE);
    blockSize = dim3(HIDDEN_SIZE);
    updateWeights<<<gridSize, blockSize>>>(net->d_W2, net->d_d_output, net->d_hidden,
                                         OUTPUT_SIZE, HIDDEN_SIZE, LEARNING_RATE);
    
    // W1 update
    gridSize = dim3(HIDDEN_SIZE);
    blockSize = dim3(INPUT_SIZE);
    updateWeights<<<gridSize, blockSize>>>(net->d_W1, net->d_d_hidden, net->d_input,
                                         HIDDEN_SIZE, INPUT_SIZE, LEARNING_RATE);
    
    // b2 update
    gridSize = dim3((OUTPUT_SIZE + blockSize.x - 1) / blockSize.x);
    blockSize = dim3(256);
    updateBiases<<<gridSize, blockSize>>>(net->d_b2, net->d_d_output, OUTPUT_SIZE, LEARNING_RATE);
    
    // b1 update
    gridSize = dim3((HIDDEN_SIZE + blockSize.x - 1) / blockSize.x);
    updateBiases<<<gridSize, blockSize>>>(net->d_b1, net->d_d_hidden, HIDDEN_SIZE, LEARNING_RATE);
}

// Update createNetwork to allocate additional buffers
NeuralNetwork* createNetwork() {
    if (VERBOSE) printf("Creating neural network...\n");
    NeuralNetwork* net = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    if (!net) {
        if (VERBOSE) printf("Failed to allocate neural network\n");
        exit(1);
    }
    
    // Allocate host memory
    net->W1 = allocateMatrix(HIDDEN_SIZE, INPUT_SIZE);
    net->W2 = allocateMatrix(OUTPUT_SIZE, HIDDEN_SIZE);
    net->b1 = (double*)calloc(HIDDEN_SIZE, sizeof(double));
    net->b2 = (double*)calloc(OUTPUT_SIZE, sizeof(double));

    // Initialize weights
    srand(time(NULL));
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            net->W1[i * INPUT_SIZE + j] = ((double)rand() / RAND_MAX) * 0.01;
        }
    }

    for (int i = 0; i < OUTPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            net->W2[i * HIDDEN_SIZE + j] = ((double)rand() / RAND_MAX) * 0.01;
        }
    }

    // Allocate device memory
    cudaMalloc(&net->d_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(double));
    cudaMalloc(&net->d_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double));
    cudaMalloc(&net->d_b1, HIDDEN_SIZE * sizeof(double));
    cudaMalloc(&net->d_b2, OUTPUT_SIZE * sizeof(double));
    cudaMalloc(&net->d_input, INPUT_SIZE * sizeof(double));
    cudaMalloc(&net->d_hidden, HIDDEN_SIZE * sizeof(double));
    cudaMalloc(&net->d_output, OUTPUT_SIZE * sizeof(double));

    // Allocate additional device memory for gradients
    cudaMalloc(&net->d_d_output, OUTPUT_SIZE * sizeof(double));
    cudaMalloc(&net->d_d_hidden, HIDDEN_SIZE * sizeof(double));
    
    // ... rest of the existing code ...
    
    // Copy initial values to device
    cudaMemcpy(net->d_W1, net->W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(net->d_W2, net->W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(net->d_b1, net->b1, HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(net->d_b2, net->b2, OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);

    return net;
}

// Update freeNetwork to free additional buffers
void freeNetwork(NeuralNetwork* net) {
    if (VERBOSE) printf("Freeing neural network...\n");
    
    // Free host memory
    free(net->W1);
    free(net->W2);
    free(net->b1);
    free(net->b2);
    
    // Free device memory
    cudaFree(net->d_d_output);
    cudaFree(net->d_d_hidden);
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

void train(NeuralNetwork* net, double* images, double* labels, int numImages) {
    if (VERBOSE) printf("\nStarting training...\n");
    
    cudaEvent_t total_start, total_stop;
    create_timer(&total_start, &total_stop);
    start_timer(total_start);
    
    // Allocate device memory for batch processing
    double* d_images;
    double* d_labels;
    cudaMalloc(&d_images, numImages * INPUT_SIZE * sizeof(double));
    cudaMalloc(&d_labels, numImages * OUTPUT_SIZE * sizeof(double));
    
    // Copy all data to device once
    cudaMemcpy(d_images, images, numImages * INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, labels, numImages * OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        cudaEvent_t epoch_start, epoch_stop;
        create_timer(&epoch_start, &epoch_stop);
        start_timer(epoch_start);

        double loss = 0.0;
        int correct = 0;

        if (VERBOSE) printf("\nEpoch %d/%d\n", epoch+1, EPOCHS);
        
        for (int i = 0; i < numImages; i++) {
            if (VERBOSE && i % 1000 == 0) printf("Processing sample %d\n", i);
            
            double output[OUTPUT_SIZE];
            
            // Get pointers to current image and label in device memory
            double* current_image = d_images + i * INPUT_SIZE;
            double* current_label = d_labels + i * OUTPUT_SIZE;
            
            // Forward pass on GPU
            forwardGPU(net, current_image, output);
            
            // Backward pass on GPU
            backwardGPU(net, current_image, current_label);

            // Compute loss & accuracy (on host)
            for (int k = 0; k < OUTPUT_SIZE; k++) {
                loss -= current_label[k] * log(output[k]);
            }
            
            int pred = 0, actual = 0;
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                if (output[j] > output[pred]) pred = j;
                if (current_label[j] > current_label[actual]) actual = j;
            }
            if (pred == actual) correct++;
        }

        printf("Epoch %d - Loss: %.4f - Train Accuracy: %.2f%% - Time: %.3fs\n",
               epoch + 1, loss / numImages, (correct / (double)numImages) * 100, 
               stop_timer(epoch_start, epoch_stop));
    }
    
    // Free device memory
    cudaFree(d_images);
    cudaFree(d_labels);
    
    printf("Total training time: %.3fs\n", stop_timer(total_start, total_stop));
    if (VERBOSE) printf("Training completed\n");
}