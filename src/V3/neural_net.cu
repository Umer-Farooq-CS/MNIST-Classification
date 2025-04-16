#include "neural_net.h"
#include "utils.h"
#include <stdlib.h>
#include <time.h>
#define BLOCK_SIZE 256

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

void backward(NeuralNetwork* net, double* d_input, double* d_target) {
    if (VERBOSE) printf("\nStarting GPU backward pass...\n");

    int blockSize = 256;
    int numBlocksOutput = (OUTPUT_SIZE + blockSize - 1) / blockSize;
    int numBlocksHidden = (HIDDEN_SIZE + blockSize - 1) / blockSize;
    int numBlocksW2 = ((OUTPUT_SIZE * HIDDEN_SIZE) + blockSize - 1) / blockSize;
    int numBlocksW1 = ((HIDDEN_SIZE * INPUT_SIZE) + blockSize - 1) / blockSize;
    
    // Step 1: Compute output gradients: d_output = d_output - d_target.
    computeDOutputKernel<<<numBlocksOutput, blockSize>>>(net->d_output, d_target, OUTPUT_SIZE);
    checkCudaError(cudaGetLastError(), "Kernel launch: computeDOutputKernel");
    checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize after computeDOutputKernel");
    
    // Step 2: Save the forward hidden activations.
    double* d_hidden_forward;
    checkCudaError(cudaMalloc(&d_hidden_forward, HIDDEN_SIZE * sizeof(double)), "cudaMalloc d_hidden_forward");
    checkCudaError(cudaMemcpy(d_hidden_forward, net->d_hidden, HIDDEN_SIZE * sizeof(double), cudaMemcpyDeviceToDevice), "cudaMemcpy d_hidden_forward");
    
    // Allocate a temporary device array for the hidden gradients.
    double* d_hidden_grad;
    checkCudaError(cudaMalloc(&d_hidden_grad, HIDDEN_SIZE * sizeof(double)), "cudaMalloc d_hidden_grad");
    
    // Step 3: Compute hidden layer gradients.
    computeDHiddenKernel<<<numBlocksHidden, blockSize>>>(net->d_W2, net->d_output, d_hidden_forward, d_hidden_grad, HIDDEN_SIZE, OUTPUT_SIZE);
    checkCudaError(cudaGetLastError(), "Kernel launch: computeDHiddenKernel");
    checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize after computeDHiddenKernel");
    
    // Step 4: Update W2 using the forward hidden activations.
    updateW2Kernel<<<numBlocksW2, blockSize>>>(net->d_W2, net->d_output, d_hidden_forward, HIDDEN_SIZE, OUTPUT_SIZE, LEARNING_RATE);
    checkCudaError(cudaGetLastError(), "Kernel launch: updateW2Kernel");
    checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize after updateW2Kernel");
    
    // Step 5: Update W1 using the computed hidden gradients.
    updateW1Kernel<<<numBlocksW1, blockSize>>>(net->d_W1, d_hidden_grad, d_input, INPUT_SIZE, HIDDEN_SIZE, LEARNING_RATE);
    checkCudaError(cudaGetLastError(), "Kernel launch: updateW1Kernel");
    checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize after updateW1Kernel");
    
    // Step 6: Update biases.
    updateBiasesKernel<<<numBlocksOutput, blockSize>>>(net->d_b2, net->d_output, OUTPUT_SIZE, LEARNING_RATE);
    checkCudaError(cudaGetLastError(), "Kernel launch: updateBiasesKernel (d_b2)");
    checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize after updateBiasesKernel (d_b2)");
    
    updateBiasesKernel<<<numBlocksHidden, blockSize>>>(net->d_b1, d_hidden_grad, HIDDEN_SIZE, LEARNING_RATE);
    checkCudaError(cudaGetLastError(), "Kernel launch: updateBiasesKernel (d_b1)");
    checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize after updateBiasesKernel (d_b1)");
    
    // Free temporary arrays.
    checkCudaError(cudaFree(d_hidden_forward), "cudaFree d_hidden_forward");
    checkCudaError(cudaFree(d_hidden_grad), "cudaFree d_hidden_grad");

    if (VERBOSE) printf("GPU backward pass completed\n");
}


// Backpropagation on host
void backward(NeuralNetwork* net, double* input, double* hidden, double* output, double* target) {
    if (VERBOSE) printf("\nStarting backward pass...\n");
    double d_output[OUTPUT_SIZE], d_hidden[HIDDEN_SIZE];

    // Compute output layer gradient.
    if (VERBOSE) printf("Computing output gradients...\n");
    for (int i = 0; i < OUTPUT_SIZE; i++)
        d_output[i] = output[i] - target[i];
    
    if (VERBOSE) {
        printf("Output gradients: ");
        for (int i = 0; i < OUTPUT_SIZE; i++) printf("%.4f ", d_output[i]);
        printf("\n");
    }

    // Compute hidden layer gradient.
    if (VERBOSE) printf("Computing hidden gradients...\n");
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        d_hidden[i] = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++)
            d_hidden[i] += net->W2[j * HIDDEN_SIZE + i] * d_output[j];
        d_hidden[i] *= (hidden[i] > 0);
    }
    
    if (VERBOSE) {
        printf("Hidden gradients (first 5): ");
        for (int i = 0; i < 5; i++) printf("%.4f ", d_hidden[i]);
        printf("\n");
    }

    // Update weights (gradient descent).
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

    // Update biases.
    if (VERBOSE) printf("Updating biases...\n");
    for (int i = 0; i < OUTPUT_SIZE; i++)
        net->b2[i] -= LEARNING_RATE * d_output[i];

    for (int i = 0; i < HIDDEN_SIZE; i++)
        net->b1[i] -= LEARNING_RATE * d_hidden[i];
    
    // Update device weights.
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

// ----------------------------------------------------------------------------
// Optimized kernel for matrix–vector multiplication using shared memory reduction.
// Each block handles one output row. Each thread in the block computes a partial sum 
// over a subset of columns.
__global__ void matrixVectorMultiplySM(const double* __restrict__ W, 
                                         const double* __restrict__ x, 
                                         const double* __restrict__ b, 
                                         double* __restrict__ y, 
                                         int rows, int cols) {
    extern __shared__ double sdata[];
    int row = blockIdx.x;  // each block does one row.
    int tid = threadIdx.x;
    double sum = 0.0;
    // Each thread sums over columns strided by blockDim.x.
    for (int j = tid; j < cols; j += blockDim.x) {
        sum += W[row * cols + j] * x[j];
    }
    sdata[tid] = sum;
    __syncthreads();
    // Perform parallel reduction in shared memory.
    for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        // Add the corresponding bias and write out result.
        y[row] = sdata[0] + b[row];
    }
}

// ----------------------------------------------------------------------------
// Optimized softmax kernel using shared memory reduction for the sum.
// This version first computes the maximum value (for numerical stability),
// then computes exponentials and reduces them.
__global__ void softmaxKernelOpt(double* x, int size) {
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    // Use the first block (assume one block launched for the vector)
    // First, find max value using parallel reduction.
    double max_val = -1e20;
    for (int i = tid; i < size; i += blockDim.x) {
        double tmp = x[i];
        if (tmp > max_val) max_val = tmp;
    }
    sdata[tid] = max_val;
    __syncthreads();
    // Reduction to get global max.
    for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    max_val = sdata[0];
    // Now compute exponentials (using max_val for stability)
    double sum = 0.0;
    for (int i = tid; i < size; i += blockDim.x) {
        double exp_val = exp(x[i] - max_val);
        x[i] = exp_val;  // store exponentials temporarily
        sum += exp_val;
    }
    sdata[tid] = sum;
    __syncthreads();
    // Reduce to get the total sum.
    for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid+s];
        __syncthreads();
    }
    double total = sdata[0];
    // Finally normalize.
    for (int i = tid; i < size; i += blockDim.x) {
        x[i] /= total;
    }
}

// ----------------------------------------------------------------------------
// Forward pass function rewritten to use asynchronous copies and a stream.
void forward(NeuralNetwork* net, double* input, double* hidden, double* output) {
    if (VERBOSE) printf("\nStarting forward pass...\n");
    
    cudaStream_t stream;
    checkCudaError(cudaStreamCreate(&stream), "cudaStreamCreate");

    // Asynchronously copy input to device.
    checkCudaError(cudaMemcpyAsync(net->d_input, input, INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice, stream),
                   "cudaMemcpyAsync input");

    int blockSize = BLOCK_SIZE;
    int numBlocksHidden = HIDDEN_SIZE;  // one block per row in hidden layer.
    size_t sharedSize = blockSize * sizeof(double);
    
    // Launch optimized matrix-vector multiplication for hidden layer.
    matrixVectorMultiplySM<<<numBlocksHidden, blockSize, sharedSize, stream>>>(
        net->d_W1, net->d_input, net->d_b1, net->d_hidden, HIDDEN_SIZE, INPUT_SIZE);
    checkCudaError(cudaGetLastError(), "Kernel launch: matrixVectorMultiplySM (hidden)");

    // Launch ReLU kernel (we assume it is memory‐bound so shared memory is not needed here).
    int numBlocks = (HIDDEN_SIZE + blockSize - 1) / blockSize;
    relu_kernel<<<numBlocks, blockSize, 0, stream>>>(net->d_hidden, HIDDEN_SIZE);
    checkCudaError(cudaGetLastError(), "Kernel launch: relu_kernel");

    // For the output layer, use the same optimized kernel.
    int numBlocksOutput = OUTPUT_SIZE;  // one block per output neuron.
    matrixVectorMultiplySM<<<numBlocksOutput, blockSize, sharedSize, stream>>>(
        net->d_W2, net->d_hidden, net->d_b2, net->d_output, OUTPUT_SIZE, HIDDEN_SIZE);
    checkCudaError(cudaGetLastError(), "Kernel launch: matrixVectorMultiplySM (output)");

    // Launch optimized softmax kernel.
    // Assume one block is sufficient if OUTPUT_SIZE is small.
    softmaxKernelOpt<<<1, BLOCK_SIZE, BLOCK_SIZE * sizeof(double), stream>>>(net->d_output, OUTPUT_SIZE);
    checkCudaError(cudaGetLastError(), "Kernel launch: softmaxKernelOpt");

    // Copy results back to host asynchronously.
    checkCudaError(cudaMemcpyAsync(hidden, net->d_hidden, HIDDEN_SIZE * sizeof(double), cudaMemcpyDeviceToHost, stream),
                   "cudaMemcpyAsync hidden");
    checkCudaError(cudaMemcpyAsync(output, net->d_output, OUTPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost, stream),
                   "cudaMemcpyAsync output");

    // Wait for stream to complete.
    checkCudaError(cudaStreamSynchronize(stream), "cudaStreamSynchronize forward");
    checkCudaError(cudaStreamDestroy(stream), "cudaStreamDestroy forward");

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

// Kernel to compute d_output = d_output - d_target, for the output layer gradients.
__global__ void computeDOutputKernel(double* d_output, const double* d_target, int outputSize) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < outputSize) {
        d_output[i] = d_output[i] - d_target[i];
    }
}

// Kernel to compute d_hidden_grad for each hidden neuron.
// Uses the layer-2 weights and the computed d_output.
// The forward activation (before replacing with gradients) is in d_hidden_forward.
__global__ void computeDHiddenKernel(const double* d_W2, const double* d_output, 
                                       const double* d_hidden_forward, double* d_hidden_grad,
                                       int hiddenSize, int outputSize) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < hiddenSize) {
        double sum = 0.0;
        for (int j = 0; j < outputSize; j++) {
            sum += d_W2[j * hiddenSize + i] * d_output[j];
        }
        // ReLU derivative: if forward hidden activation > 0, derivative is 1 else 0.
        d_hidden_grad[i] = (d_hidden_forward[i] > 0.0) ? sum : 0.0;
    }
}

// Kernel to update the weights for the output layer (W2).
// Uses the computed output gradients and the forward hidden activations.
__global__ void updateW2Kernel(double* d_W2, const double* d_output, 
                               const double* d_hidden_forward, int hiddenSize, 
                               int outputSize, double learning_rate) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outputSize * hiddenSize;
    if (idx < total) {
        int out_idx = idx / hiddenSize;  // index for output neuron
        int hid_idx = idx % hiddenSize;    // index for hidden neuron
        d_W2[idx] -= learning_rate * d_output[out_idx] * d_hidden_forward[hid_idx];
    }
}

// Kernel to update the weights for the hidden layer (W1).
// Uses the computed hidden gradients and the input.
__global__ void updateW1Kernel(double* d_W1, const double* d_hidden_grad, 
                               const double* d_input, int inputSize, int hiddenSize, 
                               double learning_rate) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = hiddenSize * inputSize;
    if (idx < total) {
        int hid_idx = idx / inputSize;
        int in_idx  = idx % inputSize;
        d_W1[idx] -= learning_rate * d_hidden_grad[hid_idx] * d_input[in_idx];
    }
}

// Kernel to update biases, used for both layers.
__global__ void updateBiasesKernel(double* d_bias, const double* d_grad, int size, double learning_rate) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        d_bias[i] -= learning_rate * d_grad[i];
    }
}


// We now show an example of using asynchronous streams in the training loop.
// Instead of processing one sample at a time strictly sequentially, we use multiple streams.
// In a real system you would batch many samples together.
void train(NeuralNetwork* net, double* images, double* labels, int numImages) {
    if (VERBOSE) printf("\nStarting training...\n");

    cudaEvent_t total_start, total_stop;
    create_timer(&total_start, &total_stop);
    start_timer(total_start);

    // Create a stream pool. For demonstration, we use two streams.
    const int numStreams = 2;
    cudaStream_t streams[numStreams];
    for (int s = 0; s < numStreams; s++) {
        checkCudaError(cudaStreamCreate(&streams[s]), "cudaStreamCreate in train");
    }

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        cudaEvent_t epoch_start, epoch_stop;
        create_timer(&epoch_start, &epoch_stop);
        start_timer(epoch_start);

        double loss = 0.0;
        int correct = 0;

        if (VERBOSE) printf("\nEpoch %d/%d\n", epoch+1, EPOCHS);

        // For simplicity, process samples in a loop and alternate streams.
        for (int i = 0; i < numImages; i++) {
            int streamId = i % numStreams;
            
            // Allocate device memory for target and copy asynchronously using the stream.
            double* d_target;
            checkCudaError(cudaMalloc(&d_target, OUTPUT_SIZE * sizeof(double)), "cudaMalloc d_target in train");
            checkCudaError(cudaMemcpyAsync(d_target, &labels[i * OUTPUT_SIZE], OUTPUT_SIZE * sizeof(double), 
                                             cudaMemcpyHostToDevice, streams[streamId]), "cudaMemcpyAsync d_target in train");

            double hidden[HIDDEN_SIZE], output[OUTPUT_SIZE];
            // Launch forward pass on the chosen stream.
            // (For simplicity, here we call forward which creates its own stream.
            // In a real implementation you would modify forward to accept a stream parameter.)
            forward(net, &images[i * INPUT_SIZE], hidden, output);

            // Call the GPU backward pass (using the stream as appropriate).
            backward(net, net->d_input, d_target);

            // Compute loss & accuracy on the host from the output.
            for (int k = 0; k < OUTPUT_SIZE; k++) {
                loss -= labels[i * OUTPUT_SIZE + k] * log(output[k]);
            }
            int pred = 0, actual = 0;
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                if (output[j] > output[pred]) pred = j;
                if (labels[i * OUTPUT_SIZE + j] > labels[i * OUTPUT_SIZE + actual]) actual = j;
            }
            if (pred == actual) correct++;

            checkCudaError(cudaFree(d_target), "cudaFree d_target in train");
        }

        printf("Epoch %d - Loss: %.4f - Train Accuracy: %.2f%% - Time: %.3fs\n",
               epoch + 1, loss / numImages, (correct / (double)numImages) * 100, 
               stop_timer(epoch_start, epoch_stop));
    }

    // Destroy streams.
    for (int s = 0; s < numStreams; s++) {
        checkCudaError(cudaStreamDestroy(streams[s]), "cudaStreamDestroy in train");
    }

    printf("Total training time: %.3fs\n", stop_timer(total_start, total_stop));

    if (VERBOSE) printf("Training completed\n");
}

// ----------------------------------------------------------------------------
// The evaluate and freeNetwork functions remain similar, though you can use 
// asynchronous copies in evaluate if desired.
// ----------------------------------------------------------------------------

void evaluate(NeuralNetwork* net, double* images, double* labels, int numImages) {
    if (VERBOSE) printf("\nStarting evaluation...\n");
    int correct = 0;
    
    for (int i = 0; i < numImages; i++) {
        double hidden[HIDDEN_SIZE], output[OUTPUT_SIZE];
        forward(net, &images[i * INPUT_SIZE], hidden, output);
        
        int pred = 0, actual = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            if (output[j] > output[pred]) pred = j;
            if (labels[i * OUTPUT_SIZE + j] > labels[i * OUTPUT_SIZE + actual]) actual = j;
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
    checkCudaError(cudaFree(net->d_W1), "cudaFree d_W1");
    checkCudaError(cudaFree(net->d_W2), "cudaFree d_W2");
    checkCudaError(cudaFree(net->d_b1), "cudaFree d_b1");
    checkCudaError(cudaFree(net->d_b2), "cudaFree d_b2");
    checkCudaError(cudaFree(net->d_input), "cudaFree d_input");
    checkCudaError(cudaFree(net->d_hidden), "cudaFree d_hidden");
    checkCudaError(cudaFree(net->d_output), "cudaFree d_output");
    
    free(net);
    if (VERBOSE) printf("Neural network freed\n");
}
