#include "neural_net.h"
#include "utils.h"
#include <cuda_fp16.h>
#include <mma.h>
#include <curand_kernel.h>
#include <cfloat>
using namespace nvcuda::wmma;

// Initialize weights using CURAND directly on GPU
__global__ void initWeightsKernel(float* W, int n, float scale, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        W[idx] = curand_uniform(&state) * scale;
    }
}

// Conversion kernel: float to half
__global__ void floatToHalfKernel(const float* in, half* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = __float2half(in[idx]);
    }
}

// Conversion kernel: half to float
__global__ void halfToFloatKernel(const half* in, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = __half2float(in[idx]);
    }
}

// Tensor Core optimized matrix-vector multiplication
__global__ void tc_matrixVectorMultiply(const half* W, const half* x, const half* b, float* y, int rows, int cols) {
    // Each thread block computes one row of the output
    const int row = blockIdx.x;
    
    // Declare the fragments
    fragment<matrix_a, TC_TILE_M, TC_TILE_N, TC_TILE_K, half, row_major> a_frag;
    fragment<matrix_b, TC_TILE_M, TC_TILE_N, TC_TILE_K, half, col_major> b_frag;
    fragment<accumulator, TC_TILE_M, TC_TILE_N, TC_TILE_K, float> acc_frag;
    
    // Initialize the accumulator to zero
    fill_fragment(acc_frag, 0.0f);
    
    // Loop over the columns of W and rows of x in TC_TILE_K steps
    for (int k = 0; k < cols; k += TC_TILE_K) {
        int k_end = min(k + TC_TILE_K, cols);
        
        // Load the tile of W
        load_matrix_sync(a_frag, W + row * cols + k, cols);
        
        // Load the tile of x
        load_matrix_sync(b_frag, x + k, 1);
        
        // Perform the matrix multiplication
        mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }
    
    // Store the accumulated result
    // float result = get_element(acc_frag, 0, 0);
    
    float result;
    memcpy(&result, &acc_frag.x[0], sizeof(float));
    


    // Add bias if provided
    if (b != nullptr) {
        result += __half2float(b[row]);
    }
    
    // Store the result
    y[row] = result;
}

// ReLU activation kernel (FP16 version)
__global__ void relu_kernel(half* x, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        x[idx] = __hgt(x[idx], __float2half(0.0f)) ? x[idx] : __float2half(0.0f);
    }
}

// Softmax activation kernel (FP32 version for stability)
__global__ void softmaxKernelOpt(float* x, int size) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    
    // Find max value for numerical stability
    float max_val = -FLT_MAX;
    for (int i = tid; i < size; i += blockDim.x) {
        if (x[i] > max_val) max_val = x[i];
    }
    sdata[tid] = max_val;
    __syncthreads();
    
    // Reduction to find global max
    for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    max_val = sdata[0];
    
    // Compute exponentials and sum
    float sum = 0.0f;
    for (int i = tid; i < size; i += blockDim.x) {
        float exp_val = expf(x[i] - max_val);
        x[i] = exp_val;
        sum += exp_val;
    }
    sdata[tid] = sum;
    __syncthreads();
    
    // Reduction to find sum
    for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    sum = sdata[0];
    
    // Normalize
    for (int i = tid; i < size; i += blockDim.x) {
        x[i] /= sum;
    }
}

// Kernel to compute d_output = d_output - d_target, for the output layer gradients
__global__ void computeDOutputKernel(float* d_output, const half* d_target, int outputSize) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < outputSize) {
        d_output[i] = d_output[i] - __half2float(d_target[i]);
    }
}

// Kernel to compute d_hidden_grad for each hidden neuron
__global__ void computeDHiddenKernel(const half* d_W2, const float* d_output, 
                                   const half* d_hidden_forward, float* d_hidden_grad,
                                   int hiddenSize, int outputSize) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < hiddenSize) {
        float sum = 0.0f;
        for (int j = 0; j < outputSize; j++) {
            sum += __half2float(d_W2[j * hiddenSize + i]) * d_output[j];
        }
        // ReLU derivative: if forward hidden activation > 0, derivative is 1 else 0
        d_hidden_grad[i] = (__half2float(d_hidden_forward[i]) > 0.0f) ? sum : 0.0f;
    }
}

// Kernel to update the weights for the output layer (W2)
__global__ void updateW2Kernel(half* d_W2, const float* d_output, 
                             const half* d_hidden_forward, int hiddenSize, 
                             int outputSize, float learning_rate) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outputSize * hiddenSize;
    if (idx < total) {
        int out_idx = idx / hiddenSize;  // index for output neuron
        int hid_idx = idx % hiddenSize;  // index for hidden neuron
        d_W2[idx] = __float2half(__half2float(d_W2[idx]) - 
                                learning_rate * d_output[out_idx] * __half2float(d_hidden_forward[hid_idx]));
    }
}

// Kernel to update the weights for the hidden layer (W1)
__global__ void updateW1Kernel(half* d_W1, const float* d_hidden_grad, 
                             const half* d_input, int inputSize, int hiddenSize, 
                             float learning_rate) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = hiddenSize * inputSize;
    if (idx < total) {
        int hid_idx = idx / inputSize;
        int in_idx  = idx % inputSize;
        d_W1[idx] = __float2half(__half2float(d_W1[idx]) - 
                    learning_rate * d_hidden_grad[hid_idx] * __half2float(d_input[in_idx]));
    }
}

// Kernel to update biases, used for both layers
__global__ void updateBiasesKernel(float* d_bias, const float* d_grad, int size, float learning_rate) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        d_bias[i] -= learning_rate * d_grad[i];
    }
}

NeuralNetwork* createNetwork() {
    if (VERBOSE) printf("Creating neural network with Tensor Core support...\n");
    NeuralNetwork* net = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    if (!net) {
        if (VERBOSE) printf("Failed to allocate neural network\n");
        exit(1);
    }
    
    // Allocate pinned host arrays (FP32)
    checkCudaError(cudaMallocHost((void**)&net->W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(float)), "cudaMallocHost W1");
    checkCudaError(cudaMallocHost((void**)&net->W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float)), "cudaMallocHost W2");
    checkCudaError(cudaMallocHost((void**)&net->b1, HIDDEN_SIZE * sizeof(float)), "cudaMallocHost b1");
    checkCudaError(cudaMallocHost((void**)&net->b2, OUTPUT_SIZE * sizeof(float)), "cudaMallocHost b2");
    
    // Initialize host arrays to zero for biases
    memset(net->b1, 0, HIDDEN_SIZE * sizeof(float));
    memset(net->b2, 0, OUTPUT_SIZE * sizeof(float));

    // Allocate device memory for weights and biases (FP32)
    checkCudaError(cudaMalloc(&net->d_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(float)), "cudaMalloc d_W1");
    checkCudaError(cudaMalloc(&net->d_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float)), "cudaMalloc d_W2");
    checkCudaError(cudaMalloc(&net->d_b1, HIDDEN_SIZE * sizeof(float)), "cudaMalloc d_b1");
    checkCudaError(cudaMalloc(&net->d_b2, OUTPUT_SIZE * sizeof(float)), "cudaMalloc d_b2");
    
    // Allocate FP16 versions for Tensor Core operations
    checkCudaError(cudaMalloc(&net->d_W1_half, HIDDEN_SIZE * INPUT_SIZE * sizeof(half)), "cudaMalloc d_W1_half");
    checkCudaError(cudaMalloc(&net->d_W2_half, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(half)), "cudaMalloc d_W2_half");
    checkCudaError(cudaMalloc(&net->d_b1_half, HIDDEN_SIZE * sizeof(half)), "cudaMalloc d_b1_half");
    checkCudaError(cudaMalloc(&net->d_b2_half, OUTPUT_SIZE * sizeof(half)), "cudaMalloc d_b2_half");
    checkCudaError(cudaMalloc(&net->d_input, INPUT_SIZE * sizeof(half)), "cudaMalloc d_input");
    checkCudaError(cudaMalloc(&net->d_hidden, HIDDEN_SIZE * sizeof(half)), "cudaMalloc d_hidden");
    checkCudaError(cudaMalloc(&net->d_output, OUTPUT_SIZE * sizeof(half)), "cudaMalloc d_output");

    // Initialize weights directly on GPU (FP32)
    int totalW1 = HIDDEN_SIZE * INPUT_SIZE;
    int totalW2 = OUTPUT_SIZE * HIDDEN_SIZE;
    int threads = BLOCK_SIZE;
    int blocksW1 = (totalW1 + threads - 1) / threads;
    int blocksW2 = (totalW2 + threads - 1) / threads;
    unsigned long seed = (unsigned long) time(NULL);
    
    initWeightsKernel<<<blocksW1, threads>>>(net->d_W1, totalW1, 0.01f, seed);
    checkCudaError(cudaGetLastError(), "Kernel launch: initWeightsKernel d_W1");
    initWeightsKernel<<<blocksW2, threads>>>(net->d_W2, totalW2, 0.01f, seed + 1);
    checkCudaError(cudaGetLastError(), "Kernel launch: initWeightsKernel d_W2");

    // Initialize biases to zero (FP32)
    checkCudaError(cudaMemset(net->d_b1, 0, HIDDEN_SIZE * sizeof(float)), "cudaMemset d_b1");
    checkCudaError(cudaMemset(net->d_b2, 0, OUTPUT_SIZE * sizeof(float)), "cudaMemset d_b2");
    
    // Convert FP32 weights to FP16 for Tensor Core operations
    floatToHalfKernel<<<blocksW1, threads>>>(net->d_W1, net->d_W1_half, totalW1);
    floatToHalfKernel<<<blocksW2, threads>>>(net->d_W2, net->d_W2_half, totalW2);
    floatToHalfKernel<<<(HIDDEN_SIZE + threads - 1)/threads, threads>>>(net->d_b1, net->d_b1_half, HIDDEN_SIZE);
    floatToHalfKernel<<<(OUTPUT_SIZE + threads - 1)/threads, threads>>>(net->d_b2, net->d_b2_half, OUTPUT_SIZE);
    checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize after floatToHalfKernel");

    // Copy initialized weights back to host for debugging
    checkCudaError(cudaMemcpy(net->W1, net->d_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy W1");
    checkCudaError(cudaMemcpy(net->W2, net->d_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy W2");
    checkCudaError(cudaMemcpy(net->b1, net->d_b1, HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy b1");
    checkCudaError(cudaMemcpy(net->b2, net->d_b2, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy b2");

    if (VERBOSE) {
        printf("Weight initialization complete\n");
        printf("W1[0][0]: %.6f\n", net->W1[0]);
        printf("W2[0][0]: %.6f\n", net->W2[0]);
        printf("b1[0]: %.6f\n", net->b1[0]);
        printf("b2[0]: %.6f\n", net->b2[0]);
        printf("Neural network with Tensor Core support created successfully\n");
    }
    
    return net;
}

void forward(NeuralNetwork* net, float* input, float* hidden, float* output) {
    if (VERBOSE) printf("\nStarting forward pass with Tensor Cores...\n");
    
    cudaStream_t stream;
    checkCudaError(cudaStreamCreate(&stream), "cudaStreamCreate");

    // Convert input to FP16 and copy to device
    half* input_half = (half*)malloc(INPUT_SIZE * sizeof(half));
    for (int i = 0; i < INPUT_SIZE; i++) {
        input_half[i] = __float2half(input[i]);
    }
    checkCudaError(cudaMemcpyAsync(net->d_input, input_half, INPUT_SIZE * sizeof(half), 
                   cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync input");
    free(input_half);

    // Temporary FP32 output buffer on device
    float* d_hidden_float;
    checkCudaError(cudaMalloc(&d_hidden_float, HIDDEN_SIZE * sizeof(float)), "cudaMalloc d_hidden_float");
    
    // Launch Tensor Core matrix-vector multiplication for hidden layer
    int numBlocksHidden = (HIDDEN_SIZE + TC_TILE_M - 1) / TC_TILE_M;
    tc_matrixVectorMultiply<<<numBlocksHidden, 32, 0, stream>>>(
        net->d_W1_half, net->d_input, net->d_b1_half, d_hidden_float, HIDDEN_SIZE, INPUT_SIZE);
    checkCudaError(cudaGetLastError(), "Kernel launch: tc_matrixVectorMultiply (hidden)");

    // Convert hidden layer to FP16 for ReLU
    halfToFloatKernel<<<(HIDDEN_SIZE + BLOCK_SIZE - 1)/BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
        net->d_hidden, d_hidden_float, HIDDEN_SIZE);
    
    // Launch ReLU kernel (FP16)
    int numBlocks = (HIDDEN_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    relu_kernel<<<numBlocks, BLOCK_SIZE, 0, stream>>>(net->d_hidden, HIDDEN_SIZE);
    checkCudaError(cudaGetLastError(), "Kernel launch: relu_kernel");

    // Temporary FP32 output buffer for output layer
    float* d_output_float;
    checkCudaError(cudaMalloc(&d_output_float, OUTPUT_SIZE * sizeof(float)), "cudaMalloc d_output_float");
    
    // Launch Tensor Core matrix-vector multiplication for output layer
    int numBlocksOutput = (OUTPUT_SIZE + TC_TILE_M - 1) / TC_TILE_M;
    tc_matrixVectorMultiply<<<numBlocksOutput, 32, 0, stream>>>(
        net->d_W2_half, net->d_hidden, net->d_b2_half, d_output_float, OUTPUT_SIZE, HIDDEN_SIZE);
    checkCudaError(cudaGetLastError(), "Kernel launch: tc_matrixVectorMultiply (output)");

    // Convert output to FP16 for softmax
    halfToFloatKernel<<<(OUTPUT_SIZE + BLOCK_SIZE - 1)/BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
        net->d_output, d_output_float, OUTPUT_SIZE);
    
    // Launch softmax kernel (FP32 for stability)
    softmaxKernelOpt<<<1, BLOCK_SIZE, BLOCK_SIZE * sizeof(float), stream>>>(d_output_float, OUTPUT_SIZE);
    checkCudaError(cudaGetLastError(), "Kernel launch: softmaxKernelOpt");

    // Copy results back to host
    checkCudaError(cudaMemcpyAsync(hidden, d_hidden_float, HIDDEN_SIZE * sizeof(float), 
                   cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync hidden");
    checkCudaError(cudaMemcpyAsync(output, d_output_float, OUTPUT_SIZE * sizeof(float), 
                   cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync output");

    // Wait for stream to complete
    checkCudaError(cudaStreamSynchronize(stream), "cudaStreamSynchronize forward");
    
    // Free temporary buffers
    checkCudaError(cudaFree(d_hidden_float), "cudaFree d_hidden_float");
    checkCudaError(cudaFree(d_output_float), "cudaFree d_output_float");
    checkCudaError(cudaStreamDestroy(stream), "cudaStreamDestroy forward");

    if (VERBOSE) {
        printf("Post-ReLU hidden (first 5): ");
        for (int i = 0; i < 5; i++) printf("%.4f ", hidden[i]);
        printf("\n");
        printf("Post-softmax output: ");
        for (int i = 0; i < OUTPUT_SIZE; i++) printf("%.4f ", output[i]);
        printf("\n");
        printf("Forward pass with Tensor Cores completed\n");
    }
}

void backward(NeuralNetwork* net, half* d_input, half* d_target) {
    if (VERBOSE) printf("\nStarting GPU backward pass...\n");

    int blockSize = 256;
    int numBlocksOutput = (OUTPUT_SIZE + blockSize - 1) / blockSize;
    int numBlocksHidden = (HIDDEN_SIZE + blockSize - 1) / blockSize;
    int numBlocksW2 = ((OUTPUT_SIZE * HIDDEN_SIZE) + blockSize - 1) / blockSize;
    int numBlocksW1 = ((HIDDEN_SIZE * INPUT_SIZE) + blockSize - 1) / blockSize;
    
    // Temporary FP32 output buffer
    float* d_output_float;
    checkCudaError(cudaMalloc(&d_output_float, OUTPUT_SIZE * sizeof(float)), "cudaMalloc d_output_float");
    halfToFloatKernel<<<numBlocksOutput, blockSize>>>(net->d_output, d_output_float, OUTPUT_SIZE);
    
    // Step 1: Compute output gradients: d_output = d_output - d_target
    computeDOutputKernel<<<numBlocksOutput, blockSize>>>(d_output_float, d_target, OUTPUT_SIZE);
    checkCudaError(cudaGetLastError(), "Kernel launch: computeDOutputKernel");
    checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize after computeDOutputKernel");
    
    // Step 2: Save the forward hidden activations
    half* d_hidden_forward;
    checkCudaError(cudaMalloc(&d_hidden_forward, HIDDEN_SIZE * sizeof(half)), "cudaMalloc d_hidden_forward");
    checkCudaError(cudaMemcpy(d_hidden_forward, net->d_hidden, HIDDEN_SIZE * sizeof(half), cudaMemcpyDeviceToDevice), 
                   "cudaMemcpy d_hidden_forward");
    
    // Allocate a temporary device array for the hidden gradients
    float* d_hidden_grad;
    checkCudaError(cudaMalloc(&d_hidden_grad, HIDDEN_SIZE * sizeof(float)), "cudaMalloc d_hidden_grad");
    
    // Step 3: Compute hidden layer gradients
    computeDHiddenKernel<<<numBlocksHidden, blockSize>>>(net->d_W2_half, d_output_float, d_hidden_forward, 
                                                       d_hidden_grad, HIDDEN_SIZE, OUTPUT_SIZE);
    checkCudaError(cudaGetLastError(), "Kernel launch: computeDHiddenKernel");
    checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize after computeDHiddenKernel");
    
    // Step 4: Update W2 using the forward hidden activations
    updateW2Kernel<<<numBlocksW2, blockSize>>>(net->d_W2_half, d_output_float, d_hidden_forward, 
                                             HIDDEN_SIZE, OUTPUT_SIZE, LEARNING_RATE);
    checkCudaError(cudaGetLastError(), "Kernel launch: updateW2Kernel");
    checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize after updateW2Kernel");
    
    // Step 5: Update W1 using the computed hidden gradients
    updateW1Kernel<<<numBlocksW1, blockSize>>>(net->d_W1_half, d_hidden_grad, d_input, 
                                             INPUT_SIZE, HIDDEN_SIZE, LEARNING_RATE);
    checkCudaError(cudaGetLastError(), "Kernel launch: updateW1Kernel");
    checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize after updateW1Kernel");
    
    // Step 6: Update biases (FP32)
    updateBiasesKernel<<<numBlocksOutput, blockSize>>>(net->d_b2, d_output_float, OUTPUT_SIZE, LEARNING_RATE);
    checkCudaError(cudaGetLastError(), "Kernel launch: updateBiasesKernel (d_b2)");
    checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize after updateBiasesKernel (d_b2)");
    
    updateBiasesKernel<<<numBlocksHidden, blockSize>>>(net->d_b1, d_hidden_grad, HIDDEN_SIZE, LEARNING_RATE);
    checkCudaError(cudaGetLastError(), "Kernel launch: updateBiasesKernel (d_b1)");
    checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize after updateBiasesKernel (d_b1)");
    
    // Free temporary arrays
    checkCudaError(cudaFree(d_hidden_forward), "cudaFree d_hidden_forward");
    checkCudaError(cudaFree(d_hidden_grad), "cudaFree d_hidden_grad");
    checkCudaError(cudaFree(d_output_float), "cudaFree d_output_float");

    // Update FP32 master weights from FP16 copies
    floatToHalfKernel<<<(HIDDEN_SIZE * INPUT_SIZE + BLOCK_SIZE - 1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        net->d_W1, net->d_W1_half, HIDDEN_SIZE * INPUT_SIZE);
    floatToHalfKernel<<<(OUTPUT_SIZE * HIDDEN_SIZE + BLOCK_SIZE - 1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        net->d_W2, net->d_W2_half, OUTPUT_SIZE * HIDDEN_SIZE);
    floatToHalfKernel<<<(HIDDEN_SIZE + BLOCK_SIZE - 1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        net->d_b1, net->d_b1_half, HIDDEN_SIZE);
    floatToHalfKernel<<<(OUTPUT_SIZE + BLOCK_SIZE - 1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        net->d_b2, net->d_b2_half, OUTPUT_SIZE);

    if (VERBOSE) printf("GPU backward pass completed\n");
}

void train(NeuralNetwork* net, float* images, float* labels, int numImages) {
    if (VERBOSE) printf("\nStarting training...\n");

    cudaEvent_t total_start, total_stop;
    create_timer(&total_start, &total_stop);
    start_timer(total_start);

    // Create a stream pool
    const int numStreams = 2;
    cudaStream_t streams[numStreams];
    for (int s = 0; s < numStreams; s++) {
        checkCudaError(cudaStreamCreate(&streams[s]), "cudaStreamCreate in train");
    }

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        cudaEvent_t epoch_start, epoch_stop;
        create_timer(&epoch_start, &epoch_stop);
        start_timer(epoch_start);

        float loss = 0.0f;
        int correct = 0;

        if (VERBOSE) printf("\nEpoch %d/%d\n", epoch+1, EPOCHS);

        for (int i = 0; i < numImages; i++) {
            int streamId = i % numStreams;
            
            // Allocate device memory for target and copy asynchronously
            half* d_target;
            checkCudaError(cudaMalloc(&d_target, OUTPUT_SIZE * sizeof(half)), "cudaMalloc d_target in train");
            
            // Convert target to FP16
            half* target_half = (half*)malloc(OUTPUT_SIZE * sizeof(half));
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                target_half[j] = __float2half(labels[i * OUTPUT_SIZE + j]);
            }
            checkCudaError(cudaMemcpyAsync(d_target, target_half, OUTPUT_SIZE * sizeof(half), 
                           cudaMemcpyHostToDevice, streams[streamId]), "cudaMemcpyAsync d_target in train");
            free(target_half);

            float hidden[HIDDEN_SIZE], output[OUTPUT_SIZE];
            forward(net, &images[i * INPUT_SIZE], hidden, output);

            // Convert input to FP16 for backward pass
            half* d_input_half;
            checkCudaError(cudaMalloc(&d_input_half, INPUT_SIZE * sizeof(half)), "cudaMalloc d_input_half in train");
            half* input_half = (half*)malloc(INPUT_SIZE * sizeof(half));
            for (int j = 0; j < INPUT_SIZE; j++) {
                input_half[j] = __float2half(images[i * INPUT_SIZE + j]);
            }
            checkCudaError(cudaMemcpyAsync(d_input_half, input_half, INPUT_SIZE * sizeof(half), 
                           cudaMemcpyHostToDevice, streams[streamId]), "cudaMemcpyAsync d_input_half in train");
            free(input_half);

            backward(net, d_input_half, d_target);

            // Compute loss & accuracy
            for (int k = 0; k < OUTPUT_SIZE; k++) {
                loss -= labels[i * OUTPUT_SIZE + k] * logf(output[k]);
            }
            int pred = 0, actual = 0;
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                if (output[j] > output[pred]) pred = j;
                if (labels[i * OUTPUT_SIZE + j] > labels[i * OUTPUT_SIZE + actual]) actual = j;
            }
            if (pred == actual) correct++;

            checkCudaError(cudaFree(d_target), "cudaFree d_target in train");
            checkCudaError(cudaFree(d_input_half), "cudaFree d_input_half in train");
        }

        printf("Epoch %d - Loss: %.4f - Train Accuracy: %.2f%% - Time: %.3fs\n",
               epoch + 1, loss / numImages, (correct / (float)numImages) * 100, 
               stop_timer(epoch_start, epoch_stop));
    }

    // Destroy streams
    for (int s = 0; s < numStreams; s++) {
        checkCudaError(cudaStreamDestroy(streams[s]), "cudaStreamDestroy in train");
    }

    printf("Total training time: %.3fs\n", stop_timer(total_start, total_stop));

    if (VERBOSE) printf("Training completed\n");
}

void evaluate(NeuralNetwork* net, float* images, float* labels, int numImages) {
    if (VERBOSE) printf("\nStarting evaluation...\n");
    int correct = 0;
    
    for (int i = 0; i < numImages; i++) {
        float hidden[HIDDEN_SIZE], output[OUTPUT_SIZE];
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
    
    printf("Test Accuracy: %.2f%%\n", (correct / (float)numImages) * 100);
    if (VERBOSE) printf("Evaluation completed\n");
}

void freeNetwork(NeuralNetwork* net) {
    if (VERBOSE) printf("Freeing neural network...\n");
    
    // Free host memory
    cudaFreeHost(net->W1);
    cudaFreeHost(net->W2);
    cudaFreeHost(net->b1);
    cudaFreeHost(net->b2);

    // Free device memory
    checkCudaError(cudaFree(net->d_W1), "cudaFree d_W1");
    checkCudaError(cudaFree(net->d_W2), "cudaFree d_W2");
    checkCudaError(cudaFree(net->d_b1), "cudaFree d_b1");
    checkCudaError(cudaFree(net->d_b2), "cudaFree d_b2");
    checkCudaError(cudaFree(net->d_W1_half), "cudaFree d_W1_half");
    checkCudaError(cudaFree(net->d_W2_half), "cudaFree d_W2_half");
    checkCudaError(cudaFree(net->d_b1_half), "cudaFree d_b1_half");
    checkCudaError(cudaFree(net->d_b2_half), "cudaFree d_b2_half");
    checkCudaError(cudaFree(net->d_input), "cudaFree d_input");
    checkCudaError(cudaFree(net->d_hidden), "cudaFree d_hidden");
    checkCudaError(cudaFree(net->d_output), "cudaFree d_output");
    
    free(net);
    if (VERBOSE) printf("Neural network freed\n");
}