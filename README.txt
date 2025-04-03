# Neural Network Acceleration on GPUs

## Project Overview
This project focuses on accelerating a neural network implementation for the MNIST classification task using GPU programming with CUDA. We begin with a sequential CPU implementation (V1) and progressively optimize it to maximize performance on the GPU (V4). The key goal is to gain hands-on experience in parallel computing, high-performance computing (HPC), and CUDA optimizations.

## Repository Structure
```
├── src
│   ├── V1  # Baseline sequential implementation
│   ├── V2  # Naive GPU implementation
│   ├── V3  # Optimized GPU implementation with performance improvements
│   ├── V4  # Optimized GPU implementation utilizing tensor cores
├── data    # Contains the MNIST dataset
├── report  # Project report
├── slides  # Presentation slides
├── README.md  # Project documentation and instructions
```

## Prerequisites
- NVIDIA GPU with CUDA support
- CUDA Toolkit installed
- `nvcc` compiler available
- `make` utility installed

## Compilation and Execution
### Compilation
Navigate to the `src` directory and run:
```sh
make
```
This will compile the project and generate an executable located in `build/nn.exe`.

### Running the Program
To execute the program, run:
```sh
make run
```
This will execute the compiled neural network and move profiling data if available.

### Profiling Execution
To run the profiling version:
```sh
make prof-run
```
This generates profiling data for performance analysis.

### Cleaning Build Files
To remove all compiled files and reset the build directory:
```sh
make clean
```

## Code Structure
- **`main.cu`**: Entry point for the neural network execution.
- **`neural_net.cu`**: Core implementation of the neural network.
- **`utils.cu`**: Utility functions for matrix operations and timers.
- **`mnist.cu`**: MNIST dataset handling functions.
- **`nn.h`**: Header file defining neural network parameters.
- **`utils.h`**: Header file defining helper functions for matrix operations and timing.

## Optimization Strategy
Each version of the project applies different optimization techniques:

### **V1 (Baseline CPU Implementation)**
- Sequential execution on CPU.
- No parallelism or GPU acceleration.

### **V2 (Naive GPU Implementation)**
- Converts matrix operations to CUDA kernels.
- Parallel execution but lacks optimizations.

### **V3 (Optimized GPU Implementation)**
- Optimized kernel launch configuration.
- Improved occupancy and memory usage.
- Reduced communication overhead.
- Efficient memory hierarchy utilization.

### **V4 (Tensor Core Optimization)**
- Utilizes Tensor Cores for matrix multiplications.
- Further speedup through specialized CUDA libraries.

## Authors
- **Umer Farooq**
- **Muhammad Irtaza Khan**

