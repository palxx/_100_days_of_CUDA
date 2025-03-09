
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#define OUT_TILE_DIM 32
#define MASK_DIM 3  // Example mask size (3x3)
#define MASK_RADIUS (MASK_DIM / 2)

__constant__ float mask_c[MASK_DIM][MASK_DIM];

__global__ void convolution_kernel(float *input, float* output, unsigned int width, unsigned int height){
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;

    if (outRow < height && outCol < width) {
        float sum = 0.0f;
        for(int maskRow = 0; maskRow < MASK_DIM; ++maskRow){
            for(int maskCol = 0; maskCol < MASK_DIM; ++maskCol){
                int inRow = outRow - MASK_RADIUS + maskRow;
                int inCol = outCol - MASK_RADIUS + maskCol;
                if(inRow >= 0 && inRow < height && inCol >= 0  && inCol < width){
                    sum += mask_c[maskRow][maskCol] * input[inRow * width + inCol];
                }
            }
        }
        output[outRow * width + outCol] = sum;
    }
}

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    const int width = 1024;
    const int height = 1024;
    const int size = width * height * sizeof(float);
    
    float *h_input = new float[width * height];
    float *h_output = new float[width * height];
    float h_mask[MASK_DIM][MASK_DIM] = {
        {0, -1, 0},
        {-1, 4, -1},
        {0, -1, 0}
    };

    // Initialize input data
    for (int i = 0; i < width * height; i++) {
        h_input[i] = static_cast<float>(rand() % 256); // Random pixel values
    }

    // Allocate device memory
    float *d_input, *d_output;
    checkCudaError(cudaMalloc(&d_input, size), "Failed to allocate input memory");
    checkCudaError(cudaMalloc(&d_output, size), "Failed to allocate output memory");

    // Copy input data to device
    checkCudaError(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice), "Failed to copy input data");

    // Copy convolution mask to constant memory
    checkCudaError(cudaMemcpyToSymbol(mask_c, h_mask, MASK_DIM * MASK_DIM * sizeof(float)), "Failed to copy mask to constant memory");

    // Configure grid and block sizes
    dim3 blockDim(OUT_TILE_DIM, OUT_TILE_DIM);
    dim3 gridDim((width + OUT_TILE_DIM - 1) / OUT_TILE_DIM, (height + OUT_TILE_DIM - 1) / OUT_TILE_DIM);

    // Setup CUDA timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start timer
    cudaEventRecord(start);

    // Launch kernel
    convolution_kernel<<<gridDim, blockDim>>>(d_input, d_output, width, height);
    checkCudaError(cudaGetLastError(), "Kernel launch failed");

    // Stop timer
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy result back to host
    checkCudaError(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost), "Failed to copy output data");

    // Print execution time
    std::cout << "Kernel Execution Time: " << milliseconds << " ms" << std::endl;

    // Print first 10 output values for verification
    std::cout << "Sample Output Values: ";
    for (int i = 0; i < 10; i++) {
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;

    // Free memory
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_input;
    delete[] h_output;

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
