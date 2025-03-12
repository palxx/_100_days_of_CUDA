#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_DIM 8
#define IN_TD BLOCK_DIM
#define OUT_TD (IN_TD - 2)

__global__ void stencil_kernel(float* in, float* out, unsigned int N) {
    // Compute global indices using threadIdx within each block.
    unsigned int i = blockIdx.z * blockDim.z + threadIdx.z;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
    int C0 = 1;
    int C1 = 2;
    
    // Process only interior points.
    if (i >= 1 && i < N - 1 &&
        j >= 1 && j < N - 1 &&
        k >= 1 && k < N - 1) {
        out[i * N * N + j * N + k] = C0 * in[i * N * N + j * N + k] +
          C1 * ( in[i * N * N + j * N + (k + 1)] +
                 in[i * N * N + j * N + (k - 1)] +
                 in[i * N * N + (j - 1) * N + k] +
                 in[i * N * N + (j + 1) * N + k] +
                 in[(i - 1) * N * N + j * N + k] +
                 in[(i + 1) * N * N + j * N + k] );
    }
}

int main() {
    const unsigned int N = 128;
    float *in, *out;
    
    // Allocate host memory.
    in = (float*)malloc(N * N * N * sizeof(float));
    out = (float*)malloc(N * N * N * sizeof(float));
    if (!in || !out) {
        fprintf(stderr, "Host memory allocation failed\n");
        return -1;
    }
    
    // Initialize the input array.
    for (unsigned int i = 0; i < N * N * N; i++) {
        in[i] = 1.0f; // Set to any appropriate value.
    }
    
    float *in_d, *out_d;
    cudaEvent_t start, stop;
    float elapsedTime;
    
    // Create CUDA events.
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Allocate device memory and time it.
    cudaEventRecord(start, 0);
    cudaMalloc((void**)&in_d, N * N * N * sizeof(float));
    cudaMalloc((void**)&out_d, N * N * N * sizeof(float));
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Allocate Time: %f ms\n", elapsedTime);
    
    // Copy data to GPU and time it.
    cudaEventRecord(start, 0);
    cudaMemcpy(in_d, in, N * N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Copy to GPU time: %f ms\n", elapsedTime);
    
    // Launch the kernel and time it.
    dim3 threadPerBlock(BLOCK_DIM, BLOCK_DIM, BLOCK_DIM);
    dim3 gridSize((N + BLOCK_DIM - 1) / BLOCK_DIM,
                  (N + BLOCK_DIM - 1) / BLOCK_DIM,
                  (N + BLOCK_DIM - 1) / BLOCK_DIM);
    
    cudaEventRecord(start, 0);
    stencil_kernel<<<gridSize, threadPerBlock>>>(in_d, out_d, N);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Kernel Time: %f ms\n", elapsedTime);
    
    // Copy data from GPU and time it.
    cudaEventRecord(start, 0);
    cudaMemcpy(out, out_d, N * N * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Copy from GPU time: %f ms\n", elapsedTime);
    
    // Cleanup.
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(in_d);
    cudaFree(out_d);
    free(in);
    free(out);
    
    return 0;
}
