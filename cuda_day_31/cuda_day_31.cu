#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_DIM 1024

__global__ void reduce_kernel(float* input, float* partialSums, unsigned int N){
    unsigned int segment = blockIdx.x * blockDim.x * 2;
    unsigned int i = segment + threadIdx.x * 2;

    __shared__ float sdata[BLOCK_DIM];

    float sum = 0.0f;
    if (i < N) sum += input[i];
    if (i + 1 < N) sum += input[i + 1];

    sdata[threadIdx.x] = sum;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        partialSums[blockIdx.x] = sdata[0];
    }
}

int main(){
    const unsigned int N = 1024 * 1024;
    float *in, *out;

    in = (float*)malloc(N * sizeof(float));
    out = (float*)malloc(sizeof(float));

    for (unsigned int i = 0; i < N; i++) {
        in[i] = 1.0f;
    }

    float *in_d, *partialSums_d;
    float elapsedTime;
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate device memory
    cudaEventRecord(start, 0);
    cudaMalloc((void**)&in_d, N * sizeof(float));
    cudaMemcpy(in_d, in, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Memory Allocation & Copy to GPU: %f ms\n", elapsedTime);

    // Kernel launch configuration
    const unsigned int numThreadsBlock = BLOCK_DIM;
    const unsigned int elePerBlock = 2 * numThreadsBlock;
    const unsigned int numBlocks = (N + elePerBlock - 1) / elePerBlock;

    float* partialSums = (float*)malloc(numBlocks * sizeof(float));
    cudaMalloc((void**)&partialSums_d, numBlocks * sizeof(float));

    // Launch kernel
    cudaEventRecord(start, 0);
    reduce_kernel<<<numBlocks, numThreadsBlock>>>(in_d, partialSums_d, N);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Kernel Execution Time: %f ms\n", elapsedTime);

    // Copy partial sums to host
    cudaMemcpy(partialSums, partialSums_d, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);

    // Final summation on CPU
    float sum = 0.0f;
    for (unsigned int i = 0; i < numBlocks; i++) {
        sum += partialSums[i];
    }

    printf("Total sum: %f\n", sum);

    // Free memory
    cudaFree(in_d);
    cudaFree(partialSums_d);
    free(in);
    free(out);
    free(partialSums);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
