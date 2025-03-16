#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_DIM 1024
#define thread_coa 4 

__global__ void reduce_kernel(float* input, float* partialSums, unsigned int N){
    unsigned int segment = blockIdx.x * blockDim.x * 2 * thread_coa;
    unsigned int i = segment + threadIdx.x;

    __shared__ float input_s[BLOCK_DIM];
    input_s[threadIdx.x] = input[i] + input[i+BLOCK_DIM];
    float sum = 0.0f;
    for(unsigned int j = 0; j < thread_coa; ++j){
      sum += input[i+j*BLOCK_DIM];
    }
    input_s[threadIdx.x] = sum;
    __syncthreads;

    for(unsigned int stride = BLOCK_DIM/2; stride > 0; stride /= 2){
        if(threadIdx.x % stride == 0){
            input_s[i] += input_s[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if(threadIdx.x == 0){
        partialSums[blockIdx.x] = input[i];
    }
}

int main(){
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

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    cudaMalloc((void**)&in_d, N * sizeof(float));
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Allocate Time: %f ms\n", elapsedTime);

    cudaEventRecord(start, 0);
    cudaMemcpy(in_d, in, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Copy to GPU time: %f ms\n", elapsedTime);

    cudaEventRecord(start, 0);
    const unsigned int numThreadsBlock = BLOCK_DIM;
    const unsigned int elePerBlock = 2 * numThreadsBlock * thread_coa;
    const unsigned int numBlocks = (N + elePerBlock - 1) / elePerBlock;
    float* partialSums = (float*)malloc(numBlocks * sizeof(float));
    float *partialSums_d;
    cudaMalloc((void**)&partialSums_d, numBlocks * sizeof(float));
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Partial sums allocation time: %f ms\n", elapsedTime);

    cudaEventRecord(start, 0);
    reduce_kernel<<<numBlocks, numThreadsBlock>>>(in_d, partialSums_d, N);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Kernel Execution Time: %f ms\n", elapsedTime);

    cudaEventRecord(start, 0);
    cudaMemcpy(partialSums, partialSums_d, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Copy from GPU: %f ms\n", elapsedTime);

    cudaEventRecord(start, 0);
    float sum = 0.0f;
    for (unsigned int i = 0; i < numBlocks; i++) {
        sum += partialSums[i];
    }
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Print Partial Sums: %f ms\n", elapsedTime);

    // Cleanup.
    cudaEventRecord(start, 0);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(in_d);
    cudaFree(partialSums_d);
    free(partialSums);
    free(in);
    free(out);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("freeup end time: %f ms\n", elapsedTime);

    return 0;
}
