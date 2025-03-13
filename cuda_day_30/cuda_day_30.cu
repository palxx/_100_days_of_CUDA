#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_DIM 8
#define IN_TD BLOCK_DIM
#define OUT_TD (IN_TD - 2)

__global__ void stencil_kernel(float* in, float* out, unsigned int N) {
    int istart = blockIdx.z * OUT_TD;
    int j = blockIdx.y * OUT_TD + threadIdx.y;
    int k = blockIdx.x * OUT_TD + threadIdx.x;
    int C0 = 1;
    int C1 = 2;

    float iprev;
    __shared__ float iCurr_s[IN_TD][IN_TD];
    float innext;

    if(istart > 1 && istart < N && j >= 0 && j < N && k >= 0 && k < N){
      iprev = in[(istart-1)*N*N + j*N + k];
    }
    if(istart > 1 && istart < N && j >= 0 && j < N && k >= 0 && k < N){
      iCurr_s[threadIdx.y][threadIdx.x] = in[(istart)*N*N + j*N + k];
    }
    __syncthreads();

    for(int i = istart; i < istart + OUT_TD; ++i){
      if (i+1 >= 0 && i+1 < N && j >= 0 && j < N && k >= 0 && k < N) {
          innext = in[(i+1)*N*N + j*N + k];
        }
        __syncthreads();
        if (i >= 1 && i < N - 1 &&
        j >= 1 && j < N - 1 &&
        k >= 1 && k < N - 1) {
          if(threadIdx.x >= 1 && threadIdx.x < blockDim.x -1 && threadIdx.y >= 1 && threadIdx.y < blockDim.y -1){
            out[i * N * N + j * N + k] = C0 * iCurr_s[threadIdx.y][threadIdx.x] +
          C1 * ( iCurr_s[threadIdx.y][threadIdx.x-1] +
                 iCurr_s[threadIdx.y][threadIdx.x+1] +
                 iCurr_s[threadIdx.y-1][threadIdx.x] +
                 iCurr_s[threadIdx.y+1][threadIdx.x] +
                 iprev +
                 innext);
          }
        }
        __syncthreads();
        iprev = iCurr_s[threadIdx.y][threadIdx.x];
        iCurr_s[threadIdx.y][threadIdx.x] = innext;
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
    dim3 gridSize((N + OUT_TD - 1) / OUT_TD,
                  (N + OUT_TD - 1) / OUT_TD,
                  (N + OUT_TD - 1) / OUT_TD); 

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
