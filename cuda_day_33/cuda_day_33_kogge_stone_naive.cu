
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define BLOCK_DIM 1024

// ====================== Timer Utilities ======================

typedef struct {
    cudaEvent_t start, stop;
} Timer;

void startTime(Timer* t) {
    cudaEventCreate(&t->start);
    cudaEventCreate(&t->stop);
    cudaEventRecord(t->start, 0);
}

void stopTime(Timer* t) {
    cudaEventRecord(t->stop, 0);
    cudaEventSynchronize(t->stop);
}

void printElapsedTime(Timer t, const char* label) {
    float elapsed;
    cudaEventElapsedTime(&elapsed, t.start, t.stop);
    printf("%s: %.4f ms\n", label, elapsed);
    cudaEventDestroy(t.start);
    cudaEventDestroy(t.stop);
}

// ====================== Scan Kernel ======================

__global__ void scan_kernel(float* input, float* output, float* partialSums, unsigned int N) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    output[i] = input[i];
    __syncthreads();

    for (unsigned int stride = 1; stride <= BLOCK_DIM / 2; stride *= 2) {
        float v;
        if (threadIdx.x >= stride) {
            v = output[i - stride];
        }
        __syncthreads();
        if (threadIdx.x >= stride) {
            output[i] += v;
        }
        __syncthreads();
    }

    if (threadIdx.x == BLOCK_DIM - 1) {
        partialSums[blockIdx.x] = output[i];
    }
}

// ====================== Add Kernel ======================

__global__ void add_kernel(float* output, float* partialSums, unsigned int N) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (blockIdx.x > 0) {
        output[i] += partialSums[blockIdx.x - 1];
    }
}

// ====================== Recursive GPU Scan ======================

void scan_gpu_d(float* input_d, float* output_d, unsigned int N) {
    Timer timer;

    const unsigned int numThreadsPerBlock   = BLOCK_DIM;
    const unsigned int numElementsPerBlock  = numThreadsPerBlock;
    const unsigned int numBlocks            = (N + numElementsPerBlock - 1) / numElementsPerBlock;

    // Allocate partial sums
    startTime(&timer);
    float* partialSums_d;
    cudaMalloc((void**)&partialSums_d, numBlocks * sizeof(float));
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Partial sums allocation time");

    // Scan kernel
    startTime(&timer);
    scan_kernel<<<numBlocks, numThreadsPerBlock>>>(input_d, output_d, partialSums_d, N);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Kernel time");

    // Scan partial sums recursively if needed
    if (numBlocks > 1) {
        scan_gpu_d(partialSums_d, partialSums_d, numBlocks);
        add_kernel<<<numBlocks, numThreadsPerBlock>>>(output_d, partialSums_d, N);
        cudaDeviceSynchronize();
    }

    cudaFree(partialSums_d);
}

// ====================== Main ======================

int main() {
    const unsigned int N = 1 << 20;

    float* input  = (float*)malloc(N * sizeof(float));
    float* output = (float*)malloc(N * sizeof(float));

    for (unsigned int i = 0; i < N; ++i) {
        input[i] = 1.0f;
    }

    float *input_d, *output_d;
    cudaMalloc((void**)&input_d, N * sizeof(float));
    cudaMalloc((void**)&output_d, N * sizeof(float));
    cudaMemcpy(input_d, input, N * sizeof(float), cudaMemcpyHostToDevice);

    scan_gpu_d(input_d, output_d, N);
    cudaMemcpy(output, output_d, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Verification
    bool correct = true;
    for (unsigned int i = 0; i < N; ++i) {
        if (fabs(output[i] - (i + 1)) > 1e-5f) {
            printf("Mismatch at %u: got %f, expected %f\n", i, output[i], (float)(i + 1));
            correct = false;
            break;
        }
    }

    if (correct) {
        printf("Scan verified successfully!\n");
    }

    cudaFree(input_d);
    cudaFree(output_d);
    free(input);
    free(output);
    return 0;
}
