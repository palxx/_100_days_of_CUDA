#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#define THREADS_PER_BLOCK 1024
#define ELEM_PER_BLOCK 1024

// Device function for finding corank
__device__ unsigned int coRank(float* A, float* B, unsigned int m, unsigned int n, unsigned int k) {
    unsigned int low = (k > n) ? (k - n) : 0;
    unsigned int high = (k < m) ? k : m;

    while (true) {
        unsigned int i = (low + high) / 2;
        unsigned int j = k - i;

        if (i > 0 && j < n && A[i - 1] > B[j]) {
            high = i - 1;
        } else if (j > 0 && i < m && B[j - 1] > A[i]) {
            low = i + 1;
        } else {
            return i;
        }
    }
}

// Kernel for parallel merge
__global__ void merge_kernel(float* A, float* B, float* C, unsigned int m, unsigned int n) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int k = tid * ((m + n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

    unsigned int k_next = min(k + ((m + n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK), m + n);

    unsigned int i = coRank(A, B, m, n, k);
    unsigned int j = k - i;
    unsigned int i_next = coRank(A, B, m, n, k_next);
    unsigned int j_next = k_next - i_next;

    unsigned int t = 0;

    while (i < i_next && j < j_next) {
        if (A[i] <= B[j]) {
            C[k + t] = A[i];
            i++;
        } else {
            C[k + t] = B[j];
            j++;
        }
        t++;
    }

    while (i < i_next) {
        C[k + t] = A[i];
        i++;
        t++;
    }

    while (j < j_next) {
        C[k + t] = B[j];
        j++;
        t++;
    }
}

void merge_gpu(float* A, float* B, float* C, unsigned int m, unsigned int n) {
    float *A_d, *B_d, *C_d;

    // Allocate GPU memory
    cudaMalloc((void**)&A_d, m * sizeof(float));
    cudaMalloc((void**)&B_d, n * sizeof(float));
    cudaMalloc((void**)&C_d, (m + n) * sizeof(float));

    cudaMemcpy(A_d, A, m * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, n * sizeof(float), cudaMemcpyHostToDevice);

    unsigned int numBlocks = (m + n + ELEM_PER_BLOCK - 1) / ELEM_PER_BLOCK;

    merge_kernel<<<numBlocks, THREADS_PER_BLOCK>>>(A_d, B_d, C_d, m, n);

    cudaMemcpy(C, C_d, (m + n) * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main() {
    unsigned int m = 5, n = 5;
    float A[] = {1, 3, 5, 7, 9};
    float B[] = {2, 4, 6, 8, 10};
    float C[m + n];

    merge_gpu(A, B, C, m, n);

    std::cout << "Merged Array: ";
    for (unsigned int i = 0; i < m + n; i++) {
        std::cout << C[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
