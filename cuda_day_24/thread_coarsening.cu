
#include <cuda.h>
#include <iostream>

#define TILE_WIDTH 32
#define COARSE_FACTOR 4

__global__ void matrixMulKernel(float* M, float* N, float* P, int width) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    // Identify the row and column of the P element to work on
    int row = by * TILE_WIDTH + ty;
    int colStart = bx * TILE_WIDTH * COARSE_FACTOR + tx;

    // Initialize Pvalue for all output elements
    float Pvalue[COARSE_FACTOR] = {0};

    // Loop over the M and N tiles required to compute P element
    for (int ph = 0; ph < width / TILE_WIDTH; ++ph) {

        // Load M tile into shared memory
        if (row < width && (ph * TILE_WIDTH + tx) < width)
            Mds[ty][tx] = M[row * width + ph * TILE_WIDTH + tx];
        else
            Mds[ty][tx] = 0.0f;

        for (int c = 0; c < COARSE_FACTOR; ++c) {
            int col = colStart + c * TILE_WIDTH;
            if (col < width && (ph * TILE_WIDTH + ty) < width)
                Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * width + col];
            else
                Nds[ty][tx] = 0.0f;
        }
        __syncthreads();  // Synchronize after loading shared memory

        // Compute for each tile
        for (int k = 0; k < TILE_WIDTH; ++k) {
            for (int c = 0; c < COARSE_FACTOR; ++c) {
                Pvalue[c] += Mds[ty][k] * Nds[k][tx];
            }
        }
        __syncthreads();
    }

    // Store results in the output matrix P
    for (int c = 0; c < COARSE_FACTOR; ++c) {
        int col = colStart + c * TILE_WIDTH;
        if (row < width && col < width)
            P[row * width + col] = Pvalue[c];
    }
}

int main() {
    int width = 1024;
    int size = width * width * sizeof(float);

    float *M, *N, *P;
    float *d_M, *d_N, *d_P;

    // Allocate host memory
    M = new float[width * width];
    N = new float[width * width];
    P = new float[width * width];

    // Initialize matrices
    for (int i = 0; i < width * width; i++) {
        M[i] = 1.0f;  // Filling with 1s
        N[i] = 2.0f;  // Filling with 2s
    }

    // Allocate device memory
    cudaMalloc((void**)&d_M, size);
    cudaMalloc((void**)&d_N, size);
    cudaMalloc((void**)&d_P, size);

    // Copy data to device
    cudaMemcpy(d_M, M, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, N, size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 dimGrid((width + TILE_WIDTH * COARSE_FACTOR - 1) / (TILE_WIDTH * COARSE_FACTOR), 
                 (width + TILE_WIDTH - 1) / TILE_WIDTH);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);

    // Launch the kernel
    matrixMulKernel<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, width);
    cudaDeviceSynchronize();

    // Copy the result back to host
    cudaMemcpy(P, d_P, size, cudaMemcpyDeviceToHost);

    // Print some of the results (for debugging)
    std::cout << "Result Matrix (First 10 Values):" << std::endl;
    for (int i = 0; i < 10; i++) {
        std::cout << P[i] << " ";
    }
    std::cout << std::endl;

    // Free memory
    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);
    delete[] M;
    delete[] N;
    delete[] P;

    return 0;
}
