// csr_spmv.cu
// Example of Sparse Matrix–Vector Multiplication using CSR (Compressed Sparse Row) format in CUDA.

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Macro for CUDA error checking.
#define CHECK_CUDA(call)                                              \
    {                                                                 \
        cudaError_t err = call;                                       \
        if (err != cudaSuccess) {                                     \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",          \
                    __FUNCTION__, __FILE__, __LINE__,                 \
                    cudaGetErrorString(err));                         \
            exit(err);                                                \
        }                                                             \
    }

// CUDA kernel for SpMV in CSR format.
__global__ void spmv_csr_kernel(int num_rows,
                                const int *row_ptr,
                                const int *col_ind,
                                const float *values,
                                const float *x,
                                float *y)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        float dot = 0.0f;
        for (int j = row_ptr[row]; j < row_ptr[row + 1]; j++) {
            dot += values[j] * x[col_ind[j]];
        }
        y[row] = dot;
    }
}

int main() {
    // Example matrix (3x3):
    // [ 10   0   0 ]
    // [  0  20  30 ]
    // [ 40   0  50 ]
    // CSR representation:
    // row_ptr: [0, 1, 3, 5]
    // col_ind: [0, 1, 2, 0, 2]
    // values:  [10, 20, 30, 40, 50]
    int h_row_ptr[] = {0, 1, 3, 5};
    int h_col_ind[] = {0, 1, 2, 0, 2};
    float h_values[] = {10, 20, 30, 40, 50};
    float h_x[] = {1, 2, 3};   // Input vector.
    float h_y[3] = {0};        // Result vector.

    int num_rows = 3;
    int nnz = 5; // Number of nonzeros.

    // Allocate device memory.
    int *d_row_ptr, *d_col_ind;
    float *d_values, *d_x, *d_y;
    CHECK_CUDA(cudaMalloc((void**)&d_row_ptr, (num_rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_col_ind, nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_values, nnz * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_x, num_rows * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_y, num_rows * sizeof(float)));

    // Copy data from host to device.
    CHECK_CUDA(cudaMemcpy(d_row_ptr, h_row_ptr, (num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_col_ind, h_col_ind, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_values, h_values, nnz * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_x, h_x, num_rows * sizeof(float), cudaMemcpyHostToDevice));

    // Launch the kernel.
    int blockSize = 256;
    int gridSize = (num_rows + blockSize - 1) / blockSize;
    spmv_csr_kernel<<<gridSize, blockSize>>>(num_rows, d_row_ptr, d_col_ind, d_values, d_x, d_y);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy the result back to host.
    CHECK_CUDA(cudaMemcpy(h_y, d_y, num_rows * sizeof(float), cudaMemcpyDeviceToHost));

    // Print the result.
    printf("Result vector y (CSR):\n");
    for (int i = 0; i < num_rows; i++) {
        printf("%f\n", h_y[i]);
    }

    // Free device memory.
    cudaFree(d_row_ptr);
    cudaFree(d_col_ind);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}
