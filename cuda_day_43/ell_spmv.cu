// ell_spmv.cu
// Example of Sparse Matrix–Vector Multiplication using ELL (ELLPACK) format in CUDA.
//
// For the same 3x3 matrix, we use a fixed maximum of 2 nonzeros per row.
// The ELL representation stores two arrays (padded as needed):
// ell_indices (row-major order): [0, -1,   1, 2,   0, 2]
// ell_data:                      [10,  0,  20, 30,  40, 50]
// A padded index of -1 indicates no valid column.

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

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

// CUDA kernel for SpMV in ELL format.
__global__ void spmv_ell_kernel(int num_rows,
                                int max_nnz_per_row,
                                const int *ell_indices,
                                const float *ell_data,
                                const float *x,
                                float *y)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        float dot = 0.0f;
        for (int j = 0; j < max_nnz_per_row; j++) {
            int idx = row * max_nnz_per_row + j;
            int col = ell_indices[idx];
            // Only compute if the column index is valid.
            if (col >= 0) {
                dot += ell_data[idx] * x[col];
            }
        }
        y[row] = dot;
    }
}

int main() {
    // Define the ELL representation.
    // For our 3x3 matrix with max 2 nonzeros per row:
    // Row 0: one element at col 0 (10), pad second with 0.
    // Row 1: elements at col 1 (20) and col 2 (30).
    // Row 2: elements at col 0 (40) and col 2 (50).
    const int num_rows = 3;
    const int max_nnz_per_row = 2;

    int h_ell_indices[] = {
        0, -1,   // Row 0
        1,  2,   // Row 1
        0,  2    // Row 2
    };
    float h_ell_data[] = {
        10, 0,   // Row 0
        20, 30,  // Row 1
        40, 50   // Row 2
    };
    float h_x[] = {1, 2, 3};  // Input vector.
    float h_y[3] = {0};       // Result vector.

    // Allocate device memory.
    int *d_ell_indices;
    float *d_ell_data, *d_x, *d_y;
    CHECK_CUDA(cudaMalloc((void**)&d_ell_indices, num_rows * max_nnz_per_row * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_ell_data, num_rows * max_nnz_per_row * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_x, num_rows * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_y, num_rows * sizeof(float)));

    // Copy data to device.
    CHECK_CUDA(cudaMemcpy(d_ell_indices, h_ell_indices, num_rows * max_nnz_per_row * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_ell_data, h_ell_data, num_rows * max_nnz_per_row * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_x, h_x, num_rows * sizeof(float), cudaMemcpyHostToDevice));

    // Launch the kernel.
    int blockSize = 256;
    int gridSize = (num_rows + blockSize - 1) / blockSize;
    spmv_ell_kernel<<<gridSize, blockSize>>>(num_rows, max_nnz_per_row, d_ell_indices, d_ell_data, d_x, d_y);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy result back to host.
    CHECK_CUDA(cudaMemcpy(h_y, d_y, num_rows * sizeof(float), cudaMemcpyDeviceToHost));

    // Print the result.
    printf("Result vector y (ELL):\n");
    for (int i = 0; i < num_rows; i++) {
        printf("%f\n", h_y[i]);
    }

    // Free device memory.
    cudaFree(d_ell_indices);
    cudaFree(d_ell_data);
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}
