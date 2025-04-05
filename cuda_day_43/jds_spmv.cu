// jds_spmv.cu
// Example of Sparse Matrix–Vector Multiplication using JDS (Jagged Diagonal Storage) format in CUDA.
//
// For illustration, we use the same matrix as above but reorder the rows by nonzero counts.
// Original matrix:
// [ 10   0   0 ]
// [  0  20  30 ]
// [ 40   0  50 ]
//
// After reordering by nonzeros (descending), permutation becomes: [1, 2, 0].
// JDS representation arrays are built as follows:
// jds_ptr:  {0, 3, 5}    --> Start indices for each diagonal.
// jds_indices: {1, 0, 0, 2, 2}
// jds_data:    {20, 40, 10, 30, 50}
// diag_len:    {3, 2}      --> Diagonal 0 has 3 entries, diagonal 1 has 2 entries.
// jds_perm:    {1, 2, 0}   --> Mapping from JDS order back to original rows.

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

// CUDA kernel for SpMV in JDS format.
__global__ void spmv_jds_kernel(int num_rows,
                                int num_diagonals,
                                const int *jds_ptr,
                                const int *jds_indices,
                                const float *jds_data,
                                const int *diag_len,
                                const int *jds_perm,
                                const float *x,
                                float *y)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_rows) {
        float dot = 0.0f;
        // Loop over each diagonal.
        for (int d = 0; d < num_diagonals; d++) {
            // Only process if the current row has an element in this diagonal.
            if (tid < diag_len[d]) {
                int idx = jds_ptr[d] + tid;
                dot += jds_data[idx] * x[jds_indices[idx]];
            }
        }
        // Write the result back in the original row order.
        y[jds_perm[tid]] = dot;
    }
}

int main() {
    // Set up the JDS representation for the same 3x3 matrix.
    // JDS arrays as described above:
    const int num_rows = 3;
    const int num_diagonals = 2;
    const int nnz = 5;

    // Host arrays.
    int h_jds_ptr[] = {0, 3, 5};
    int h_jds_indices[] = {1, 0, 0, 2, 2};
    float h_jds_data[] = {20, 40, 10, 30, 50};
    int h_diag_len[] = {3, 2};
    int h_jds_perm[] = {1, 2, 0};
    float h_x[] = {1, 2, 3};  // Input vector.
    float h_y[3] = {0};       // Result vector.

    // Allocate device memory.
    int *d_jds_ptr, *d_jds_indices, *d_diag_len, *d_jds_perm;
    float *d_jds_data, *d_x, *d_y;
    CHECK_CUDA(cudaMalloc((void**)&d_jds_ptr, (num_diagonals + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_jds_indices, nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_jds_data, nnz * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_diag_len, num_diagonals * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_jds_perm, num_rows * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_x, num_rows * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_y, num_rows * sizeof(float)));

    // Copy host data to device.
    CHECK_CUDA(cudaMemcpy(d_jds_ptr, h_jds_ptr, (num_diagonals + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_jds_indices, h_jds_indices, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_jds_data, h_jds_data, nnz * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_diag_len, h_diag_len, num_diagonals * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_jds_perm, h_jds_perm, num_rows * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_x, h_x, num_rows * sizeof(float), cudaMemcpyHostToDevice));

    // Launch the kernel.
    int blockSize = 256;
    int gridSize = (num_rows + blockSize - 1) / blockSize;
    spmv_jds_kernel<<<gridSize, blockSize>>>(num_rows, num_diagonals, d_jds_ptr,
                                              d_jds_indices, d_jds_data, d_diag_len,
                                              d_jds_perm, d_x, d_y);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy the result back to host.
    CHECK_CUDA(cudaMemcpy(h_y, d_y, num_rows * sizeof(float), cudaMemcpyDeviceToHost));

    // Print the result.
    printf("Result vector y (JDS):\n");
    for (int i = 0; i < num_rows; i++) {
        printf("%f\n", h_y[i]);
    }

    // Free device memory.
    cudaFree(d_jds_ptr);
    cudaFree(d_jds_indices);
    cudaFree(d_jds_data);
    cudaFree(d_diag_len);
    cudaFree(d_jds_perm);
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}
