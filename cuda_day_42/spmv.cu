
#include <cstdio>
#include <cuda_runtime.h>
#include "coo_matrix.h"
#define THREAD_PER_BLOCK 1024

// Forward declaration
__global__ void cooKernel(const cooMatrix mat, const float *x, float *y);

void coo_kernel(const cooMatrix &h_cooMat, const float *h_invec, float *outvec) {
  cooMatrix d_cooMat;
  d_cooMat.numRows = h_cooMat.numRows;
  d_cooMat.numCols = h_cooMat.numCols;
  d_cooMat.numNonZeros = h_cooMat.numNonZeros;

  float *d_inVec = nullptr;
  float *d_outVector = nullptr;

  // Allocate device memory using correct sizes
  cudaMalloc((void**)&d_cooMat.rowId, d_cooMat.numNonZeros * sizeof(unsigned int));
  cudaMalloc((void**)&d_cooMat.colId, d_cooMat.numNonZeros * sizeof(unsigned int));
  cudaMalloc((void**)&d_cooMat.values, d_cooMat.numNonZeros * sizeof(float));
  cudaMalloc((void**)&d_inVec, d_cooMat.numCols * sizeof(float));
  cudaMalloc((void**)&d_outVector, d_cooMat.numRows * sizeof(float));

  // Copy host data to device memory with the correct source and sizes
  cudaMemcpy(d_cooMat.rowId, h_cooMat.rowId, d_cooMat.numNonZeros * sizeof(unsigned int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_cooMat.colId, h_cooMat.colId, d_cooMat.numNonZeros * sizeof(unsigned int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_cooMat.values, h_cooMat.values, d_cooMat.numNonZeros * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_inVec, h_invec, d_cooMat.numCols * sizeof(float), cudaMemcpyHostToDevice);

  // Initialize output vector on device to zero
  cudaMemset(d_outVector, 0, d_cooMat.numRows * sizeof(float));

  dim3 gridsize((d_cooMat.numNonZeros + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK);

  // Launch the kernel
  cooKernel<<<gridsize, THREAD_PER_BLOCK>>>(d_cooMat, d_inVec, d_outVector);

  // Copy result back to host
  cudaMemcpy(outvec, d_outVector, d_cooMat.numRows * sizeof(float), cudaMemcpyDeviceToHost);

  // Free allocated device memory
  cudaFree(d_cooMat.rowId);
  cudaFree(d_cooMat.colId);
  cudaFree(d_cooMat.values);
  cudaFree(d_inVec);
  cudaFree(d_outVector);
}
