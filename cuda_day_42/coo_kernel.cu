
#include "coo_matrix.h"

__global__ void cooKernel(const cooMatrix mat, const float *x, float *y){
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if(tid < mat.numNonZeros){
    unsigned int row = mat.rowId[tid];
    unsigned int col = mat.colId[tid];
    float val = mat.values[tid];

    atomicAdd(&y[row], val*x[col]);
  }
}
