#include <cuda.h>
#include <iostream>

__global__ void vectorAddKernel(float *A, float *B, float *C, int n){
  int i = blockIdx.x * blockDim.x + threadIdx.x; 
  if (i < n) C[i] = A[i] + B[i];
}

int main(){
  int n = 1024; 

  float *h_A = new float[n]; 
  float *h_B = new float[n];
  float *h_C = new float[n];

  int size = n * sizeof(float);
  float *d_A, *d_B, *d_C;

  for(int i = 0; i < n; i++){
    h_A[i] = 2 * i;
    h_B[i] = i;
  }

  cudaMalloc((void**)(&d_A), size);
  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  cudaMalloc((void**)(&d_B), size); 
  cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
  cudaMalloc((void**)(&d_C), size);

  vectorAddKernel<<<ceil(n/256), 256>>>(d_A, d_B, d_C, n);


  cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

  for(int i =0; i<n; i++){
    std::cout << "C[" << i << "] = " << h_C[i] << std::endl;
  }

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  delete[] h_A;
  delete[] h_B;
  delete[] h_C;

  return 0;
}
