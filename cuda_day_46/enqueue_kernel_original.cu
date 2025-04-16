#include <cuda_runtime.h>
#include <stdio.h>

__device__ bool cond(unsigned int val) {
    return (val % 2 == 0);
}

__global__ void enqueue_kernel(unsigned int* input, unsigned int* queue, unsigned int N, unsigned int* queueSize){
unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
if(i< N){
  unsigned int val = input[i];
  if(cond(val)){
    unsigned int j = atomicAdd(queueSize, 1);
    queue[j] = val;
  }
}
}

extern "C" unsigned int enqueue_gpu(unsigned int* input, unsigned int* queue, unsigned int N){
unsigned int* input_d;
unsigned int* queue_d;
unsigned int *queueSize_d;

cudaMalloc((void**)&input_d, N*sizeof(unsigned int));
cudaMalloc((void**)&queue_d, N*sizeof(unsigned int));
cudaMalloc((void**)&queueSize_d, sizeof(unsigned int));
cudaDeviceSynchronize();

cudaMemcpy(input_d, input, N*sizeof(unsigned int), cudaMemcpyHostToDevice);
cudaMemcpy(queue_d, queue, N*sizeof(unsigned int), cudaMemcpyHostToDevice);
cudaMemset(queueSize_d, 0, sizeof(unsigned int));

int numThreadsPerBlock = 256;
int numBlocks = (N + numThreadsPerBlock -1)/numThreadsPerBlock;
enqueue_kernel<<<numBlocks, numThreadsPerBlock>>>(input_d, queue_d, N, queueSize_d);

unsigned int queueSize;
cudaMemcpy(&queueSize, queueSize_d, sizeof(unsigned int), cudaMemcpyDeviceToHost);

cudaMemcpy(queue, queue_d, queueSize * sizeof(unsigned int), cudaMemcpyHostToDevice);

cudaFree(input_d);
cudaFree(queue_d);
cudaFree(queueSize_d);

return queueSize;
}
