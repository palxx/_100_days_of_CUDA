#include <cuda_runtime.h>
#include <stdio.h>
#define WRAP_SIZE 32

__device__ bool cond(unsigned int val) {
    return (val % 2 == 0);
}

__global__ void enqueue_kernel(unsigned int* input, unsigned int* queue, unsigned int N, unsigned int* queueSize){
unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
if(i< N){
  unsigned int val = input[i];
  if(cond(val)){
    unsigned int activeThreads = __activemask();
    unsigned int leader = __ffs(activeThreads)-1;
    unsigned int numActive = __popc(activeThreads);
    unsigned int j;
    if(threadIdx.x % WRAP_SIZE ==leader){
      j = atomicAdd(queueSize, numActive);
    }
    j = __shfl_sync(activeThreads, j, leader);
    unsigned int previousThreads = (1 << (threadIdx.x % WRAP_SIZE)) -1;
    unsigned int activePreviousThreads = activeThreads & previousThreads;
    unsigned int offset = __popc(activePreviousThreads);
    queue[j+offset] = val;
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
