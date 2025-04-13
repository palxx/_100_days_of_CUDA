#include "csrgraph.h"
#include <cuda_runtime.h>
#include <limits.h>
#include <stdio.h>

__global__ void bfs_kernel(CSRgraph csrgraph, unsigned int* level, unsigned int* prevFrontier, unsigned int* currFrontier, unsigned int numPrevFrontier,
unsigned int numCurrFrontier,unsigned int currLevel){
  unsigned int idx = blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<numPrevFrontier){
    unsigned int vertex = prevFrontier[idx];
    for(unsigned int edge = csrgraph.scrPointers[vertex]; edge < csrgraph.scrPointers[vertex+1; edge++]){
      unsigned int neighbor = csrgraph.dst[edge];
      if(atomicCAS(&level[neighbor], UINT_MAX, currLevel)==UINT_MAX){
        unsigned int currFrontierIdx = atomicAdd(numCurrFrontier, 1);
        currFrontier[currFrontierIdx] = neighbor;
      }
    }
  }
}

void bfs_gpu(CSRgraph csrgraph, unsigned int srcVertex, unsigned int* level){
  CSRgraph csrgraph_d;
  csrgraph_d.numVertices = csrgraph.numVertices;
  csrgraph_d.numEdges = csrgraph.numEdges;
  cudaMalloc((void**)&csrgraph_d.scrPointers, (csrgraph_d.numEdges+1)*sizeof(unsigned int));
  cudaMalloc((void**)&csrgraph_d.dst,csrgraph_d.numEdges*sizeof(unsigned int));
  unsigned int* level_d;
  cudaMalloc((void**)&level_d, csrgraph.numVertices * sizeof(unsigned int));
  unsigned int* buffer_1;
  cudaMalloc((void**)&buffer_1, csrgraph.numVertices * sizeof(unsigned int));
  unsigned int* buffer_2;
  cudaMalloc((void**)&buffer_2, csrgraph_d.numVertices * sizeof(unsigned int));
  unsigned int* numCurrFrontier_d; 
  cudaMalloc((void**)&numCurrFrontier_d, sizeof(unsigned int));
  unsigned int* prevfrontier_d;
  unsigned int* currFrontier_d;
  cudaDeviceSynchronize();

  cudaMemcpy(csrgraph_d.scrPointers, csrgraph.scrPointers, csrgraph_d.numVertices*sizeof(unsigned int), cudaMemcpyHostToDevice);
  cudaMemcpy(csrgraph_d.dst, csrgraph.dst, csrgraph_d.numVertices*sizeof(unsigned int), cudaMemcpyHostToDevice);
  level_d[srcVertex] = 0;
  cudaMemcpy(level_d, level, csrgraph_d.numVertices*sizeof(unsigned int), cudaMemcpyHostToDevice);
  cudaMemcpy(prevfrontier_d, &srcVertex, sizeof(unsigned int), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();

  int numPrevFrontier = 1;
  int numThreadsPerBlock = 256;
  for(unsigned int currLevel =1; numPrevFrontier>0; ++currLevel){
    cudaMemset(numCurrFrontier_d, 0, sizeof(unsigned int));
    int numBlocks = (numPrevFrontier+numThreadsPerBlock-1/numThreadsPerBlock);
    bfs_kernel<<<numBlocks, numThreadsPerBlock>>>(csrgraph_d, level_d, prevfrontier_d, currFrontier_d, numPrevFrontier, numCurrFrontier_d);
    unsigned int* tmp = prevfrontier_d;
    prevfrontier_d = currFrontier_d;
    currFrontier_d = tmp;
    cudaMemcpy(&numPrevFrontier, numCurrFrontier_d, sizeof(unsigned int), cudaMemcpyDeviceToHost);
  }

  cudaDeviceSynchronize;
  cudaMemcpy(level, level_d, csrgraph_d.numVertices*sizeof(unsigned int), cudamemcpyDeviceToHost);


    // Free device memory.
    cudaFree(csrgraph_d.scrPointers);
    cudaFree(csrgraph_d.dst);
    cudaFree(numCurrFrontier_d);
    cudaFree(level_d);
    cudaFree(prevFrontier_d);
    cudaFree(currFrontier_d);
}
