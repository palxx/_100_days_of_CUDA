#include "csrgraph.h"
#include <cuda_runtime.h>
#include <limits.h>
#include <stdio.h>

__global__ void gpu_kernel(CSRgraph csrGraph, unsigned int* level, unsigned int* newVertexVisited, unsigned int currLevel){
  unsigned int vertex = blockIdx.x * blockDim.x + threadIdx.x;
  if(vertex < csrGraph.numVertices){
    if(level[vertex] == currLevel -1 ){
      for(unsigned int edge = csrGraph.scrPointers[vertex]; edge < csrGraph.scrPointers[vertex +1]; edge++){
        unsigned int neighbor = csrGraph.dst[edge];
        if(level[neighbor] == UINT_MAX){
          level[neighbor] = currLevel;
          *newVertexVisited = 1;
        }
      }
    }
  }
}

void bfs_gpu_normal(CSRgraph csrGraph, unsigned int srcVertex, unsigned int* level){

  CSRgraph csrGraph_d;
  csrGraph_d.numVertices = csrGraph.numVertices;
  csrGraph_d.numEdges = csrGraph.numEdges;
  cudaMalloc((void**)&csrGraph_d.scrPointers, (csrGraph_d.numVertices+1)*sizeof(unsigned int));
  cudaMalloc((void**)&csrGraph_d.dst, csrGraph_d.numEdges * sizeof(unsigned int));
  unsigned int* level_d;
  cudaMalloc((void**)&level_d, csrGraph_d.numVertices*sizeof(unsigned int));
  unsigned int* newVertexVisited_d;
  cudaMalloc((void**)&newVertexVisited_d, sizeof(unsigned int));

  cudaMemcpy(csrGraph_d.scrPointers, csrGraph.scrPointers, (csrGraph.numVertices + 1)*sizeof(unsigned int), cudaMemcpyHostToDevice);
  cudaMemcpy(csrGraph_d.dst, csrGraph.dst, csrGraph_d.numEdges * sizeof(unsigned int), cudaMemcpyHostToDevice);
  level[srcVertex] = 0;
  cudaMemcpy(level_d, level, csrGraph_d.numVertices * sizeof(unsigned int), cudaMemcpyHostToDevice);

  unsigned int numThreadsperblock = 256;
  unsigned int numBlocks = (csrGraph_d.numVertices + numThreadsperblock -1)/numThreadsperblock;
  unsigned newVertexVisited = 1;

  for(unsigned int currLevel = 1; newVertexVisited; currLevel++){
  newVertexVisited = 0;
  cudaMemcpy(newVertexVisited_d, &newVertexVisited, sizeof(unsigned int), cudaMemcpyHostToDevice);
  gpu_kernel<<<numBlocks, numThreadsperblock>>>(csrGraph_d, level_d, newVertexVisited_d, currLevel);
  cudaMemcpy(&newVertexVisited, newVertexVisited_d, sizeof(unsigned int), cudaMemcpyDeviceToHost);
  }

  cudaMemcpy(level, level_d, csrGraph_d.numVertices* sizeof(unsigned int), cudaMemcpyDeviceToHost);

  cudaFree(csrGraph_d.scrPointers);
  cudaFree(csrGraph_d.dst);
  cudaFree(newVertexVisited_d);
  cudaFree(level_d);
}
