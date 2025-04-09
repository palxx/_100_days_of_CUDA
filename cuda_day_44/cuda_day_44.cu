#include <cuda_runtime.h>
#include "csrgraph.h"
#include <vector>
#include <limits>
#include <iostream>

void bfs_gpu_normal(CSRgraph csrGraph, unsigned int srcVertex, unsigned int* level);

int main(){
   int numVertices = 4;
   int numEdges = 5;

  CSRgraph csrGraph;
  csrGraph.numVertices = numVertices;
  csrGraph.numEdges = numEdges;
  std::vector<unsigned int>scrPointers = {0,2,3,5};
  std::vector<unsigned int>dst={2,3,3, 0, 1};

  unsigned int* level = new unsigned int[numVertices];
  for(unsigned int i = 0; i< numVertices; i++){
    level[i] = UINT_MAX;
  }

  csrGraph.scrPointers = scrPointers.data();
  csrGraph.dst = dst.data();

  int srcVertex = 0;
  bfs_gpu_normal(csrGraph, srcVertex, level);

  std::cout << "Vertex : Level" << std::endl;
    for (int i = 0; i < numVertices; i++) {
        std::cout << i << " : " << level[i] << std::endl;
    }
    
  delete[] level;
  return 0;
}

