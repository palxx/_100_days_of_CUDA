#include <iostream>
#include <vector>
#include <limits.h>
#include "csrgraph.h"

// Declaration for the GPU BFS function defined in bfs_kernel_gpu_efficient.cu.
extern "C" void bfs_gpu(CSRgraph csrgraph, unsigned int srcVertex, unsigned int* level);

int main(){
    // Sample graph:
    // Let's define a graph with 4 vertices and 5 edges.
    // For example, consider the following CSR representation:
    //  numVertices = 4
    //  scrPointers = [0, 2, 3, 5, 5]
    //  dst = [1, 3, 2, 0, 1]
    // Interpretation:
    //  - Vertex 0 has neighbors: 1, 3.
    //  - Vertex 1 has neighbor: 2.
    //  - Vertex 2 has neighbors: 0, 1.
    //  - Vertex 3 has no neighbors.
    
    int numVertices = 4;
    int numEdges = 5;
    CSRgraph csrGraph;
    csrGraph.numVertices = numVertices;
    csrGraph.numEdges = numEdges;
    
    std::vector<unsigned int> scrPointers = {0, 2, 3, 5, 5};
    std::vector<unsigned int> dst = {1, 3, 2, 0, 1};
    
    csrGraph.scrPointers = scrPointers.data();
    csrGraph.dst = dst.data();
    
    // Initialize the level array: All vertices start as unvisited, marked with UINT_MAX.
    std::vector<unsigned int> level(numVertices, UINT_MAX);
    // Set the source vertex (say vertex 0) to level 0.
    unsigned int srcVertex = 0;
    level[srcVertex] = 0;
    
    // Run BFS on the GPU.
    bfs_gpu(csrGraph, srcVertex, level.data());
    
    // Print out the resulting levels.
    std::cout << "Vertex : Level" << std::endl;
    for (int i = 0; i < numVertices; i++) {
        std::cout << i << " : " << level[i] << std::endl;
    }
    
    return 0;
}
