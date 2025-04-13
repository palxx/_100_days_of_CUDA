#include "csrgraph.h"
#include <cuda_runtime.h>
#include <limits.h>
#include <stdio.h>

#define local_queue_size 2048

// BFS kernel: each thread processes one vertex from the previous frontier.
// New frontier vertices discovered are first stored in shared memory
// then merged into the global frontier.
__global__ void bfs_kernel(CSRgraph csrgraph, unsigned int* level, 
                             unsigned int* prevFrontier, unsigned int* currFrontier, 
                             unsigned int numPrevFrontier,
                             unsigned int* numCurrFrontier, // pointer to global counter
                             unsigned int currLevel) {

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ int currFrontier_s[local_queue_size];
    __shared__ int numCurrFrontier_s;  // counter in shared memory

    // Only one thread resets the block’s shared counter.
    if (threadIdx.x == 0) {
        numCurrFrontier_s = 0;
    }
    __syncthreads();

    if (idx < numPrevFrontier) {
        unsigned int vertex = prevFrontier[idx];
        // For all neighbors of 'vertex'
        for (unsigned int edge = csrgraph.scrPointers[vertex]; 
             edge < csrgraph.scrPointers[vertex + 1]; edge++) {
            unsigned int neighbor = csrgraph.dst[edge];
            // If the neighbor is not visited, mark it with the current level.
            if (atomicCAS(&level[neighbor], UINT_MAX, currLevel) == UINT_MAX) {
                // Use a shared-memory atomic increment to reserve a slot in the block–local queue.
                unsigned int currFrontierIdx_s = atomicAdd(&numCurrFrontier_s, 1);
                if (currFrontierIdx_s < local_queue_size) {
                    currFrontier_s[currFrontierIdx_s] = neighbor;
                } else {
                    // If the local queue is full, directly push into the global frontier.
                    unsigned int currFrontierIdx = atomicAdd(numCurrFrontier, 1);
                    currFrontier[currFrontierIdx] = neighbor;
                }
            }
        }
    }
    __syncthreads();

    // Now, one block-level thread (thread 0) reserves the next available index in the global frontier.
    __shared__ unsigned int currFronStartIdx;
    if (threadIdx.x == 0) {
        // Reserve a contiguous block in the global frontier array for the block's local results.
        currFronStartIdx = atomicAdd(numCurrFrontier, numCurrFrontier_s);
    }
    __syncthreads();

    // Copy the block's shared queue into the global queue (at the reserved offset).
    if (threadIdx.x < numCurrFrontier_s) {
        currFrontier[currFronStartIdx + threadIdx.x] = currFrontier_s[threadIdx.x];
    }
}

// Host function: performs BFS on the GPU using iterative kernel launches.
extern "C" void bfs_gpu(CSRgraph csrgraph, unsigned int srcVertex, unsigned int* level) {
    // Create a device copy of the CSR graph.
    CSRgraph csrgraph_d;
    csrgraph_d.numVertices = csrgraph.numVertices;
    csrgraph_d.numEdges = csrgraph.numEdges;
    // Allocate device memory for scrPointers (size: numVertices+1) and dst (size: numEdges).
    cudaMalloc((void**)&csrgraph_d.scrPointers, (csrgraph.numVertices + 1) * sizeof(unsigned int));
    cudaMalloc((void**)&csrgraph_d.dst, csrgraph.numEdges * sizeof(unsigned int));

    unsigned int* level_d;
    cudaMalloc((void**)&level_d, csrgraph.numVertices * sizeof(unsigned int));

    // Global counter for the new frontier elements.
    unsigned int* numCurrFrontier_d; 
    cudaMalloc((void**)&numCurrFrontier_d, sizeof(unsigned int));

    // Allocate device memory for two frontier buffers.
    unsigned int* prevFrontier_d;
    unsigned int* currFrontier_d;
    cudaMalloc((void**)&prevFrontier_d, csrgraph.numVertices * sizeof(unsigned int));
    cudaMalloc((void**)&currFrontier_d, csrgraph.numVertices * sizeof(unsigned int));

    // Copy the graph data from host to device.
    cudaMemcpy(csrgraph_d.scrPointers, csrgraph.scrPointers, (csrgraph.numVertices + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(csrgraph_d.dst, csrgraph.dst, csrgraph.numEdges * sizeof(unsigned int), cudaMemcpyHostToDevice);

    // Copy the "level" array to device.
    cudaMemcpy(level_d, level, csrgraph.numVertices * sizeof(unsigned int), cudaMemcpyHostToDevice);

    // Set the source vertex level to 0.
    level[srcVertex] = 0;
    cudaMemcpy(&level_d[srcVertex], &level[srcVertex], sizeof(unsigned int), cudaMemcpyHostToDevice);

    // Initialize the previous frontier with the source vertex.
    cudaMemcpy(prevFrontier_d, &srcVertex, sizeof(unsigned int), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    int numPrevFrontier = 1;
    int numThreadsPerBlock = 256;
    // We'll use 'numCurrFrontier_d' as our global frontier counter.
    unsigned int h_numCurrFrontier = 0;

    // Iteratively expand the BFS frontier.
    for (unsigned int currLevel = 1; numPrevFrontier > 0; ++currLevel) {
        // Reset the global counter for the new frontier.
        cudaMemset(numCurrFrontier_d, 0, sizeof(unsigned int));
        int numBlocks = (numPrevFrontier + numThreadsPerBlock - 1) / numThreadsPerBlock;
        bfs_kernel<<<numBlocks, numThreadsPerBlock>>>(csrgraph_d, level_d, prevFrontier_d, currFrontier_d, numPrevFrontier, numCurrFrontier_d, currLevel);
        cudaDeviceSynchronize();

        // Swap the frontier buffers.
        unsigned int* tmp = prevFrontier_d;
        prevFrontier_d = currFrontier_d;
        currFrontier_d = tmp;

        // Copy the number of new frontier vertices back to host.
        cudaMemcpy(&numPrevFrontier, numCurrFrontier_d, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    }

    cudaDeviceSynchronize();
    // Copy the resulting level array back to host.
    cudaMemcpy(level, level_d, csrgraph.numVertices * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    // Free device memory.
    cudaFree(csrgraph_d.scrPointers);
    cudaFree(csrgraph_d.dst);
    cudaFree(numCurrFrontier_d);
    cudaFree(level_d);
    cudaFree(prevFrontier_d);
    cudaFree(currFrontier_d);
}
