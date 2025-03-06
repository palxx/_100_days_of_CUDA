#CUDA Matrix Multiplication with Tiling and Coarse Factor Optimization#

Coarse Factor Optimization: Each thread computes multiple outputs to improve memory efficiency.

#Performance Optimization#

1. Reduces Global Memory Accesses by loading tiles into shared memory.

2. Each Thread Computes COARSE_FACTOR Results, reducing redundant loads.

3. Blocks Synchronize using __syncthreads() to avoid race conditions.

4. For a particular tile A multiple consecutive tiles of B are caluculated at the same time.

5. Having more coarsening factor can hurt your performance due to more serialization. It is important to choose a good and optimal coasening factor.

