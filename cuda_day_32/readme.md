# 🔢 CUDA Reduction Kernel – Sum of Array Elements on GPU

## 📋 Summary

This CUDA program demonstrates how to perform **parallel reduction**, i.e., summing all elements of a large array using **GPU kernels**, which is much faster than doing it on the CPU for large datasets.

The goal is to **reduce a large array into a single sum** using **multiple threads in parallel**, making the most out of **CUDA's thread-level parallelism**.

---

## 🔧 What This Code Does

1. **Initial Setup**:
   - An array is initialized on the host (`in`) with all values set to `1.0f`.
   - Memory is allocated on the GPU (device) and the data is copied from host to device.

2. **Reduction Kernel**:
   - Threads are grouped in blocks. Each block processes a **chunk (segment)** of the input array.
   - Threads **collaborate** to compute a partial sum for that segment.
   - These **partial sums are stored** in another array called `partialSums`.

3. **Final Reduction on CPU**:
   - After kernel execution, `partialSums` is copied back to the host.
   - A **final sum is computed on CPU** by adding up all the partial sums.

---

## 💡 Key Concepts Used

- **Thread Blocks and Grids**:
  - Threads are grouped into blocks; blocks form a grid.
  - Each block processes a fixed number of elements.

- **Shared Memory (optional versions)**:
  - Shared memory can be used to store intermediate sums within a block to reduce global memory accesses.

- **Thread Coarsening (in some versions)**:
  - Each thread processes more than one element, increasing efficiency.

- **Synchronization**:
  - `__syncthreads()` ensures all threads are at the same point before continuing computation.

---

## ✨ Why Use Reduction?

Reduction is a **classic problem in parallel computing** used in:
- Summing arrays
- Computing dot products
- Finding max/min in a list
- Histogramming
- And many more

---

## 🧠 Takeaway

This code teaches:
- **Efficient memory usage (shared vs global)**
- **Parallel algorithm design**
- **Thread indexing and synchronization**
- **Performance optimization with thread coarsening**

It's a foundational building block for **high-performance GPU programming** in CUDA.

---

Feel free to play around with block sizes, stride values, and input array sizes to see how performance changes!
