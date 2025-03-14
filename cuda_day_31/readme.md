# 🚀 CUDA Day 31: Parallel Reduction using CUDA

This project implements a **parallel reduction** operation on a large array using **CUDA**. The goal is to perform an efficient summation of elements in a large array using the GPU. This code demonstrates how reduction can be optimized using shared memory and multiple kernel blocks.

---

## 📌 Problem Statement

- Given a large array of floating-point numbers, compute the **sum of all elements** using parallel GPU computation.

---

## 📐 Key Concepts Covered

### ✅ Parallel Reduction
- The array is broken into **segments** processed by CUDA blocks.
- Each block loads its segment into **shared memory**, performs reduction in **log₂(N)** steps.
- Partial results are written to `partialSums` array.
- Final summation is done on CPU for simplicity.

### ✅ Shared Memory
- Used for fast intra-block communication between threads.

### ✅ Performance Measurement
- Code includes **CUDA Events** to measure execution time for:
  - Memory Allocation
  - Copying to GPU
  - Kernel Execution
  - Copying results from GPU
  - Final Summation

---

## 📊 Kernel Logic

```cpp
__global__ void reduce_kernel(float* input, float* partialSums, unsigned int N) {
    // Load two elements per thread and perform first step of reduction
    // Store intermediate result in shared memory
    // Use binary tree reduction to sum elements
    // First thread of each block writes final block sum
}
