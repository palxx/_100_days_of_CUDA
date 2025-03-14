# CUDA Day 31 – Parallel Reduction with CUDA

This program implements a **parallel reduction** algorithm in CUDA to compute the sum of a large array using **GPU acceleration**. The reduction is performed using CUDA kernel functions and optimized memory access patterns for better performance.

---

## 🔍 Overview

- Each block performs reduction in parallel using a **binary tree-style summation** technique.
- Final results (partial sums from blocks) are summed on the CPU.
- GPU timings are captured using **CUDA events** to evaluate memory transfer and kernel execution performance.

---

## 📂 Files
- `cuda_day_31.cu`: Contains the full implementation of the CUDA parallel reduction algorithm along with timing and memory operations.

---

## ⚙️ How It Works

### 🔸 Kernel Execution
- Each thread block processes a **segment of the array**.
- Reduction is performed inside a kernel using **shared memory** and **strided pairwise summation**.

### 🔸 Memory Transfers
- Host to Device and Device to Host memory transfers are timed.
- Device memory allocation and cleanup are included.

### 🔸 Performance Timing
- Uses `cudaEventRecord` to measure:
  - Memory allocation time
  - Memory transfer time (Host ↔ Device)
  - Kernel execution time
  - Final result accumulation on CPU

---

## 📌 Key Definitions

| Macro            | Description                                  |
|------------------|----------------------------------------------|
| `BLOCK_DIM`      | Number of threads per block (1024)           |
| `numThreadsBlock`| Number of threads per block used in kernel   |
| `elePerBlock`    | Elements processed by each block (2 * BLOCK_DIM) |
| `numBlocks`      | Total blocks required for given N elements   |

---

## 🔢 Sample Output (Printed on Console)
Allocate Time: 0.127616 ms
Copy to GPU time: 0.048864 ms
Partial sums allocation time: 0.012288 ms
Kernel Execution Time: 0.154496 ms
Copy from GPU: 0.041152 ms
Print Partial Sums: 0.008544 ms
