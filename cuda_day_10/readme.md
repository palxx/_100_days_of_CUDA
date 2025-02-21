# CUDA-Based Self-Attention Implementation

## Overview
This project implements **self-attention** in CUDA using two different approaches:

1. **DAY 09 1 (CODE 1): Traditional Attention Kernel** - Implements self-attention using **shared memory tiling** and global memory operations.
2. **DAY 10 2 (CODE 2): FlashAttention Style Kernel** - Uses **block-wise computation** and **optimized memory access** for improved performance.

Both implementations aim to accelerate self-attention computations on **NVIDIA GPUs** using CUDA.

---

## **1️⃣ Traditional Self-Attention (CODE 1)**
### **Key Features:**
- Uses **shared memory tiles** for Query (`Q`), Key (`K`), Value (`V`), and Scores (`S`).
- Loads entire **blocks of `Q`, `K`, and `V`** into shared memory before computation.
- Computes **softmax row-wise**, tracking **row max (`row_max`)** and **sum (`row_sum`)** for numerical stability.
- Writes computed **attention output (`O`)** to global memory.

### **Execution Flow:**
1. **Load `K` and `V` blocks into shared memory** (Column-wise access)
2. **Iterate over `Q` blocks** (Row-wise access)
3. **Compute `Q × K^T` to get attention scores**
4. **Apply softmax normalization** using row-wise tracking
5. **Compute final output: `P × V` (softmax scores × values)**
6. **Store results back to global memory**

### **Pros & Cons:**
✅ **Pros:**
- Works well for small and medium sequence lengths.
- Ensures correct softmax computation with **row max tracking**.

❌ **Cons:**
- **Shared memory conflicts** can occur with large batch sizes.
- **Frequent memory access stalls** in global memory.
- **Does not scale efficiently** for long sequences.

---

## **2️⃣ FlashAttention Style (CODE 2)**
### **Key Features:**
- Uses **local thread storage** instead of excessive shared memory.
- Loads `K` and `V` **column-wise** and `Q` **row-wise** dynamically.
- Computes **softmax per block**, reducing **numerical precision issues**.
- Optimizes memory access **to avoid excessive global memory writes**.

### **Execution Flow:**
1. **Load a small tile of `K` and `V` into thread-local storage**.
2. **Load `Q` row-wise into thread-local storage**.
3. **Compute `Q × K^T` for attention scores per thread block**.
4. **Perform softmax with row-wise normalization**.
5. **Compute output as `P × V` efficiently per thread**.
6. **Store results efficiently to minimize global memory writes**.

### **Pros & Cons:**
✅ **Pros:**
- **Better memory efficiency** due to localized computations.
- **Scales well** to large sequence lengths (**longer inputs = better speedup**).
- **Numerically stable** without needing row-wise max tracking in shared memory.

❌ **Cons:**
- **More complex indexing and block-wise scheduling**.
- **Might underutilize GPU cores** for very small sequence lengths.

---

## **3️⃣ Performance Comparison**
| Feature | Traditional Attention (CODE 1) | FlashAttention (CODE 2) |
|---------|-----------------------------|-------------------------|
| **Memory Usage** | Uses **large shared memory buffers** | Uses **local memory for better efficiency** |
| **Parallelization** | **Loads full matrices into shared memory** | **Loads blocks dynamically** |
| **Softmax Computation** | Uses **row-wise tracking** | Uses **block-wise softmax tracking** |
| **Global Memory Usage** | Writes frequently to global memory | Minimizes global memory accesses |
| **Numerical Stability** | Uses **row max and sum tracking** | Uses **block-wise normalization** |
| **Performance** | **Slower for long sequences** | **Scales better for long sequences** |

✅ **FlashAttention (CODE 2) is the recommended approach for large-scale self-attention computations!**

---

## **4️⃣ How to Compile and Run**
### **Prerequisites**
- **NVIDIA GPU** with CUDA support
- **CUDA Toolkit Installed**
- **C++ Compiler (e.g., `g++`)**

### **Compilation**
```sh
# Compile the first code (Traditional Attention)
nvcc -arch=sm_75 cuda_day_10.cu -o attention_traditional

# Compile the second code (FlashAttention Style)
nvcc -arch=sm_75 flash_attention.cu -o attention_flash
```

### **Running the Program**
```sh
# Run Traditional Attention
./attention_traditional

# Run FlashAttention Style
./attention_flash
```

---

## **5️⃣ Future Improvements**
- **Support for Multi-Head Attention (MHA)**
- **Further optimization of memory access patterns**
- **Implementation of dropout and layer normalization in CUDA**
