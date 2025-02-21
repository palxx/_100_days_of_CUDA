# ELL & COO Sparse Matrix Representation in CUDA 🚀

## **Overview**
This project implements **ELL (Ellpack-Itpack) and COO (Coordinate) sparse matrix representations** in **CUDA** to efficiently perform **Sparse Matrix-Vector Multiplication (SpMV)**. By leveraging **GPU parallelization**, we can optimize memory access patterns and computational efficiency for sparse matrices.

---

## **Why ELL & COO Formats?**
### **1️⃣ ELL (Ellpack-Itpack Format)**
- Stores a **fixed number of non-zero elements per row**.
- Uses **two arrays**:
  - `data_ell`: Stores non-zero values.
  - `indices_ell`: Stores column indices of non-zero values.
- **Best for structured sparsity**, where rows have a similar number of non-zeros.
- **Fast memory access on GPUs** but wastes memory if sparsity varies significantly per row.

### **2️⃣ COO (Coordinate List Format)**
- Stores **only the non-zero values** along with their **row and column indices**.
- Uses **three arrays**:
  - `data_coo`: Stores non-zero values.
  - `row_coo`: Stores row indices of non-zero values.
  - `col_coo`: Stores column indices of non-zero values.
- **Flexible and compact**, good for **highly irregular sparsity patterns**.
- **Slower than ELL** due to **atomic operations** in parallel implementations.

### **3️⃣ Hybrid Approach (ELL + COO)**
- **Uses ELL** for the **first `threshold` non-zeros per row**.
- **Uses COO** for any **additional non-zeros beyond the threshold**.
- **Optimizes memory and computational efficiency!**

---

## **CUDA Implementation**
### **1️⃣ Kernel for ELL Storage & SpMV**
Each **thread processes a row**:
```cpp
for (int i = 0; i < threshold; i++) {
    int col = indices_ell[i * N + row];
    if (col != -1) {
        acc += data_ell[i * N + row] * X[col];
    }
}
```
✅ **Fast, coalesced memory access** for structured sparsity.

### **2️⃣ Kernel for COO Storage & SpMV**
Each **thread processes a non-zero element**:
```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i >= nnz) return;
atomicAdd(&output[row_coo[i]], data_coo[i] * X[col_coo[i]]);
```
✅ **Handles irregular sparsity** but requires **atomic operations**.

---

## **Performance Comparison**
| **Format** | **Memory Efficiency** | **Performance on GPUs** | **Best Use Case** |
|------------|----------------------|----------------------|------------------|
| **ELL** | Fixed-size, wastes memory if sparsity varies | Fast, efficient for structured sparsity | Structured sparsity |
| **COO** | Compact, no wasted memory | Slower due to atomic operations | Irregular sparsity |
| **ELL + COO** | Balanced approach | Optimized for both structured & unstructured sparsity | Mixed sparsity |

---

## **How to Run** 🛠️
### **1️⃣ Compilation**
```sh
nvcc -arch=sm_75 .cu -o ell_coo_spmv
```
### **2️⃣ Execution**
```sh
./ell_coo_spmv
```

### **3️⃣ Expected Output**
```
CUDA kernel time: 0.0021 seconds
COO[0]: val = 24.0, row = 0, col = 3
COO[1]: val = 35.0, row = 1, col = 5
...
```
✅ **Final results stored in `cuda_results.txt`.**

---

## **Future Improvements**
🔹 **Dynamic COO allocation** instead of preallocating `N * M` memory.  
🔹 **Warp-wise optimizations** to improve COO memory access.  
🔹 **Further optimizations for deep learning sparse computations.**  

Would love to hear feedback! 🚀 #CUDA #SparseMatrices #ELL #COO
