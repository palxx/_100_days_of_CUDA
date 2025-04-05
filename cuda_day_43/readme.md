# Sparse Matrix-Vector Multiplication in CUDA

This repository provides three CUDA implementations of Sparse Matrix–Vector Multiplication (SpMV) using different sparse matrix storage formats: **CSR (Compressed Sparse Row)**, **JDS (Jagged Diagonal Storage)**, and **ELL (ELLPACK)**. Each implementation demonstrates how to handle sparse matrix data, perform computations on the GPU, and manage memory transfers between the host and device.

## Contents

- `csr_spmv.cu` - CUDA implementation using the CSR format.
- `jds_spmv.cu` - CUDA implementation using the JDS format.
- `ell_spmv.cu` - CUDA implementation using the ELL format.
- `README.md` - This documentation file.

## Overview

Sparse matrix-vector multiplication is a critical operation in many scientific computing and engineering applications. This project includes three different storage formats for sparse matrices:

1. **CSR (Compressed Sparse Row)**
   - Stores non-zero values, their column indices, and row pointers.
2. **JDS (Jagged Diagonal Storage)**
   - Reorders rows based on the number of non-zeros, optimizing for parallel execution.
3. **ELL (ELLPACK)**
   - Uses a fixed number of non-zeros per row with padding, ensuring uniform memory access.

Each example:
- Allocates and transfers matrix and vector data to the GPU.
- Launches a CUDA kernel to perform SpMV.
- Copies the result back to the host and prints it.

## Prerequisites

- **NVIDIA GPU** with CUDA support.
- **CUDA Toolkit** installed (includes the `nvcc` compiler).

## Compilation

To compile any of the provided files, navigate to the project directory in your terminal and run the following commands:

```bash
# Compile CSR example
nvcc -O2 -o csr_spmv csr_spmv.cu

# Compile JDS example
nvcc -O2 -o jds_spmv jds_spmv.cu

# Compile ELL example
nvcc -O2 -o ell_spmv ell_spmv.cu
