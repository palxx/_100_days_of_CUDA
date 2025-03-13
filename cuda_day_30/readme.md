# 🚀 CUDA 3D Stencil Computation – `cuda_day_30.cu`

This CUDA program implements a **3D stencil computation** using shared memory and temporal reuse techniques to improve performance. The goal is to process a 3D grid efficiently with neighbor dependency (stencil pattern), which is a common operation in scientific computing and simulations.

---

## 📌 Key Definitions

- **`BLOCK_DIM`**: `8`
- **`IN_TD`**: Block dimension (equal to `BLOCK_DIM`)
- **`OUT_TD`**: Output tile dimension `IN_TD - 2` (accounts for halo boundaries)

---

## 🧠 What It Does

- Performs **3D stencil computation** using shared memory.
- Uses `iprev`, `iCurr_s` (shared), and `innext` to optimize memory accesses across iterations.
- Uses **CUDA Events** to measure:
  - Device memory allocation time
  - Data transfer time (Host ↔ Device)
  - Kernel execution time
