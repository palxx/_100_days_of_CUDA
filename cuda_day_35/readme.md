# Group Normalization in CUDA

This project demonstrates a simple CUDA implementation of **Group Normalization (GN)** for input tensors of shape `N x C x H x W`. Group Normalization is widely used in deep learning tasks when batch sizes are small or in specific neural network layers where traditional batch normalization isn't ideal.

---

### 📂 Files
- `group_norm.cu`: Main CUDA C++ source code implementing group normalization kernel and testing with sample data.

---

### 🧠 Concept
Group Normalization divides the channels into groups and computes mean and variance for each group. The kernel then normalizes each value in the group using the formula:

```
x_norm = (x - mean) / sqrt(var + epsilon)
out = gamma * x_norm + beta
```

This implementation uses shared memory for efficient reduction of mean and variance and avoids the `atomicAdd()` limitation.

---

### 🚀 How to Build and Run
1. Ensure you have CUDA Toolkit installed (`nvcc` compiler).
2. Compile the code:
```bash
nvcc -o group_norm group_norm.cu
```
3. Run the executable:
```bash
./group_norm
```

---

### ⚙️ Parameters
You can configure the following parameters in the code:
- `N`: Number of data points (batch size)
- `C`: Channels per data point
- `H`: Height
- `W`: Width
- `G`: Number of groups (must divide `C` evenly)
- `epsilon`: Small float added to avoid divide-by-zero

---

### 📈 Output
The output is a printed array of normalized values after applying group normalization with scaling and shifting (`gamma`, `beta`).

---

### 💡 Future Improvements
- Add support for `float16` (half precision)
- Add integration for PyCUDA or CuPy
- Extend to handle 3D tensors (e.g., volumetric data)
- Add unit testing and benchmarks

---

### 📜 License
MIT License — feel free to use, modify, and share.

---

### 👨‍💻 Author
CUDA Group Norm Project — by [Your Name Here]

---

Happy coding! ⚡
