Today we are going to see a new parallel computation: Sparse Matrix Computation. 
We do this to save space and we are going to do case on "Sparse Matrix Multiplication"
There are two storage formats: 
1. Coordinate Format (COO)
2. Compressed Sparse Row (CSR)
These storage formats affects how we process sparse matrices.

This is an example of sparse matrix:

<img width="294" alt="Image" src="https://github.com/user-attachments/assets/8a50e3cb-5035-44e5-a1a6-aa62b17a453f" />

The advantage of this computation is such that we don't need to allocate memory, load or compute these zeros into or from the memory. (which saves us a lot of hassel)
