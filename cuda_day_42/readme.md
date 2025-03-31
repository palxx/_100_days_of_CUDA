Today we are going to see a new parallel computation: Sparse Matrix Computation. 
We do this to save space and we are going to do case on "Sparse Matrix Multiplication"
There are two storage formats: 
1. Coordinate Format (COO)
2. Compressed Sparse Row (CSR)
3. ELLPACK Format (ELL)
4. Jagged Diagonal Format (JDL)
These storage formats affects how we process sparse matrices.

This is an example of sparse matrix:

<img width="294" alt="Image" src="https://github.com/user-attachments/assets/8a50e3cb-5035-44e5-a1a6-aa62b17a453f" />

The advantage of this computation is such that we don't need to allocate memory, load or compute these zeros into or from the memory. (which saves us a lot of hassel)

The main issue for us starts with storage format for sparse matrices, there are many formats as mentioned above and more. 

So the choice of the format matters alot, according to our certain type of application it can vary, suppose the need arises as follows:
1. Space Efficieny
2. Flexibility
3. Accessibility
4. Memory access pattern (Memory Coalesing)
5. Load Balancing (Minimize Control Divergence)

According to your need, format changes.

We are going to study Sparse Matrix - Vector Multiplication, there can be dense matrix-vector multiplication, sparse matrix multiplication and there can be different types of matrix. This is just one of the cases, we will be seeing:

<img width="608" alt="Image" src="https://github.com/user-attachments/assets/4287d5f0-289c-4b19-b5f8-521bc8b2dabc" />

The first format we will be seeing is:
### Coordinate Format - COO Format

<img width="1018" alt="Image" src="https://github.com/user-attachments/assets/85ea91a5-bfee-4341-aae9-11f861f943bd" />

In the image you can see, there is a sparse matrix. We store this matrix in row major format into an array and for every corresponding element, we have other 2 array column and row, where we store the subsequent element's position from sparse matrix.

There can issues such that storage required for a sparse matrix is more than dense matrix, but when the matrix becomes large (less than 1% of the storage is occupied) this method becomes useful.

<img width="1098" alt="Screenshot 2025-03-31 at 7 16 49 AM" src="https://github.com/user-attachments/assets/46c2df20-ab40-43d9-9cc3-397da60b772f" />
 
 We will assign threads to each value element in the array and add to its corresponsing location in the output matrix --> do atomic operation because multiple threads are performing operations at the same location in the output matrix.

 
 












