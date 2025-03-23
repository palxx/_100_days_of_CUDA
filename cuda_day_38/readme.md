### ** Merge sort using parallel computation
Parallelization approach is dividing partition. How does each? the output into equal size segments and assign a threat to perform a sequential merge of each segment


<img width="726" alt="Image" src="https://github.com/user-attachments/assets/af8e35ac-0676-4889-a09b-e952fed5c363" />

Now, the challenge came of how does each thread find the input segments corresponding to its output segment?
1. We are going to partition the output C into equal parts
2. As we can see in the image, then we are going to find I and J given K.
   i) then we are finding i given k (suppose could be find J given K as well)
   ii) then we find j = k - i
   iii) setting bounds on i, 0<=i<=m
   iv) setting bounds on j, 0<= j <=n, 0<= k-i <=n
