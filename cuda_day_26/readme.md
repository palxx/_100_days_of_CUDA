# **Chapter 7: Convolution**
This is the first parallel pattren we are studying and learning a new feature called "Constant Memory"
1. Here, every thread is assigned each pixel and each pixel is calculated by help of something called a "Convolution Mask" aka "Convolution Kernel"
2. In general, the weights are determined by convolutional mask and the pixels which we want to change is the average of neighbouring pixels with weights guven to it by convolution mask.

## **1️⃣: Applications of convolution:**
1. usually to transform signal (1D convolution, or image 2D convolution)
2. Gaussian blur sharpen, edge detection, etc

## **2️⃣: Approach:**
1. Since the convolution mask we have is small and all threads in a grid are going to access the same mask, we assign it to constant memory that doesn't change.
2. The size of constant memrory is 64KB.
3. it is declared as,    ```cpp __constant__ float mask_c[mask_dim][mask_dim]```
4. to copy mask from host to gpu we use, cudaMemcpyToSymbol(mask_c, mask, mask_dim*mask_dim*sizeof(float));

## **Benefits of constant memory:**
1. we can store the mask in shared memory too, but the extra overhead of updating dirty bits and keeping track of changes is minimized
2. since we are using constant memory and multiple thread are accessing we dont have to care about cache coherance accorss various SM's.

Therefore, because of all these reasons we are using constant memory over shared memory. 
