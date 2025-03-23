//atomic add
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define NUM_BINS 256

// GPU kernel: each thread processes one pixel and updates the corresponding bin atomically.
__global__ void histogram_kernel(unsigned char* image, unsigned int* bins, unsigned int width, unsigned int height) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < width * height) {
        // Use atomic add to prevent race conditions when updating bins.
        unsigned char b = image[i];
        atomicAdd(&bins[b], 1);
    }
}

// GPU function to compute the histogram.
void histogram_gpu(unsigned char* image, unsigned int* bins, unsigned int width, unsigned int height) {
    // Create CUDA events to time the kernel execution.
    cudaEvent_t start, stop;
    float kernelTime = 0.0f;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    size_t img_size = width * height * sizeof(unsigned char);
    size_t bins_size = NUM_BINS * sizeof(unsigned int);
    
    // Start overall GPU timing.
    auto gpu_start = std::chrono::high_resolution_clock::now();

    // Allocate device memory.
    unsigned char *image_d;
    unsigned int *bins_d;
    cudaMalloc((void**)&image_d, img_size);
    cudaMalloc((void**)&bins_d, bins_size);
    
    // Copy the image to the device.
    cudaMemcpy(image_d, image, img_size, cudaMemcpyHostToDevice);
    // Initialize device bins array to 0.
    cudaMemset(bins_d, 0, bins_size);
    
    // Launch kernel.
    unsigned int numThreadsPerBlock = 1024;
    unsigned int numBlocks = (width * height + numThreadsPerBlock - 1) / numThreadsPerBlock;
    cudaEventRecord(start, 0);
    histogram_kernel<<<numBlocks, numThreadsPerBlock>>>(image_d, bins_d, width, height);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&kernelTime, start, stop);
    
    // Copy the result back to host.
    cudaMemcpy(bins, bins_d, bins_size, cudaMemcpyDeviceToHost);
    
    // Free device memory.
    cudaFree(image_d);
    cudaFree(bins_d);
    
    auto gpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> gpu_total_time = gpu_end - gpu_start;
    printf("GPU kernel execution time: %f ms\n", kernelTime);
    printf("Total GPU operation time (including memory transfers): %f ms\n", gpu_total_time.count());

    // Cleanup CUDA events.
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// CPU function to compute the histogram for verification.
void histogram_cpu(unsigned char* image, unsigned int* bins, unsigned int width, unsigned int height) {
    // Initialize bins to 0.
    for (int i = 0; i < NUM_BINS; ++i) {
        bins[i] = 0;
    }
    
    // Compute the histogram.
    for (unsigned int i = 0; i < width * height; i++) {
        bins[image[i]]++;
    }
}

int main(){
    const unsigned int width = 256;
    const unsigned int height = 256;
    size_t img_size = width * height * sizeof(unsigned char);
    
    // Allocate memory for the image and histograms.
    unsigned char* image = new unsigned char[width * height];
    unsigned int* hist_gpu = new unsigned int[NUM_BINS];
    unsigned int* hist_cpu = new unsigned int[NUM_BINS];
    
    // Timer for image initialization.
    auto init_start = std::chrono::high_resolution_clock::now();
    // Initialize the image with random pixel values in the range [0, 255].
    for (unsigned int i = 0; i < width * height; ++i) {
        image[i] = rand() % 256;
    }
    auto init_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> init_time = init_end - init_start;
    printf("Image initialization time: %f microseconds\n", init_time.count());
    
    // Compute the histogram on the GPU and measure time.
    auto gpu_func_start = std::chrono::high_resolution_clock::now();
    histogram_gpu(image, hist_gpu, width, height);
    auto gpu_func_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> gpu_func_time = gpu_func_end - gpu_func_start;
    printf("Total GPU function time: %f microseconds\n", gpu_func_time.count());
    
    // Compute the histogram on the CPU and measure time.
    auto cpu_start = std::chrono::high_resolution_clock::now();
    histogram_cpu(image, hist_cpu, width, height);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> cpu_time = cpu_end - cpu_start;
    printf("CPU histogram computation time: %f microseconds\n", cpu_time.count());
    
    // Verify the GPU results by comparing to the CPU results.
    bool correct = true;
    for (int i = 0; i < NUM_BINS; i++) {
        if(hist_gpu[i] != hist_cpu[i]){
            correct = false;
            printf("Mismatch at bin %d: GPU = %u, CPU = %u\n", i, hist_gpu[i], hist_cpu[i]);
        }
    }
    
    if(correct)
        printf("Histogram verification PASSED!\n");
    else
        printf("Histogram verification FAILED!\n");
    
    
    // Free allocated memory.
    delete[] image;
    delete[] hist_gpu;
    delete[] hist_cpu;
    
    return 0;
}
