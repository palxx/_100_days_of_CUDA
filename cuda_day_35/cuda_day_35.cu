#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define IDX4(n, c, h, w, C, H, W) (((n)*(C)*(H)*(W)) + ((c)*(H)*(W)) + ((h)*(W)) + (w))

__global__ void group_norm_kernel(float *input, float *output, float *gamma, float *beta,
                                   int N, int C, int H, int W, int G, float epsilon) {
    int n = blockIdx.x;
    int g = blockIdx.y;
    int hw = H * W;
    int group_size = C / G;

    int base_c = g * group_size;

    extern __shared__ float sdata[];
    float *s_mean = sdata;
    float *s_var = sdata + blockDim.x;

    float local_sum = 0.0f;
    float local_var = 0.0f;

    for (int c = 0; c < group_size; ++c) {
        for (int i = threadIdx.x; i < hw; i += blockDim.x) {
            int h = i / W;
            int w = i % W;
            int idx = IDX4(n, base_c + c, h, w, C, H, W);
            local_sum += input[idx];
        }
    }

    s_mean[threadIdx.x] = local_sum;
    __syncthreads();

    // Reduce mean
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            s_mean[threadIdx.x] += s_mean[threadIdx.x + stride];
        __syncthreads();
    }
    float mean = s_mean[0] / (group_size * hw);

    for (int c = 0; c < group_size; ++c) {
        for (int i = threadIdx.x; i < hw; i += blockDim.x) {
            int h = i / W;
            int w = i % W;
            int idx = IDX4(n, base_c + c, h, w, C, H, W);
            float diff = input[idx] - mean;
            local_var += diff * diff;
        }
    }

    s_var[threadIdx.x] = local_var;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            s_var[threadIdx.x] += s_var[threadIdx.x + stride];
        __syncthreads();
    }
    float var = s_var[0] / (group_size * hw);

    for (int c = 0; c < group_size; ++c) {
        for (int i = threadIdx.x; i < hw; i += blockDim.x) {
            int h = i / W;
            int w = i % W;
            int idx = IDX4(n, base_c + c, h, w, C, H, W);
            float norm = (input[idx] - mean) / sqrtf(var + epsilon);
            output[idx] = gamma[base_c + c] * norm + beta[base_c + c];
        }
    }
}

void launch_group_norm(float *input, float *output, float *gamma, float *beta,
                        int N, int C, int H, int W, int G, float epsilon) {
    dim3 blocks(N, G);
    dim3 threads(256);
    size_t shared_size = 2 * threads.x * sizeof(float);
    group_norm_kernel<<<blocks, threads, shared_size>>>(input, output, gamma, beta, N, C, H, W, G, epsilon);
    cudaDeviceSynchronize();
}

int main() {
    int N = 1, C = 4, H = 2, W = 2, G = 2;
    float epsilon = 1e-5f;

    int size = N * C * H * W;
    float h_input[size], h_output[size], h_gamma[C], h_beta[C];

    for (int i = 0; i < size; ++i) h_input[i] = float(i);
    for (int i = 0; i < C; ++i) {
        h_gamma[i] = 1.0f;
        h_beta[i] = 0.0f;
    }

    float *d_input, *d_output, *d_gamma, *d_beta;
    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));
    cudaMalloc(&d_gamma, C * sizeof(float));
    cudaMalloc(&d_beta, C * sizeof(float));

    cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, h_gamma, C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, h_beta, C * sizeof(float), cudaMemcpyHostToDevice);

    launch_group_norm(d_input, d_output, d_gamma, d_beta, N, C, H, W, G, epsilon);

    cudaMemcpy(h_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < size; ++i) {
        printf("%f ", h_output[i]);
    }
    printf("\n");

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_gamma);
    cudaFree(d_beta);

    return 0;
}
