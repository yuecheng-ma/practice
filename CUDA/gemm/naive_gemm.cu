#include "cuda_gemm.h"
#include <cuda_runtime.h>
#include <bits/stdc++.h>

__global__ void cuda_gemm_kernel(const float* A, const float* B, float* C, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row >= M || col >= N) {
        return;
    }

    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

void cuda_gemm(const float* h_A, const float* h_B, float* h_C, int M, int K, int N, int block_size) {
    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;

    cudaMalloc((void**)&d_A, M*K*sizeof(float));
    cudaMalloc((void**)&d_B, K*N*sizeof(float));
    cudaMalloc((void**)&d_C, M*N*sizeof(float));

    cudaMemcpy(d_A, h_A, M*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K*N*sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(block_size, block_size);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    cuda_gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, M, K, N);

    cudaMemcpy(h_C, d_C, M*N*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}