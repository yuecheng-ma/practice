#ifndef CUDA_GEMM_H
#define CUDA_GEMM_H

#ifdef __cplusplus
extern "C" {
#endif

void cuda_gemm(const float* h_A, const float* h_B, float* h_C, int M, int K, int N, int block_size = 16);

#ifdef __cplusplus
}
#endif

#endif // CUDA_GEMM_H