// sample cuBLAS program derived from
// https://github.com/Infatoshi/cuda-course/
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include "../cuda_utilities.hpp"
#include "../data.hpp"  // mat A, B
#include "../utils.hpp"

#undef PRINT_MATRIX
#define PRINT_MATRIX(mat, rows, cols)                                   \
  for (int i = 0; i < rows; i++) {                                      \
    for (int j = 0; j < cols; j++) printf("%8.3f ", mat[i * cols + j]); \
    printf("\n");                                                       \
  }                                                                     \
  printf("\n");

int main() {
  float C_cpu[M * N], C_cublas_s[M * N], C_cublas_h[M * N];

  // CPU matmul using ijk loop order
  cpu_matmul_ikj(A, B, std::span<float>(C_cpu, M * N), M, K, N);

  // CUDA setup
  cublasHandle_t handle;
  CHECK_CUBLAS(cublasCreate(&handle));

  float *d_A, *d_B, *d_C;
  CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(float)));

  CHECK_CUDA(
      cudaMemcpy(d_A, A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(
      cudaMemcpy(d_B, B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));

  // cuBLAS SGEMM
  float alpha = 1.0f, beta = 0.0f;
  CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                           d_B, N, d_A, K, &beta, d_C, N));
  CHECK_CUDA(cudaMemcpy(C_cublas_s, d_C, M * N * sizeof(float),
                        cudaMemcpyDeviceToHost));

  // cuBLAS HGEMM
  half *d_A_h, *d_B_h, *d_C_h;
  CHECK_CUDA(cudaMalloc(&d_A_h, M * K * sizeof(half)));
  CHECK_CUDA(cudaMalloc(&d_B_h, K * N * sizeof(half)));
  CHECK_CUDA(cudaMalloc(&d_C_h, M * N * sizeof(half)));

  // Convert to half precision on CPU
  half A_h[M * K], B_h[K * N];
  for (int i = 0; i < M * K; i++) {
    A_h[i] = __float2half(A[i]);
  }
  for (int i = 0; i < K * N; i++) {
    B_h[i] = __float2half(B[i]);
  }

  // Copy half precision data to device
  CHECK_CUDA(
      cudaMemcpy(d_A_h, A_h, M * K * sizeof(half), cudaMemcpyHostToDevice));
  CHECK_CUDA(
      cudaMemcpy(d_B_h, B_h, K * N * sizeof(half), cudaMemcpyHostToDevice));

  __half alpha_h = __float2half(1.0f), beta_h = __float2half(0.0f);
  CHECK_CUBLAS(cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha_h,
                           d_B_h, N, d_A_h, K, &beta_h, d_C_h, N));

  // Copy result back to host and convert to float
  half C_h[M * N];
  CHECK_CUDA(
      cudaMemcpy(C_h, d_C_h, M * N * sizeof(half), cudaMemcpyDeviceToHost));
  for (int i = 0; i < M * N; i++) {
    C_cublas_h[i] = __half2float(C_h[i]);
  }

  // Print results
  printf("Matrix A (%dx%d):\n", M, K);
  PRINT_MATRIX(A.data(), M, K);
  printf("Matrix B (%dx%d):\n", K, N);
  PRINT_MATRIX(B.data(), K, N);
  printf("CPU Result (%dx%d):\n", M, N);
  PRINT_MATRIX(C_cpu, M, N);
  printf("cuBLAS SGEMM Result (%dx%d):\n", M, N);
  PRINT_MATRIX(C_cublas_s, M, N);
  printf("cuBLAS HGEMM Result (%dx%d):\n", M, N);
  PRINT_MATRIX(C_cublas_h, M, N);

  // Compare results
  if (compareResults(std::span<float>(C_cpu, M * N),
                     std::span<float>(C_cublas_s, M * N), 1e-3f)) {
    printf("cuBLAS SGEMM Result matches CPU Result\n");
  } else {
    printf("cuBLAS SGEMM Result does not match CPU Result\n");
  }

  // Clean up
  CHECK_CUDA(cudaFree(d_A));
  CHECK_CUDA(cudaFree(d_B));
  CHECK_CUDA(cudaFree(d_C));
  CHECK_CUDA(cudaFree(d_A_h));
  CHECK_CUDA(cudaFree(d_B_h));
  CHECK_CUDA(cudaFree(d_C_h));
  CHECK_CUBLAS(cublasDestroy(handle));

  return 0;
}
