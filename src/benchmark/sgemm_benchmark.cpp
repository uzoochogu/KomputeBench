#include <benchmark/benchmark.h>

#include <kompute/Kompute.hpp>
#include <shader/sgemm_naive.hpp>
#include <shader/sgemm_tiled.hpp>

#include "../cuda_utilities.hpp"
#include "../utils.hpp"

constexpr const uint32_t M = 4096;
constexpr const uint32_t K = 4096;
constexpr const uint32_t N = 4096;

static void CPU_Matmul(benchmark::State& state) {
  std::vector<float> A(M * K, 0.0f);
  std::vector<float> B(K * N, 0.0f);
  std::vector<float> C(M * N, 0.0f);

  initializeMatrix(A);
  initializeMatrix(B);
  for (auto _ : state) {
    cpu_matmul_ikj(A, B, C, M, K, N);
  }
}

static void Kompute_Naive(benchmark::State& state) {
  kp::Manager mgr;

  std::vector<float> A(M * K, 0.0f);
  std::vector<float> B(K * N, 0.0f);
  std::vector<float> C(M * N, 0.0f);

  initializeMatrix(A);
  initializeMatrix(B);

  auto tensorA = mgr.tensor(A);
  auto tensorB = mgr.tensor(B);
  auto tensorC = mgr.tensor(C);

  // Parameters for the algorithm
  std::vector<std::shared_ptr<kp::Memory>> params = {tensorA, tensorB, tensorC};

  const std::vector<uint32_t> sgemm_shader =
      std::vector<uint32_t>(sgemm_shader::SGEMM_NAIVE_COMP_SPV.begin(),
                            sgemm_shader::SGEMM_NAIVE_COMP_SPV.end());

  // Create algorithm with the shader
  std::shared_ptr<kp::Algorithm> algo =
      mgr.algorithm(params, sgemm_shader, kp::Workgroup({32, 32, 1}), {},
                    {float(M), float(N), float(K)});

  for (auto _ : state) {
    // Record and run sequence
    mgr.sequence()
        ->record<kp::OpSyncDevice>({tensorA, tensorB, tensorC})
        ->record<kp::OpAlgoDispatch>(algo)
        ->record<kp::OpSyncLocal>({tensorC})
        ->eval();
  }
}

static void Kompute_Tiled(benchmark::State& state) {
  kp::Manager mgr;

  std::vector<float> A(M * K, 0.0f);
  std::vector<float> B(K * N, 0.0f);
  std::vector<float> C(M * N, 0.0f);

  initializeMatrix(A);
  initializeMatrix(B);

  auto tensorA = mgr.tensor(A);
  auto tensorB = mgr.tensor(B);
  auto tensorC = mgr.tensor(C);

  std::vector<std::shared_ptr<kp::Memory>> params = {tensorA, tensorB, tensorC};

  const std::vector<uint32_t> sgemm_shader =
      std::vector<uint32_t>(sgemm_shader::SGEMM_TILED_COMP_SPV.begin(),
                            sgemm_shader::SGEMM_TILED_COMP_SPV.end());

  // 16x16 workgroup size
  std::shared_ptr<kp::Algorithm> algo =
      mgr.algorithm(params, sgemm_shader, kp::Workgroup({16, 16, 1}), {},
                    {float(M), float(N), float(K)});

  for (auto _ : state) {
    mgr.sequence()
        ->record<kp::OpSyncDevice>({tensorA, tensorB, tensorC})
        ->record<kp::OpAlgoDispatch>(algo)
        ->record<kp::OpSyncLocal>({tensorC})
        ->eval();
  }
}

static void CUBLAS(benchmark::State& state) {
  std::vector<float> A(M * K, 0.0f);
  std::vector<float> B(K * N, 0.0f);
  std::vector<float> C(M * N, 0.0f);

  initializeMatrix(A);
  initializeMatrix(B);

  // cuBLAS setup
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

  float alpha = 1.0f, beta = 0.0f;
  for (auto _ : state) {
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                             d_B, N, d_A, K, &beta, d_C, N));
    CHECK_CUDA(cudaMemcpy(C.data(), d_C, M * N * sizeof(float),
                          cudaMemcpyDeviceToHost));
  }
  CHECK_CUDA(cudaFree(d_A));
  CHECK_CUDA(cudaFree(d_B));
  CHECK_CUDA(cudaFree(d_C));
  CHECK_CUBLAS(cublasDestroy(handle));
}

BENCHMARK(CUBLAS)->Arg(100);
BENCHMARK(Kompute_Naive)->Arg(100);
BENCHMARK(Kompute_Tiled)->Arg(100);
// BENCHMARK(CPU_Matmul)->Arg(10); // too slow

BENCHMARK_MAIN();
