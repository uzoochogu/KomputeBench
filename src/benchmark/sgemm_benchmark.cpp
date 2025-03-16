#include <benchmark/benchmark.h>

#include <kompute/Kompute.hpp>
#include <shader/sgemm_naive_col.hpp>
#include <shader/sgemm_naive_v1.hpp>
#include <shader/sgemm_naive_v2.hpp>
#include <shader/sgemm_regblock_col.hpp>
#include <shader/sgemm_tiled_col.hpp>
#include <shader/sgemm_tiled_v1.hpp>
#include <shader/sgemm_tiled_v2.hpp>
#include <shader/sgemm_tiled_v3.hpp>
#include <shader/sgemm_tiled_v4.hpp>

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

static void Kompute_Naive_V1(benchmark::State& state) {
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
      std::vector<uint32_t>(sgemm_shader::SGEMM_NAIVE_V1_COMP_SPV.begin(),
                            sgemm_shader::SGEMM_NAIVE_V1_COMP_SPV.end());

  // Create algorithm with the shader
  std::shared_ptr<kp::Algorithm> algo =
      mgr.algorithm(params, sgemm_shader, kp::Workgroup({M / 32, N / 32, 1}),
                    {}, {float(M), float(N), float(K)});

  for (auto _ : state) {
    // Record and run sequence
    mgr.sequence()
        ->record<kp::OpSyncDevice>({tensorA, tensorB})
        ->record<kp::OpAlgoDispatch>(algo)
        ->eval();  // blocks until completion - appends a barrier
  }
}

static void Kompute_Naive_V2(benchmark::State& state) {
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
      std::vector<uint32_t>(sgemm_shader::SGEMM_NAIVE_V2_COMP_SPV.begin(),
                            sgemm_shader::SGEMM_NAIVE_V2_COMP_SPV.end());

  std::shared_ptr<kp::Algorithm> algo = mgr.algorithm(
      params, sgemm_shader, kp::Workgroup({M / 32, N / 32, 1}), {float(K)}, {});

  for (auto _ : state) {
    mgr.sequence()
        ->record<kp::OpSyncDevice>({tensorA, tensorB})
        ->record<kp::OpAlgoDispatch>(algo)
        ->eval();
  }
}

static void Kompute_Naive_Col(benchmark::State& state) {
  kp::Manager mgr;

  std::vector<float> A(M * K, 0.0f);
  std::vector<float> B(K * N, 0.0f);
  std::vector<float> C(M * N, 0.0f);

  initializeMatrix(A);
  initializeMatrix(B);

  A = transposeMatrix(A, M, K);
  B = transposeMatrix(B, K, N);

  auto tensorA = mgr.tensor(A);
  auto tensorB = mgr.tensor(B);
  auto tensorC = mgr.tensor(C);

  std::vector<std::shared_ptr<kp::Memory>> params = {tensorA, tensorB, tensorC};

  const std::vector<uint32_t> sgemm_shader =
      std::vector<uint32_t>(sgemm_shader::SGEMM_NAIVE_COL_COMP_SPV.begin(),
                            sgemm_shader::SGEMM_NAIVE_COL_COMP_SPV.end());

  std::shared_ptr<kp::Algorithm> algo =
      mgr.algorithm(params, sgemm_shader, kp::Workgroup({M / 8, N / 8, 1}), {},
                    {float(M), float(N), float(K)});

  for (auto _ : state) {
    mgr.sequence()
        ->record<kp::OpSyncDevice>({tensorA, tensorB})
        ->record<kp::OpAlgoDispatch>(algo)
        ->eval();
  }
}

static void Kompute_Tiled_V1(benchmark::State& state) {
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
      std::vector<uint32_t>(sgemm_shader::SGEMM_TILED_V1_COMP_SPV.begin(),
                            sgemm_shader::SGEMM_TILED_V1_COMP_SPV.end());

  std::shared_ptr<kp::Algorithm> algo =
      mgr.algorithm(params, sgemm_shader, kp::Workgroup({M / 32, N / 32, 1}),
                    {}, {float(M), float(N), float(K)});

  for (auto _ : state) {
    mgr.sequence()
        ->record<kp::OpSyncDevice>({tensorA, tensorB})
        ->record<kp::OpAlgoDispatch>(algo)
        ->eval();
  }
}

static void Kompute_Tiled_V2(benchmark::State& state) {
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
      std::vector<uint32_t>(sgemm_shader::SGEMM_TILED_V2_COMP_SPV.begin(),
                            sgemm_shader::SGEMM_TILED_V2_COMP_SPV.end());

  std::shared_ptr<kp::Algorithm> algo = mgr.algorithm(
      params, sgemm_shader, kp::Workgroup({M / 16, N / 16, 1}), {float(K)}, {});

  for (auto _ : state) {
    mgr.sequence()
        ->record<kp::OpSyncDevice>({tensorA, tensorB})
        ->record<kp::OpAlgoDispatch>(algo)
        ->eval();
  }
}

static void Kompute_Tiled_V3(benchmark::State& state) {
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
      std::vector<uint32_t>(sgemm_shader::SGEMM_TILED_V3_COMP_SPV.begin(),
                            sgemm_shader::SGEMM_TILED_V3_COMP_SPV.end());

  std::shared_ptr<kp::Algorithm> algo = mgr.algorithm(
      params, sgemm_shader, kp::Workgroup({M / 16, N / 16, 1}), {float(K)}, {});

  for (auto _ : state) {
    mgr.sequence()
        ->record<kp::OpSyncDevice>({tensorA, tensorB})
        ->record<kp::OpAlgoDispatch>(algo)
        ->eval();
  }
}

static void Kompute_Tiled_V4(benchmark::State& state) {
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
      std::vector<uint32_t>(sgemm_shader::SGEMM_TILED_V4_COMP_SPV.begin(),
                            sgemm_shader::SGEMM_TILED_V4_COMP_SPV.end());

  std::shared_ptr<kp::Algorithm> algo = mgr.algorithm(
      params, sgemm_shader, kp::Workgroup({M / 16, N / 16, 1}), {float(K)}, {});

  for (auto _ : state) {
    mgr.sequence()
        ->record<kp::OpSyncDevice>({tensorA, tensorB})
        ->record<kp::OpAlgoDispatch>(algo)
        ->eval();
  }
}

static void Kompute_Tiled_Col(benchmark::State& state) {
  kp::Manager mgr;

  std::vector<float> A(M * K, 0.0f);
  std::vector<float> B(K * N, 0.0f);
  std::vector<float> C(M * N, 0.0f);

  initializeMatrix(A);
  initializeMatrix(B);

  A = transposeMatrix(A, M, K);
  B = transposeMatrix(B, K, N);

  auto tensorA = mgr.tensor(A);
  auto tensorB = mgr.tensor(B);
  auto tensorC = mgr.tensor(C);

  std::vector<std::shared_ptr<kp::Memory>> params = {tensorA, tensorB, tensorC};

  const std::vector<uint32_t> sgemm_shader =
      std::vector<uint32_t>(sgemm_shader::SGEMM_TILED_COL_COMP_SPV.begin(),
                            sgemm_shader::SGEMM_TILED_COL_COMP_SPV.end());

  std::shared_ptr<kp::Algorithm> algo =
      mgr.algorithm(params, sgemm_shader, kp::Workgroup({M / 32, N / 32, 1}),
                    {}, {float(M), float(N), float(K)});

  for (auto _ : state) {
    mgr.sequence()
        ->record<kp::OpSyncDevice>({tensorA, tensorB})
        ->record<kp::OpAlgoDispatch>(algo)
        ->eval();
  }
}

static void Kompute_Register_Blocking_Col(benchmark::State& state) {
  kp::Manager mgr;

  std::vector<float> A(M * K, 0.0f);
  std::vector<float> B(K * N, 0.0f);
  std::vector<float> C(M * N, 0.0f);

  initializeMatrix(A);
  initializeMatrix(B);

  A = transposeMatrix(A, M, K);  // Only A is transposed.

  auto tensorA = mgr.tensor(A);
  auto tensorB = mgr.tensor(B);
  auto tensorC = mgr.tensor(C);

  std::vector<std::shared_ptr<kp::Memory>> params = {tensorA, tensorB, tensorC};

  const std::vector<uint32_t> sgemm_shader =
      std::vector<uint32_t>(sgemm_shader::SGEMM_REGBLOCK_COL_COMP_SPV.begin(),
                            sgemm_shader::SGEMM_REGBLOCK_COL_COMP_SPV.end());
  std::vector<float> push_constants = {
      static_cast<float>(M), static_cast<float>(N), static_cast<float>(K)};

  // Calculate grid dimensions based on tile sizes
  // TSM=128, TSN=128 from shader definitions
  uint32_t numGroupsM = (M + 127) / 128;  // Ceiling division by TSM
  uint32_t numGroupsN = (N + 127) / 128;  // Ceiling division by TSN

  // Create algorithm with workgroup size matching shader local_size
  // RTSM=16, RTSN=16 from shader definitions
  std::shared_ptr<kp::Algorithm> algo = mgr.algorithm(
      params, sgemm_shader, kp::Workgroup({numGroupsM, numGroupsN, 1}), {},
      push_constants);

  for (auto _ : state) {
    mgr.sequence()
        ->record<kp::OpSyncDevice>({tensorA, tensorB})
        ->record<kp::OpAlgoDispatch>(algo)
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
    CHECK_CUDA(cudaDeviceSynchronize());
  }
  CHECK_CUDA(cudaFree(d_A));
  CHECK_CUDA(cudaFree(d_B));
  CHECK_CUDA(cudaFree(d_C));
  CHECK_CUBLAS(cublasDestroy(handle));
}

BENCHMARK(CUBLAS)->Arg(100);
BENCHMARK(Kompute_Naive_V1)->Arg(100);
BENCHMARK(Kompute_Naive_V2)->Arg(100);
BENCHMARK(Kompute_Naive_Col)->Arg(100);

BENCHMARK(Kompute_Tiled_V1)->Arg(100);
BENCHMARK(Kompute_Tiled_V2)->Arg(100);
BENCHMARK(Kompute_Tiled_V3)->Arg(100);
BENCHMARK(Kompute_Tiled_V4)->Arg(100);
BENCHMARK(Kompute_Tiled_Col)->Arg(100);

BENCHMARK(Kompute_Register_Blocking_Col)->Arg(100);
// BENCHMARK(CPU_Matmul)->Arg(10); // too slow

BENCHMARK_MAIN();
