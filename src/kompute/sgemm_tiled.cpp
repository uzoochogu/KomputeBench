#include <format>
#include <iostream>
#include <kompute/Kompute.hpp>
#include <memory>
#include <random>
#include <shader/sgemm_tiled.hpp>
#include <vector>

#include "../data.hpp"  // mat A and B
#include "../utils.hpp"

int main() {
  kp::Manager mgr;

  std::vector<float> C(M * N, 0.0f);

  auto tensorA = mgr.tensor(std::vector<float>(A.begin(), A.end()));
  auto tensorB = mgr.tensor(std::vector<float>(B.begin(), B.end()));
  auto tensorC = mgr.tensor(C);

  // Parameters for the algorithm
  std::vector<std::shared_ptr<kp::Memory>> params = {tensorA, tensorB, tensorC};

  const std::vector<uint32_t> sgemm_shader =
      std::vector<uint32_t>(sgemm_shader::SGEMM_TILED_COMP_SPV.begin(),
                            sgemm_shader::SGEMM_TILED_COMP_SPV.end());

  std::shared_ptr<kp::Algorithm> algo =
      mgr.algorithm(params, sgemm_shader, kp::Workgroup({16, 16, 1}), {},
                    {float(M), float(N), float(K)});

  mgr.sequence()
      ->record<kp::OpSyncDevice>({tensorA, tensorB})
      ->record<kp::OpAlgoDispatch>(algo)
      ->record<kp::OpSyncLocal>({tensorC})
      ->eval();

  // Print results
  std::cout << std::format("Result matrix C ({}x{}):\n", M, N);
  for (uint32_t i = 0; i < M; i++) {
    for (uint32_t j = 0; j < N; j++) {
      std::cout << std::format("{:>8}", tensorC->data()[i * N + j]);
    }
    std::cout << "\n";
  }

  // Verify results
  std::vector<float> C_cpu(M * N, 0.0f);
  cpu_matmul_ikj(A, B, C_cpu, M, K, N);
  if (verifyResults(C, C_cpu, 1e-3f)) {
    printf("sgemm_tiled Result matches CPU Result\n");
  } else {
    printf("sgemm_tiled Result does not match CPU Result\n");
  }
}
