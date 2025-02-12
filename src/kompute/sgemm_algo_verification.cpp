#include <array>
#include <format>
#include <iostream>
#include <kompute/Kompute.hpp>
#include <memory>
#include <shader/sgemm_naive_col.hpp>
#include <shader/sgemm_naive_v1.hpp>
#include <shader/sgemm_naive_v2.hpp>
#include <shader/sgemm_regblock_col.hpp>
#include <shader/sgemm_tiled_col.hpp>
#include <shader/sgemm_tiled_v1.hpp>
#include <shader/sgemm_tiled_v2.hpp>
#include <shader/sgemm_tiled_v3.hpp>
#include <shader/sgemm_tiled_v4.hpp>
#include <vector>

#include "../utils.hpp"

constexpr uint32_t M = 1024;
constexpr uint32_t K = 1024;
constexpr uint32_t N = 1024;

struct ShaderConfig {
  std::string name;
  std::vector<uint32_t> spirv;
  kp::Workgroup workgroup;
  bool needs_column_major_A;
  bool needs_column_major_B;
  bool uses_spec_constants;
};

int main() {
  kp::Manager mgr;

  // Initialize matrices
  std::vector<float> A(M * K), B(K * N), C(M * N, 0.0f);
  initializeMatrix(A);
  initializeMatrix(B);

  // Column major versions
  auto A_col = transposeMatrix(A, M, K);
  auto B_col = transposeMatrix(B, K, N);

  std::cout << std::format("Matrix A ({}x{}):\n", M, N);
  for (uint32_t i = 0; i < M; i += 128) {
    for (uint32_t j = 0; j < N; j += 128) {
      std::cout << std::format("{:>8.4f}", A[i * N + j]);
    }
    std::cout << "\n";
  }
  std::cout << std::format("\n\ntranspose(A) ({}x{}):\n", N, M);
  for (uint32_t i = 0; i < M; i += 128) {
    for (uint32_t j = 0; j < N; j += 128) {
      std::cout << std::format("{:>8.4f}", A_col[i * N + j]);
    }
    std::cout << "\n";
  }

  std::cout << "\n\n";

  // Create tensors
  auto tensorA = mgr.tensor(A);
  auto tensorB = mgr.tensor(B);
  auto tensorA_col = mgr.tensor(A_col);
  auto tensorB_col = mgr.tensor(B_col);

  // For 2D Register Blocking
  // Grid dimensions based on tile sizes
  // TSM=128, TSN=128 from shader definitions
  uint32_t numGroupsM = (M + 127) / 128;  // Ceiling division by TSM
  uint32_t numGroupsN = (N + 127) / 128;  // Ceiling division by TSN

  // Define shader configurations
  auto shaderConfigs = std::to_array<ShaderConfig>(
      {{"Naive V1",
        std::vector<uint32_t>(sgemm_shader::SGEMM_NAIVE_V1_COMP_SPV.begin(),
                              sgemm_shader::SGEMM_NAIVE_V1_COMP_SPV.end()),
        kp::Workgroup({M / 32, N / 32, 1}), false, false, false},
       {"Naive V2",
        std::vector<uint32_t>(sgemm_shader::SGEMM_NAIVE_V2_COMP_SPV.begin(),
                              sgemm_shader::SGEMM_NAIVE_V2_COMP_SPV.end()),
        kp::Workgroup({M / 32, N / 32, 1}), false, false, true},
       {"Naive Col",
        std::vector<uint32_t>(sgemm_shader::SGEMM_NAIVE_COL_COMP_SPV.begin(),
                              sgemm_shader::SGEMM_NAIVE_COL_COMP_SPV.end()),
        kp::Workgroup({M / 8, N / 8, 1}), true, true, false},
       {"Tiled V1",
        std::vector<uint32_t>(sgemm_shader::SGEMM_TILED_V1_COMP_SPV.begin(),
                              sgemm_shader::SGEMM_TILED_V1_COMP_SPV.end()),
        kp::Workgroup({M / 32, N / 32, 1}), false, false, false},
       {"Tiled V2",
        std::vector<uint32_t>(sgemm_shader::SGEMM_TILED_V2_COMP_SPV.begin(),
                              sgemm_shader::SGEMM_TILED_V2_COMP_SPV.end()),
        kp::Workgroup({M / 16, N / 16, 1}), false, false, true},
       {"Tiled V3",
        std::vector<uint32_t>(sgemm_shader::SGEMM_TILED_V3_COMP_SPV.begin(),
                              sgemm_shader::SGEMM_TILED_V3_COMP_SPV.end()),
        kp::Workgroup({M / 16, N / 16, 1}), false, false, true},
       {"Tiled V4",
        std::vector<uint32_t>(sgemm_shader::SGEMM_TILED_V4_COMP_SPV.begin(),
                              sgemm_shader::SGEMM_TILED_V4_COMP_SPV.end()),
        kp::Workgroup({M / 16, N / 16, 1}), false, false, true},
       {"Tiled Col",
        std::vector<uint32_t>(sgemm_shader::SGEMM_TILED_COL_COMP_SPV.begin(),
                              sgemm_shader::SGEMM_TILED_COL_COMP_SPV.end()),
        kp::Workgroup({M / 32, N / 32, 1}), true, true, false},
       {"2D Register Blocking",
        std::vector<uint32_t>(sgemm_shader::SGEMM_REGBLOCK_COL_COMP_SPV.begin(),
                              sgemm_shader::SGEMM_REGBLOCK_COL_COMP_SPV.end()),
        kp::Workgroup({numGroupsM, numGroupsN, 1}), true, false,
        false}});  // only A is converted to column major

  // Results storage
  std::vector<std::shared_ptr<kp::TensorT<float>>> results;
  results.reserve(shaderConfigs.size());

  // Push/Spec constants
  std::vector<float> push_constants = {float(M), float(N), float(K)};
  std::vector<float> spec_constants = {float(K)};

  // Execute all shaders
  for (const auto& config : shaderConfigs) {
    auto result = mgr.tensor(C);

    std::vector<std::shared_ptr<kp::Memory>> params = {
        (config.needs_column_major_A ? tensorA_col : tensorA),
        (config.needs_column_major_B ? tensorB_col : tensorB), result};

    std::shared_ptr<kp::Algorithm> algo = mgr.algorithm(
        params, config.spirv, config.workgroup,
        config.uses_spec_constants ? spec_constants : std::vector<float>{},
        config.uses_spec_constants ? std::vector<float>{} : push_constants);

    mgr.sequence()
        ->record<kp::OpSyncDevice>({params[0], params[1]})
        ->record<kp::OpAlgoDispatch>(algo)
        ->record<kp::OpSyncLocal>({result})
        ->eval();

    results.push_back(result);
  }

  // Verify results
  std::vector<float> C_cpu(M * N);
  cpu_matmul_ikj(A, B, C_cpu, M, K, N);
  auto C_cpu_col = transposeMatrix(C_cpu, M, N);

  // Print verification results
  for (size_t i = 0; i < results.size(); i++) {
    const auto& expected =
        shaderConfigs[i].needs_column_major_A ? C_cpu_col : C_cpu;
    std::cout << std::format(
        "{:<15} SGEMM {}\n", shaderConfigs[i].name,
        verifyResults(expected, results[i]->vector(), 1e-3f)
            ? "matches CPU result"
            : "does not match CPU result");
  }
}
