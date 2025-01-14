#include <format>
#include <iostream>
#include <kompute/Kompute.hpp>
#include <memory>
#include <shader/dist_scale.hpp>
#include <vector>

int main() {
  // Initialize manager
  kp::Manager mgr;

  const int N = 64;        // Total number of elements
  const float ref = 0.5f;  // Reference value

  // Create output tensor initialized with zeros
  std::vector<float> output(N, 0.0f);
  auto tensorOut = mgr.tensor(output);  // std::shared_ptr<kp::TensorT<float>>

  // Parameters for the algorithm
  std::vector<std::shared_ptr<kp::Memory>> params = {tensorOut};

  const std::vector<uint32_t> shader = std::vector<uint32_t>(
      shader::DIST_SCALE_COMP_SPV.begin(), shader::DIST_SCALE_COMP_SPV.end());

  // Create algorithm with the shader
  std::shared_ptr<kp::Algorithm> algo =
      mgr.algorithm(params, shader, kp::Workgroup({32, 1, 1}), {}, {ref, N});

  // Record and run sequence
  mgr.sequence()
      ->record<kp::OpSyncDevice>({tensorOut})
      ->record<kp::OpAlgoDispatch>(algo)  // , N, 1, 1)  // Dispatch N threads
      ->record<kp::OpSyncLocal>({tensorOut})
      ->eval();

  // Print results
  for (uint32_t i{0}; const auto& result : tensorOut->vector()) {
    std::cout << std::format("i = {:>2}: dist from {} to Scale is {:<8}\n", i,
                             ref, result);
    i++;
  }
}
