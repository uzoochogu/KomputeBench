#include <iostream>
#include <memory>
#include <vector>
#include <fstream>
#include <format>


#include <kompute/Kompute.hpp>
#include <shader/dist_scale.hpp>

int main() {
    // Initialize manager
    kp::Manager mgr;

    const int N = 64;  // Total number of elements
    const float ref = 0.5f;  // Reference value

    // Create output tensor initialized with zeros
    std::vector<float> output(N, 0.0f);
    auto tensorOut = mgr.tensor(output); // std::shared_ptr<kp::TensorT<float>>

    // Parameters for the algorithm
    std::vector<std::shared_ptr<kp::Tensor>> params = {tensorOut};

    const std::vector<uint32_t> shader = std::vector<uint32_t>(
      shader::DIST_SCALE_COMP_SPV.begin(), shader::DIST_SCALE_COMP_SPV.end());

    // Create algorithm with the shader
    std::shared_ptr<kp::Algorithm> algo = mgr.algorithm(params, shader, kp::Workgroup({32, 1, 1}), {}, {ref, N});

    // Record and run sequence
    mgr.sequence()
        ->record<kp::OpTensorSyncDevice>({tensorOut})
        ->record<kp::OpAlgoDispatch>(algo) // , N, 1, 1)  // Dispatch N threads
        ->record<kp::OpTensorSyncLocal>({tensorOut})
        ->eval();

    // Print results
    for (int i = 0; i < N; i++) {
        float result = tensorOut->data()[i];
        std::cout << std::format("i = {:>2}: dist from {} to Scale is {:<8}\n",
        i, ref, result);
    }
}
