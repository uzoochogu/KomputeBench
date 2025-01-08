#include <iostream>
#include <format>

#include <kompute/Kompute.hpp>
#include <shader/dist_scale.hpp>

class OpMyCustom : public kp::OpAlgoDispatch
{
  public:
    OpMyCustom(std::vector<std::shared_ptr<kp::Memory>> tensors,
         std::shared_ptr<kp::Algorithm> algorithm)
      : kp::OpAlgoDispatch(algorithm)
    {
         if (tensors.size() != 1) {
             throw std::runtime_error("Kompute OpMult expected 1 tensors but got " + tensors.size());
         }

    const std::vector<uint32_t> shader = std::vector<uint32_t>(
      shader::DIST_SCALE_COMP_SPV.begin(), shader::DIST_SCALE_COMP_SPV.end());
         algorithm->rebuild(tensors, shader);
    }
};


int main() {
    // Initialize manager
    kp::Manager mgr; // Automatically selects Device 0

    const int N = 64;  // Total number of elements
    const float ref = 0.5f;  // Reference value

    // Create output tensor initialized with zeros
    std::vector<float> output(N, 0.0f);
    auto tensorOut = mgr.tensor(output);

    // Record and run sequence
    mgr.sequence()
        ->record<kp::OpSyncDevice>({tensorOut})
        ->record<OpMyCustom>({tensorOut}, mgr.algorithm())
        ->record<kp::OpSyncLocal>({tensorOut})
        ->eval();


    // Print results
    for (int i = 0; i < N; i++) {
        float result = tensorOut->data()[i];
        std::cout << std::format("i = {:>2}: dist from {} to Scale is {:<8}\n",
        i, ref, result);
    }
}