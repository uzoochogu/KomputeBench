#ifndef UTILS_HPP
#define UTILS_HPP

#include <format>
#include <iostream>
#include <random>
#include <span>
#include <vector>

// Initialize matrix with random values
void initializeMatrix(std::span<float> matrix) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-0.5, 0.5);

  for (float& element : matrix) {
    element = static_cast<float>(dis(gen));
  }
}

bool verifyResults(const std::span<const float> expected,
                   const std::span<const float> actual,
                   float tolerance = 1e-2) {
  if (expected.size() != actual.size()) {
    return false;
  }

  for (size_t i = 0; i < expected.size(); ++i) {
    float rel_error = std::abs(expected[i] - actual[i]);
    if (rel_error > tolerance) {
      std::cout << std::format(
          "Mismatch at index {}: expected {}, got {}, relative error: {}\n", i,
          expected[i], actual[i], rel_error);
      return false;
    }
  }

  return true;
}

bool compareResults(const std::span<const float> result1,
                    const std::span<const float> result2, float tolerance) {
  if (result1.size() != result2.size()) {
    return false;
  }

  for (size_t i = 0; i < result1.size(); ++i) {
    float diff = std::abs(result1[i] - result2[i]);
    float max_val = std::max(std::abs(result1[i]), std::abs(result2[i]));
    if (diff / max_val > tolerance) {
      std::cout << std::format(
          "Results do not match at index {}\n"
          "Matrix 1: {}, Matrix 2: {}\n"
          "Relative difference: {}\n",
          i, result1[i], result2[i], diff / max_val);
      return false;
    }
  }

  return true;
}

void cpu_matmul(const std::span<const float> A, const std::span<const float> B,
                std::span<float> C, int M, int K, int N) {
  // bounds check
  if (A.size() != M * K || B.size() != K * N || C.size() != M * N) {
    std::cerr << "Error: Invalid matrix dimensions\n";
    return;
  }

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      float sum = 0.0f;
      for (int k = 0; k < K; k++) {
        sum += A[i * K + k] * B[k * N + j];
      }
      C[i * N + j] = sum;
    }
  }
}

// cpu_matmul using ikj loop order
void cpu_matmul_ikj(const std::span<const float> A,
                    const std::span<const float> B, std::span<float> C, int M,
                    int K, int N) {
  if (A.size() != M * K || B.size() != K * N || C.size() != M * N) {
    std::cerr << "Error: Invalid matrix dimensions\n";
    return;
  }
  for (int i = 0; i < M; i++) {
    for (int k = 0; k < K; k++) {
      for (int j = 0; j < N; j++) {
        C[i * N + j] += A[i * K + k] * B[k * N + j];
      }
    }
  }
}

#endif  // UTILS_HPP
