#ifndef DATA_HPP
#define DATA_HPP

#include <array>

// Data for sample programs and verification

// Dimensions
constexpr uint32_t M = 3;
constexpr uint32_t K = 4;
constexpr uint32_t N = 2;

// Input matrices
constexpr std::array<float, M * K> A = {1.0f, 2.0f, 3.0f, 4.0f,  5.0f,  6.0f,
                                        7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};

constexpr std::array<float, K * N> B = {1.0f, 2.0f, 3.0f, 4.0f,
                                        5.0f, 6.0f, 7.0f, 8.0f};

// row major A =
// 1.0 2.0 3.0 4.0
// 5.0 6.0 7.0 8.0

// col major A =
// 1.0 5.0
// 2.0 6.0
// 3.0 7.0
// 4.0 8.0

// memory layout (row major)
// 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0

// memory layout (col major)
// 1.0 5.0 2.0 6.0 3.0 7.0 4.0 8.0

#endif  // DATA_HPP
