cmake_minimum_required(VERSION 3.28)

include(./.env.cmake OPTIONAL RESULT_VARIABLE LOCAL_ENV)
message(STATUS "ENV file: ${LOCAL_ENV}")

project(KomputeBench LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA flags
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES native)
message(STATUS "CUDA Arch: ${CMAKE_CUDA_ARCHITECTURES}")

find_package(Vulkan REQUIRED)
find_package(CUDAToolkit REQUIRED)

set(FETCHCONTENT_BASE_DIR "${PROJECT_SOURCE_DIR}/external")

include(FetchContent)
FetchContent_Declare(
    kompute
    GIT_REPOSITORY https://github.com/KomputeProject/kompute.git
    GIT_TAG master
)

FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG main
)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
)

FetchContent_MakeAvailable(
    kompute
    googletest
    googlebenchmark
)

# shaders compilation
vulkan_compile_shader(
    INFILE shader/dist_scale.comp
    OUTFILE shader/dist_scale.hpp
    NAMESPACE "shader")

add_library(shader INTERFACE "shader/dist_scale.hpp")
target_include_directories(shader INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)

vulkan_compile_shader(
    INFILE shader/sgemm_naive.comp
    OUTFILE shader/sgemm_naive.hpp
    NAMESPACE "sgemm_shader")

add_library(sgemm_naive INTERFACE "shader/sgemm_naive.hpp")
target_include_directories(sgemm_naive INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)

vulkan_compile_shader(
    INFILE shader/sgemm_tiled.comp
    OUTFILE shader/sgemm_tiled.hpp
    NAMESPACE "sgemm_shader")

add_library(sgemm_tiled INTERFACE "shader/sgemm_tiled.hpp")
target_include_directories(sgemm_tiled INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)

# Executables
# kompute
add_executable(kompute_dist_scale_embed
    src/kompute/dist_scale_embed.cpp)
target_link_libraries(kompute_dist_scale_embed
    PRIVATE
    shader
    kompute::kompute
)

add_executable(kompute_dist_scale_op
    src/kompute/dist_scale_op.cpp)
target_link_libraries(kompute_dist_scale_op
    PRIVATE
    shader
    kompute::kompute
)

add_executable(sgemm_naive_kompute
    src/kompute/sgemm_naive.cpp)
target_link_libraries(sgemm_naive_kompute
    PRIVATE
    sgemm_naive
    kompute::kompute
)

add_executable(sgemm_tiled_kompute
    src/kompute/sgemm_tiled.cpp)
target_link_libraries(sgemm_tiled_kompute
    PRIVATE
    sgemm_tiled
    kompute::kompute
)

# cuda
add_executable(cuda_dist_scale src/cuda/distance_scale.cu)
target_link_libraries(cuda_dist_scale PRIVATE CUDA::cudart)

add_executable(cuda_sgemm_hgemm src/cuda/hgemm_sgemm.cu)
target_link_libraries(cuda_sgemm_hgemm PRIVATE CUDA::cudart CUDA::cublas )

# Benchmark
add_executable(sgemm_benchmark src/benchmark/sgemm_benchmark.cpp)
target_link_libraries(
    sgemm_benchmark 
    PRIVATE
    sgemm_naive
    sgemm_tiled
    benchmark::benchmark
    CUDA::cudart 
    CUDA::cublas
    kompute::kompute
)
