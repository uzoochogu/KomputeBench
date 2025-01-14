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
)
FetchContent_MakeAvailable(kompute)

# shader compilation
vulkan_compile_shader(
    INFILE shader/dist_scale.comp
    OUTFILE shader/dist_scale.hpp
    NAMESPACE "shader")

add_library(shader INTERFACE "shader/dist_scale.hpp")
target_include_directories(shader INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)

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

# cuda
add_executable(cuda_dist_scale src/cuda/distance_scale.cu)
target_link_libraries(cuda_dist_scale PRIVATE CUDA::cudart)

add_executable(cuda_sgemm_hgemm src/cuda/hgemm_sgemm.cu)
target_link_libraries(cuda_sgemm_hgemm PRIVATE CUDA::cudart CUDA::cublas )
