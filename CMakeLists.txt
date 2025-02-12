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

# Function to compile shader and create interface library
function(add_shader_library SHADER_NAME SHADER_SOURCE NAMESPACE)
    vulkan_compile_shader(
        INFILE ${SHADER_SOURCE}
        OUTFILE shader/${SHADER_NAME}.hpp
        NAMESPACE ${NAMESPACE}
    )

    add_library(${SHADER_NAME} INTERFACE "shader/${SHADER_NAME}.hpp")
    target_include_directories(${SHADER_NAME} INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
endfunction()

# Lists of shaders
set(BASIC_SHADERS
    dist_scale
    transpose
)

set(SGEMM_SHADERS
    sgemm_naive_v1
    sgemm_naive_v2
    sgemm_naive_col
    sgemm_tiled_v1
    sgemm_tiled_v2
    sgemm_tiled_v3
    sgemm_tiled_v4
    sgemm_tiled_col
    sgemm_regblock_col
)

foreach(shader IN LISTS BASIC_SHADERS)
    add_shader_library(${shader} "shader/${shader}.comp" "shader")
endforeach()

# Add SGEMM shaders
foreach(shader IN LISTS SGEMM_SHADERS)
    add_shader_library(${shader} "shader/${shader}.comp" "sgemm_shader")
endforeach()

# Executables
# kompute
add_executable(kompute_dist_scale_embed
    src/kompute/dist_scale_embed.cpp)
target_link_libraries(kompute_dist_scale_embed
    PRIVATE
    dist_scale
    kompute::kompute
)

add_executable(kompute_dist_scale_op
    src/kompute/dist_scale_op.cpp)
target_link_libraries(kompute_dist_scale_op
    PRIVATE
    dist_scale
    kompute::kompute
)

add_executable(sgemm_algo_verify
    src/kompute/sgemm_algo_verification.cpp)
target_link_libraries(sgemm_algo_verify
    PRIVATE
    sgemm_naive_v1
    sgemm_naive_v2
    sgemm_naive_col
    sgemm_tiled_v1
    sgemm_tiled_v2
    sgemm_tiled_v3
    sgemm_tiled_v4
    sgemm_tiled_col
    sgemm_regblock_col
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
    sgemm_naive_v1
    sgemm_naive_v2
    sgemm_naive_col
    sgemm_tiled_v1
    sgemm_tiled_v2
    sgemm_tiled_v3
    sgemm_tiled_v4
    sgemm_tiled_col
    sgemm_regblock_col
    benchmark::benchmark
    CUDA::cudart 
    CUDA::cublas
    kompute::kompute
)
