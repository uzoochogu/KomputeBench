cmake_minimum_required(VERSION 3.28)

include(./.env.cmake OPTIONAL RESULT_VARIABLE LOCAL_ENV)
message(STATUS ${LOCAL_ENV})

project(KomputeBench LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(FETCHCONTENT_BASE_DIR "${PROJECT_SOURCE_DIR}/external")

include(FetchContent)
FetchContent_Declare(
    kompute
    GIT_REPOSITORY https://github.com/KomputeProject/kompute.git
    GIT_TAG        v0.9.0
)
FetchContent_MakeAvailable(kompute)

# Find Vulkan which is required for Kompute
find_package(Vulkan REQUIRED)


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
