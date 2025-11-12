cmake_minimum_required(VERSION 3.18)
project(cuda_matmul_lib LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(cuda_matmul_lib STATIC
    src/matmul_base.cu
    src/matmul_cublas.cu
    src/matmul_naive.cu
    src/matmul_coalescing.cu
    src/matmul_smem_block.cu
    src/matmul_1d_block_tiling.cu
)

# Add project header directory
target_include_directories(cuda_matmul_lib PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

# Link CUDA runtime + cuBLAS 
target_link_libraries(cuda_matmul_lib PUBLIC cudart cublas)

set_target_properties(cuda_matmul_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

add_executable(cuda_matmul_demo main.cu)

target_link_libraries(cuda_matmul_demo PRIVATE cuda_matmul_lib)