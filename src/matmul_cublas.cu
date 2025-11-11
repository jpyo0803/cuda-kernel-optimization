#include <cuda_runtime.h>

#include <iostream>

#include "matmul_cublas.h"

namespace jpyo0803 {

MatmulCublas::MatmulCublas() {
  cublasStatus_t stat = cublasCreate(&handle_);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "[MatmulCublas] Failed to create cuBLAS handle.\n";
  }
}

MatmulCublas::~MatmulCublas() {
  // Destroy handle
  cublasDestroy(handle_);

  // Deallocate GPU memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}

void MatmulCublas::ComputeCore(int M, int K, int N) {
  float alpha = 1.0f;
  float beta = 0.0f;

  // cuBLAS는 column-major 기반.
  // row-major 데이터를 그대로 쓰려면 A와 B의 순서를 바꿔서 호출해야 함.
  cublasStatus_t stat =
      cublasSgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N,  // transpose 없음
                  N, M, K,         // cuBLAS는 (N, M, K) 순서로 받음
                  &alpha, d_B, N,  // B의 leading dimension = N
                  d_A, K,          // A의 leading dimension = K
                  &beta, d_C, N);  // 결과 C의 leading dimension = N

  if (stat != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "[MatmulCublas] cuBLAS SGEMM failed.\n";
  }
}

}  // namespace jpyo0803
