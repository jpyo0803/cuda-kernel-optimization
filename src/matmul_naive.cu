#include <cuda_runtime.h>

#include <iostream>

#include "matmul_naive.h"

namespace {
constexpr int kBlockDim = 32;

__global__ void SgemmNaive(int M, int K, int N, float alpha, const float *A,
                           const float *B, float beta, float *C) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;

  /*
     입력 행렬 A의 dimension: (M, K)
     입력 행렬 B의 dimension: (K, N)
     출력 행렬 C의 dimension: (M, N)

     각 스레드는 출력 행렬 C의 (row, col) 원소 계산
  */
  if (row < M && col < N) {  // 범위 체크 (실제 계산에 필요한 스레드수보다 더
                             // 많은 스레드가 실행될 수 있으므로)
    float value = 0.0f;
    for (int k = 0; k < K; ++k) {
      value += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = alpha * value + beta * C[row * N + col];
  }
}
}  // namespace

namespace jpyo0803 {

MatmulNaive::~MatmulNaive() {
  // Deallocate GPU memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}

void MatmulNaive::ComputeCore(int M, int K, int N) {
  float alpha = 1.0f;
  float beta = 0.0f;

  dim3 grid_dim(CeilDiv(M, kBlockDim), CeilDiv(N, kBlockDim),
                1);                         // grid의 block 수
  dim3 block_dim(kBlockDim, kBlockDim, 1);  // 한 block에서 스레드 수

  SgemmNaive<<<grid_dim, block_dim>>>(M, K, N, alpha, d_A, d_B, beta, d_C);
}

}  // namespace jpyo0803
