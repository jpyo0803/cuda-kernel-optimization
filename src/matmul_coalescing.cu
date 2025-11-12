#include <cuda_runtime.h>

#include <iostream>

#include "matmul_coalescing.h"

namespace {
constexpr int kBlockDim = 32;

__global__ void SgemmCoalescing(int M, int K, int N, float alpha,
                                const float *A, const float *B, float beta,
                                float *C) {
  /*
     같은 block 내에서 row는 고정, col은 변화하도록 하여
     같은 warp내의 thread들이 연속적인 메모리 접근을 하도록 개선

     실질적으로는 naive 버전의 x, y 좌표를 바꾸는 것과 동일
  */
  int row = blockIdx.x * kBlockDim + (threadIdx.x / kBlockDim);
  int col = blockIdx.y * kBlockDim + (threadIdx.x % kBlockDim);

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

MatmulCoalescing::~MatmulCoalescing() {
  // Deallocate GPU memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}

void MatmulCoalescing::ComputeCore(int M, int K, int N) {
  float alpha = 1.0f;
  float beta = 0.0f;

  dim3 grid_dim(CeilDiv(M, kBlockDim), CeilDiv(N, kBlockDim),
                1);                       // grid의 block 수
  dim3 block_dim(kBlockDim * kBlockDim);  // 한 block에서 스레드 수

  SgemmCoalescing<<<grid_dim, block_dim>>>(M, K, N, alpha, d_A, d_B, beta, d_C);
}

}  // namespace jpyo0803
