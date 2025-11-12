#include <cuda_runtime.h>

#include <iostream>

#include "matmul_smem_block.h"

namespace {
constexpr int kBlockDim = 32;

__global__ void SgemmSmemBlock(int M, int K, int N, float alpha, const float *A,
                               const float *B, float beta, float *C) {
  // 타겟 Block의 row, col
  int block_row = blockIdx.x;
  int block_col = blockIdx.y;

  // Block Tiling 연산을 위한 공유 메모리 할당
  __shared__ float As[kBlockDim][kBlockDim];
  __shared__ float Bs[kBlockDim][kBlockDim];

  // Block내에서의 현재 스레드의 row, col
  int thread_row = threadIdx.x / kBlockDim;
  int thread_col = threadIdx.x % kBlockDim;

  // 실제 타겟 위치 계산
  A += block_row * kBlockDim * K;
  B += block_col * kBlockDim;
  C += block_row * kBlockDim * N + block_col * kBlockDim;

  float value = 0.0f;
  // K dimension을 kBlockDim 크기만큼 나누어 처리
  for (int k_off = 0; k_off < K; k_off += kBlockDim) {
    As[thread_row][thread_col] = 0.0f;
    Bs[thread_row][thread_col] = 0.0f;

    // 각 스레드는 협력적으로 데이터를 공유 메모리로 로드
    if (k_off + thread_col < K && block_row * kBlockDim + thread_row < M)
      As[thread_row][thread_col] = A[thread_row * K + thread_col];

    if (k_off + thread_row < K && block_col * kBlockDim + thread_col < N)
      Bs[thread_row][thread_col] = B[thread_row * N + thread_col];

    __syncthreads();  // 모든 스레드가 데이터 로드를 마칠 때까지 대기

    // 공유 메모리를 이용한 행렬 곱셈
    for (int k = 0; k < kBlockDim; ++k) {
      value += As[thread_row][k] * Bs[k][thread_col];
    }

    A += kBlockDim;  // 다음 블록으로 이동
    B += kBlockDim * N;

    __syncthreads();  // 다음 블록을 위해 동기화
  }

  // 결과 저장
  C[thread_row * N + thread_col] =
      alpha * value + beta * C[thread_row * N + thread_col];
}
}  // namespace

namespace jpyo0803 {

MatmulSmemBlock::~MatmulSmemBlock() {
  // Deallocate GPU memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}

void MatmulSmemBlock::ComputeCore(int M, int K, int N) {
  float alpha = 1.0f;
  float beta = 0.0f;

  dim3 grid_dim(CeilDiv(M, kBlockDim), CeilDiv(N, kBlockDim),
                1);                       // grid의block 수
  dim3 block_dim(kBlockDim * kBlockDim);  // 한 block에서 스레드 수

  // SMEM과 L1 캐시 공간은 기본적으로 같이 사용됨. 하지만 SMEM 사용량이 많을경우
  // L1 캐시 공간을 줄여서 SMEM 공간을 늘릴 수 있음. 아래 설정은 SMEM 공간을
  // 최대한으로 늘리는 설정임.
  cudaFuncSetAttribute(SgemmSmemBlock,
                       cudaFuncAttributePreferredSharedMemoryCarveout,
                       cudaSharedmemCarveoutMaxShared);

  SgemmSmemBlock<<<grid_dim, block_dim>>>(M, K, N, alpha, d_A, d_B, beta, d_C);
}

}  // namespace jpyo0803
