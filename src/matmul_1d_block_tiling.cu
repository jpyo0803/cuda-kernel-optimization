#include <cuda_runtime.h>

#include <iostream>

#include "matmul_1d_block_tiling.h"

namespace {
constexpr int kBM = 64;
constexpr int kBN = 64;
constexpr int kBK = 8;
constexpr int kTM = 8;  // 한 쓰레드가 처리하는 결과 cell의 수

__global__ void Sgemm1DBlockTiling(int M, int K, int N, float alpha,
                                   const float *A, const float *B, float beta,
                                   float *C) {
  int br = blockIdx.y;  // block row
  int bc = blockIdx.x;  // block col

  int tr = threadIdx.x / kBN;  // thread row
  int tc = threadIdx.x % kBN;  // thread col

  __shared__ float As[kBM][kBK];
  __shared__ float Bs[kBK][kBN];

  A += br * kBM * K;
  B += bc * kBN;
  C += br * kBM * N + bc * kBN;

  int ira = threadIdx.x / kBK;  // inner row of A
  int ica = threadIdx.x % kBK;  // inner col of A
  int irb = threadIdx.x / kBN;  // inner row of B
  int icb = threadIdx.x % kBN;  // inner col of B

  // 각 스레드가 계산할 결과값 저장 공간
  float values[kTM] = {0.0f};

  for (int bk_off = 0; bk_off < K; bk_off += kBK) {
    As[ira][ica] = 0.0f;
    Bs[irb][icb] = 0.0f;

    if (bk_off + ica < K && br * kBM + ira < M) As[ira][ica] = A[ira * K + ica];
    if (bk_off + irb < K && bc * kBN + icb < N) Bs[irb][icb] = B[irb * N + icb];

    // 다른 쓰레드도 데이터 로드를 마칠 때까지 대기
    __syncthreads();

    for (int k = 0; k < kBK; ++k) {
      float b_value = Bs[k][tc];
      for (int tm = 0; tm < kTM; ++tm) {
        values[tm] += As[tr * kTM + tm][k] * b_value;
      }
    }

    A += kBK;
    B += kBK * N;

    __syncthreads();
  }

  for (int i = 0; i < kTM; ++i) {
    C[(tr * kTM + i) * N + tc] =
        alpha * values[i] + beta * C[(tr * kTM + i) * N + tc];
  }
}
}  // namespace

namespace jpyo0803 {

Matmul1DBlockTiling::~Matmul1DBlockTiling() {
  // Deallocate GPU memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}

void Matmul1DBlockTiling::ComputeCore(int M, int K, int N) {
  float alpha = 1.0f;
  float beta = 0.0f;

  // M, N 위치를 바꿔 L2 캐시 효율 극대화
  dim3 grid_dim(CeilDiv(N, kBN), CeilDiv(M, kBM),
                1);                   // grid의block 수
  dim3 block_dim((kBM * kBN) / kTM);  // block당 필요 스레드 수

  // SMEM과 L1 캐시 공간은 기본적으로 같이 사용됨. 하지만 SMEM 사용량이 많을경우
  // L1 캐시 공간을 줄여서 SMEM 공간을 늘릴 수 있음. 아래 설정은 SMEM 공간을
  // 최대한으로 늘리는 설정임.
  cudaFuncSetAttribute(Sgemm1DBlockTiling,
                       cudaFuncAttributePreferredSharedMemoryCarveout,
                       cudaSharedmemCarveoutMaxShared);

  Sgemm1DBlockTiling<<<grid_dim, block_dim>>>(M, K, N, alpha, d_A, d_B, beta,
                                              d_C);
}

}  // namespace jpyo0803
