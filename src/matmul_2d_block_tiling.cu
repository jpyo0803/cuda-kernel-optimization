#include <cuda_runtime.h>

#include <iostream>

#include "matmul_2d_block_tiling.h"

namespace {
constexpr int kBM = 128;
constexpr int kBN = 128;
constexpr int kBK = 8;
constexpr int kTM = 8;  // kTM * kTN은 한 쓰레드가 처리하는 결과 cell의 수
constexpr int kTN = 8;  //

__global__ void Sgemm2DBlockTiling(int M, int K, int N, float alpha,
                                   const float *A, const float *B, float beta,
                                   float *C) {
  int br = blockIdx.y;  // block row
  int bc = blockIdx.x;  // block col

  // 결과 블록 당 필요한 스레드 수 계산
  int num_threads_per_block = (kBM * kBN) / (kTM * kTN);

  int tr = threadIdx.x / (kBN / kTN);  // thread row
  int tc = threadIdx.x % (kBN / kTN);  // thread col

  __shared__ float As[kBM][kBK];
  __shared__ float Bs[kBK][kBN];

  A += br * kBM * K;
  B += bc * kBN;
  C += br * kBM * N + bc * kBN;

  int ira = threadIdx.x / kBK;  // inner row of A
  int ica = threadIdx.x % kBK;  // inner col of A
  // 각 쓰레드가 A 입력 데이터 로드할 때의 간격. 각 Thread는 kBK 개수의 데이터를 로드
  int stride_a = num_threads_per_block / kBK;

  int irb = threadIdx.x / kBN;  // inner row of B
  int icb = threadIdx.x % kBN;  // inner col of B
  // 각 쓰레드가 B 입력 데이터 로드할 때의 간격. 각 Thread는 kBN 개수의 데이터를 로드
  int stride_b = num_threads_per_block / kBN;

  // 각 스레드가 계산 결과 저장을 위한 2D 메모리 공간
  float values[kTM * kTN] = {0.0f};

  // register 캐시 공간
  float reg_m[kTM] = {0.0f};
  float reg_n[kTN] = {0.0f};

  for (int bk_off = 0; bk_off < K; bk_off += kBK) {
    // 각 쓰레드는 stride만큼 행을 건너뛰면서 A와 B 데이터를 SMEM으로 로드
    for (int offset = 0; offset < kBM; offset += stride_a) {
      As[ira + offset][ica] = 0.0f;
      if (bk_off + ica < K && br * kBM + ira + offset < M)
        As[ira + offset][ica] = A[(ira + offset) * K + ica];
    }

    for (int offset = 0; offset < kBK; offset += stride_b) {
      Bs[irb + offset][icb] = 0.0f;
      if (bk_off + irb + offset < K && bc * kBN + icb < N)
        Bs[irb + offset][icb] = B[(irb + offset) * N + icb];
    }
    // 다른 쓰레드도 데이터 로드를 마칠 때까지 대기
    __syncthreads();

    for (int k = 0; k < kBK; ++k) {
      // 한번 register cache로 가져와서 최대한 재사용해 arithmetic intensity
      // 극대화
      for (int i = 0; i < kTM; ++i) {
        reg_m[i] = As[tr * kTM + i][k];
      }
      for (int j = 0; j < kTN; ++j) {
        reg_n[j] = Bs[k][tc * kTN + j];
      }

      // Outer product 방식으로 계산
      for (int i = 0; i < kTM; ++i) {
        for (int j = 0; j < kTN; ++j) {
          values[i * kTN + j] += reg_m[i] * reg_n[j];
        }
      }
    }

    A += kBK;
    B += kBK * N;

    // 다음 블록을 위해 동기화
    __syncthreads();
  }

  // 계산된 결과를 Global Memory에 저장
  for (int i = 0; i < kTM; ++i) {
    for (int j = 0; j < kTN; ++j) {
      C[(tr * kTM + i) * N + (tc * kTN + j)] =
          alpha * values[i * kTN + j] +
          beta * C[(tr * kTM + i) * N + (tc * kTN + j)];
    }
  }
}
}  // namespace

namespace jpyo0803 {

Matmul2DBlockTiling::~Matmul2DBlockTiling() {
  // Deallocate GPU memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}

void Matmul2DBlockTiling::ComputeCore(int M, int K, int N) {
  float alpha = 1.0f;
  float beta = 0.0f;

  if (M < 128 || N < 128) {
    std::cerr << "Error: M and N must be at least 128 for 2D Block Tiling for "
                 "the current implementation."
              << std::endl;
    return;
  }

  // M, N 위치를 바꿔 L2 캐시 효율 극대화
  dim3 grid_dim(CeilDiv(N, kBN), CeilDiv(M, kBM),
                1);                           // grid의block 수
  dim3 block_dim((kBM * kBN) / (kTM * kTN));  // block당 필요 스레드 수

  // SMEM과 L1 캐시 공간은 기본적으로 같이 사용됨. 하지만 SMEM 사용량이 많을경우
  // L1 캐시 공간을 줄여서 SMEM 공간을 늘릴 수 있음. 아래 설정은 SMEM 공간을
  // 최대한으로 늘리는 설정임.
  cudaFuncSetAttribute(Sgemm2DBlockTiling,
                       cudaFuncAttributePreferredSharedMemoryCarveout,
                       cudaSharedmemCarveoutMaxShared);

  Sgemm2DBlockTiling<<<grid_dim, block_dim>>>(M, K, N, alpha, d_A, d_B, beta,
                                              d_C);
}

}  // namespace jpyo0803
