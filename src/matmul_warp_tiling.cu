#include <cuda_runtime.h>

#include <iostream>

#include "matmul_warp_tiling.h"

namespace {
constexpr int kNumThreads = 128;
constexpr int kNumThreadsPerWarp = 32;
constexpr int kNumWarps = kNumThreads / kNumThreadsPerWarp;

constexpr int kBM = 128;
constexpr int kBN = 128;
constexpr int kBK = 16;

constexpr int kWM = 64;
constexpr int kWN = 64;
constexpr int kWNITER = 4;

constexpr int kTM = 8;  // kTM * kTN은 한 쓰레드가 처리하는 결과 cell의 수
constexpr int kTN = 4;

constexpr int kWMITER = (kWM * kWN) / (kNumThreadsPerWarp * kTM * kTN * kWNITER);  // warp당 WM 크기 처리 횟수
constexpr int kWSUBM = kWM / kWMITER;
constexpr int kWSUBN = kWN / kWNITER;

__global__ void SgemmWarpTiling(int M, int K, int N, float alpha,
                                   const float *A, const float *B, float beta,
                                   float *C) {
  int br = blockIdx.y;  // block row
  int bc = blockIdx.x;  // block col

  int warp_idx = threadIdx.x / kNumThreadsPerWarp;
  int wr = warp_idx / (kBN / kWN); // warp row
  int wc = warp_idx % (kBN / kWN); // warp col

  int thread_idx_in_warp = threadIdx.x % kNumThreadsPerWarp;
  int tr = thread_idx_in_warp / (kWSUBN / kTN); // thread row
  int tc = thread_idx_in_warp % (kWSUBN / kTN); // thread col

  __shared__ float As[kBM * kBK];
  __shared__ float Bs[kBK * kBN];

  A += br * kBM * K;
  B += bc * kBN;
  C += (br * kBM + wr * kWM) * N + bc * kBN + wc * kWN;

  int ira = threadIdx.x / (kBK / 4);  // inner row of A
  int ica = threadIdx.x % (kBK / 4);  // inner col of A
  int row_stride_a = (kNumThreads * 4) / kBK;

  int irb = threadIdx.x / (kBN / 4);  // inner row of B
  int icb = threadIdx.x % (kBN / 4);  // inner col of B
  int row_stride_b = kNumThreads / (kBN / 4);

  float values[kWMITER * kTM * kWNITER * kTN] = {0.0f};

  float reg_m[kWMITER * kTM] = {0.0f};
  float reg_n[kWNITER * kTN] = {0.0f};

  for (int bk_off = 0; bk_off < K; bk_off += kBK) {
    for (int offset = 0; offset < kBM; offset += row_stride_a) {
      float4 tmp = reinterpret_cast<const float4 *>(
          &A[(ira + offset) * K + ica * 4])[0];
      As[ira + offset + kBM * ica * 4] = tmp.x;
      As[ira + offset + kBM * (ica * 4 + 1)] = tmp.y;
      As[ira + offset + kBM * (ica * 4 + 2)] = tmp.z;
      As[ira + offset + kBM * (ica * 4 + 3)] = tmp.w;
    }

    for (int offset = 0; offset < kBK; offset += row_stride_b) {
      reinterpret_cast<float4 *>(
          &Bs[(irb + offset) * kBN + icb * 4])[0] =
          reinterpret_cast<const float4 *>(&B[(irb + offset) * N + icb * 4])[0];
    }

    __syncthreads();

    for (int k = 0; k < kBK; ++k) {
      for (int wsubrow = 0; wsubrow < kWMITER; ++wsubrow) {
        for (int i = 0; i < kTM; ++i) {
          reg_m[wsubrow * kTM + i] =
              As[k * kBM + wr * kWM + wsubrow * kWSUBM + tr * kTM + i];
        }
      }
      for (int wsubcol = 0; wsubcol < kWNITER; ++wsubcol) {
        for (int j = 0; j < kTN; ++j) {
          reg_n[wsubcol * kTN + j] =
              Bs[k * kBN + wc * kWN + wsubcol * kWSUBN + tc * kTN + j];
        }
      }

      for (int wsubrow = 0; wsubrow < kWMITER; ++wsubrow) {
        for (int wsubcol = 0; wsubcol < kWNITER; ++wsubcol) {
          for (int i = 0; i < kTM; ++i) {
            for (int j = 0; j < kTN; ++j) {
              values[(wsubrow * kTM + i) * (kWNITER * kTN) + wsubcol * kTN + j] +=
                  reg_m[wsubrow * kTM + i] * reg_n[wsubcol * kTN + j];
            }
          }
        }
      }
    }

    A += kBK;
    B += kBK * N;

    __syncthreads();
  }

  for (int wsubrow = 0; wsubrow < kWMITER; ++wsubrow) {
    for (int wsubcol = 0; wsubcol < kWNITER; ++wsubcol) {
      float* C_inter = &C[(wsubrow * kWSUBM) * N + wsubcol * kWSUBN];
      for (int i = 0; i < kTM; ++i) {
        for (int j = 0; j < kTN; j += 4) {
          float4 tmp = reinterpret_cast<float4 *>(&C_inter[(tr * kTM + i) * N + tc * kTN + j])[0];

          int idx = (wsubrow * kTM + i) * (kWNITER * kTN) + wsubcol * kTN + j;
          tmp.x = alpha * values[idx] + beta * tmp.x;
          tmp.y = alpha * values[idx + 1] + beta * tmp.y;
          tmp.z = alpha * values[idx + 2] + beta * tmp.z;
          tmp.w = alpha * values[idx + 3] + beta * tmp.w;
          reinterpret_cast<float4 *>(&C_inter[(tr * kTM + i) * N + tc * kTN + j])[0] = tmp;
        }
      }
    }
  }
}
}  // namespace

namespace jpyo0803 {

MatmulWarpTiling::~MatmulWarpTiling() {
  // Deallocate GPU memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}

void MatmulWarpTiling::ComputeCore(int M, int K, int N) {
  float alpha = 1.0f;
  float beta = 0.0f;

  static_assert((kBM % kWM) == 0);
  static_assert((kBN % kWN) == 0);
  static_assert((kBN / kWN) * (kBM / kWM) >= kNumWarps);
  
  static_assert((kWM % kWN) % (32 * kTM * kTN * kWNITER) == 0);

  static_assert((kWM % kWMITER) == 0);
  static_assert((kWN % kWNITER) == 0);

  static_assert((kNumThreads * 4) % kBK == 0);
  static_assert((kNumThreads * 4) % kBN == 0);
  static_assert(kBN % (16 * kTN) == 0);
  static_assert(kBM % (16 * kTM) == 0);
  static_assert((kBM * kBK) % (kNumThreads * 4) == 0);
  static_assert((kBN * kBK) % (kNumThreads * 4) == 0);

  // M, N 위치를 바꿔 L2 캐시 효율 극대화
  dim3 grid_dim(CeilDiv(N, kBN), CeilDiv(M, kBM),
                1);                           // grid의block 수
  dim3 block_dim(kNumThreads);  // block당 필요 스레드 수

  // SMEM과 L1 캐시 공간은 기본적으로 같이 사용됨. 하지만 SMEM 사용량이 많을경우
  // L1 캐시 공간을 줄여서 SMEM 공간을 늘릴 수 있음. 아래 설정은 SMEM 공간을
  // 최대한으로 늘리는 설정임.
  cudaFuncSetAttribute(SgemmWarpTiling,
                       cudaFuncAttributePreferredSharedMemoryCarveout,
                       cudaSharedmemCarveoutMaxShared);

  SgemmWarpTiling<<<grid_dim, block_dim>>>(M, K, N, alpha, d_A, d_B, beta,
                                              d_C);
}

}  // namespace jpyo0803
