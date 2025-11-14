#include <cuda_runtime.h>

#include <iostream>

#include "matmul_warp_tiling.h"

namespace {
constexpr int kNumThreads = 128;        // block당 스레드 수
constexpr int kNumThreadsPerWarp = 32;  // warp당 스레드 수
constexpr int kNumWarps = kNumThreads / kNumThreadsPerWarp;  // block당 warp 수

constexpr int kBM = 128;  // As의 행 크기
constexpr int kBN = 128;  // Bs의 열 크기
constexpr int kBK = 16;   // As, Bs의 공통 축 크기

constexpr int kWM = 64;  // 한 Warp가 처리하는 Warptile의 행 크기
constexpr int kWN = 64;  // 한 Warp가 처리하는 Warptile의 열 크기
constexpr int kWNITER = 4;  // 한 Warptile에서 Warp-subtile의 열 방향 처리 횟수

constexpr int kTM =
    8;  // 하나의 Warp-subtile에서 한 Thread가 처리하는 영역의 행 크기
constexpr int kTN =
    4;  // 하나의 Warp-subtile에서 한 Thread가 처리하는 영역의 열 크기

// 한 Warptile에서 처리하는 Warp-subtile의 행 방향 처리 횟수
constexpr int kWMITER =
    (kWM * kWN) / (kNumThreadsPerWarp * kTM * kTN * kWNITER);
constexpr int kWSUBM = kWM / kWMITER;  // 한 Warp-subtile의 행 크기
constexpr int kWSUBN = kWN / kWNITER;  // 한 Warp-subtile의 열 크기

__global__ void SgemmWarpTiling(int M, int K, int N, float alpha,
                                const float *A, const float *B, float beta,
                                float *C) {
  int br = blockIdx.y;  // Grid내 현재 block row
  int bc = blockIdx.x;  // Grid내 현재 block col

  // 현재 스레드가 속한 warp 인덱스
  int warp_idx = threadIdx.x / kNumThreadsPerWarp;

  // 워프의 행, 열 위치 계산
  int wr = warp_idx / (kBN / kWN);  // Block내 warp row
  int wc = warp_idx % (kBN / kWN);  // Block내 warp col

  // 워프 내에서의 스레드 인덱스 계산
  int thread_idx_in_warp = threadIdx.x % kNumThreadsPerWarp;

  // Warp subtile 내에서의 스레드 행, 열 위치 계산
  int tr = thread_idx_in_warp / (kWSUBN / kTN);
  int tc = thread_idx_in_warp % (kWSUBN / kTN);

  __shared__ float As[kBM * kBK];
  __shared__ float Bs[kBK * kBN];

  A += br * kBM * K;  // A 행렬 행 오프셋을 block 위치로 이동
  B += bc * kBN;      // B 행렬 열 오프셋을 block 위치로 이동

  // C 행렬의 행과 열 오프셋을 타겟 Block의 시작 위치로 이동
  C += (br * kBM + wr * kWM) * N + bc * kBN + wc * kWN;

  int ira = threadIdx.x / (kBK / 4);  // As내에서 현재 Thread의 행 위치
  int ica = threadIdx.x % (kBK / 4);  // As내에서 현재 Thread의 열 위치
  int row_stride_a = (kNumThreads * 4) / kBK;  // 32

  int irb = threadIdx.x / (kBN / 4);  // Bs내에서 현재 Thread의 행 위치
  int icb = threadIdx.x % (kBN / 4);  // Bs내에서 현재 Thread의 열 위치
  int row_stride_b = kNumThreads / (kBN / 4);

  /*
    각 Thread는 논리적으로 (kWMITER * kTM, kWNITER * kTN) 크기의 결과 영역을
    담당하지만, 실제 C 상에서는 warp 내에서 균등 분산되도록 서로 떨어진 위치의
    원소들을 맡게 된다.
  */
  float values[kWMITER * kTM * kWNITER * kTN] = {0.0f};

  /*
    각 Thread는 kWMITER by kWNITER 개의 타일을 순차적으로 처리하게 된다.
    각 타일은 Partial outer product를 계산하기 위해 kWMITER * kTM크기의 A 행렬
    조각과 kWNITER * kTN 크기의 B 행렬 조각을 필요로 한다.
  */
  float reg_m[kWMITER * kTM] = {0.0f};
  float reg_n[kWNITER * kTN] = {0.0f};

  for (int bk_off = 0; bk_off < K; bk_off += kBK) {
    // 각 Thread가 협력적으로 자신이 담당하는 As 부분을 GMEM에서 SMEM으로 로드
    for (int offset = 0; offset < kBM; offset += row_stride_a) {
      float4 tmp =
          reinterpret_cast<const float4 *>(&A[(ira + offset) * K + ica * 4])[0];
      As[ira + offset + kBM * ica * 4] = tmp.x;
      As[ira + offset + kBM * (ica * 4 + 1)] = tmp.y;
      As[ira + offset + kBM * (ica * 4 + 2)] = tmp.z;
      As[ira + offset + kBM * (ica * 4 + 3)] = tmp.w;
    }

    // 각 Thread가 협력적으로 자신이 담당하는 Bs 부분을 GMEM에서 SMEM으로 로드
    for (int offset = 0; offset < kBK; offset += row_stride_b) {
      reinterpret_cast<float4 *>(&Bs[(irb + offset) * kBN + icb * 4])[0] =
          reinterpret_cast<const float4 *>(&B[(irb + offset) * N + icb * 4])[0];
    }

    // 모든 쓰레드가 데이터 로드를 마칠 때까지 대기
    __syncthreads();

    // 한 Warp가 맡은 Warptile에 대해 kBK 축을 따라 순차적으로 처리
    for (int k = 0; k < kBK; ++k) {
      /* 현재 Thread는 자신이 처리할 warp subtile의 일부분을 register로 로드
         여기서 Tricky한 점은 현재 Thread가 모든 warp subtile에 대해
         자신이 처리할 여러 부분의 데이터를 레지스터로 한번에 로드한다는 점이다

         예를들어 만약 kWMITER가 2, kWNITER가 2이고 어떤 Thread가
         (0, 0)위치의 warp subtile에서 (1, 3) 위치의 결과를 담당하면
         (0, 0)위치 subtile 뿐만 아니라 (0, 1), (1, 0), (1, 1) 위치의 subtile에
         대해서도 (1, 3) 위치의 결과를 계산해야 하므로 이들에 대한 A, B 행렬
         조각도 레지스터로 미리 로드해놓아야 한다.
      */
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

      // 한 Thread가 (kWMITER, kWNITER)만큼의 warp subtile을 순차적으로 처리
      for (int wsubrow = 0; wsubrow < kWMITER; ++wsubrow) {
        for (int wsubcol = 0; wsubcol < kWNITER; ++wsubcol) {
          // 각 Thread는 partial outer product 수행
          for (int i = 0; i < kTM; ++i) {
            for (int j = 0; j < kTN; ++j) {
              values[(wsubrow * kTM + i) * (kWNITER * kTN) + wsubcol * kTN +
                     j] += reg_m[wsubrow * kTM + i] * reg_n[wsubcol * kTN + j];
            }
          }
        }
      }
    }

    A += kBK;
    B += kBK * N;

    __syncthreads();
  }

  // 결과 행렬 C에 지금까지 계산한 값을 저장
  for (int wsubrow = 0; wsubrow < kWMITER; ++wsubrow) {
    for (int wsubcol = 0; wsubcol < kWNITER; ++wsubcol) {
      float *C_inter = &C[(wsubrow * kWSUBM) * N + wsubcol * kWSUBN];
      for (int i = 0; i < kTM; ++i) {
        for (int j = 0; j < kTN; j += 4) {
          float4 tmp = reinterpret_cast<float4 *>(
              &C_inter[(tr * kTM + i) * N + tc * kTN + j])[0];

          int idx = (wsubrow * kTM + i) * (kWNITER * kTN) + wsubcol * kTN + j;
          tmp.x = alpha * values[idx] + beta * tmp.x;
          tmp.y = alpha * values[idx + 1] + beta * tmp.y;
          tmp.z = alpha * values[idx + 2] + beta * tmp.z;
          tmp.w = alpha * values[idx + 3] + beta * tmp.w;
          reinterpret_cast<float4 *>(
              &C_inter[(tr * kTM + i) * N + tc * kTN + j])[0] = tmp;
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
                1);             // grid의block 수
  dim3 block_dim(kNumThreads);  // block당 필요 스레드 수

  // SMEM과 L1 캐시 공간은 기본적으로 같이 사용됨. 하지만 SMEM 사용량이 많을경우
  // L1 캐시 공간을 줄여서 SMEM 공간을 늘릴 수 있음. 아래 설정은 SMEM 공간을
  // 최대한으로 늘리는 설정임.
  cudaFuncSetAttribute(SgemmWarpTiling,
                       cudaFuncAttributePreferredSharedMemoryCarveout,
                       cudaSharedmemCarveoutMaxShared);

  SgemmWarpTiling<<<grid_dim, block_dim>>>(M, K, N, alpha, d_A, d_B, beta, d_C);
}

}  // namespace jpyo0803
