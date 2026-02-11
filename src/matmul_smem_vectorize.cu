#include <cuda_runtime.h>

#include <iostream>

#include "matmul_smem_vectorize.h"

namespace {
constexpr int kBM = 128;
constexpr int kBN = 128;
constexpr int kBK = 8;
constexpr int kTM = 8;  // kTM * kTN은 한 쓰레드가 처리하는 결과 cell의 수
constexpr int kTN = 8;  //

__global__ void SgemmSmemVectorize(int M, int K, int N, float alpha,
                                   const float *A, const float *B, float beta,
                                   float *C) {
  int br = blockIdx.y;  // block row
  int bc = blockIdx.x;  // block col

  int tr = threadIdx.x / (kBN / kTN);  // thread row
  int tc = threadIdx.x % (kBN / kTN);  // thread col

  __shared__ float As[kBM * kBK];
  __shared__ float Bs[kBK * kBN];

  A += br * kBM * K;
  B += bc * kBN;
  C += br * kBM * N + bc * kBN;

  // 4개의 float을 한번에 로딩 (vectorize)
  int ira = threadIdx.x / (kBK / 4);  // inner row of A
  int ica = threadIdx.x % (kBK / 4);  // inner col of A

  int irb = threadIdx.x / (kBN / 4);  // inner row of B
  int icb = threadIdx.x % (kBN / 4);  // inner col of B

  // 각 스레드가 계산 결과 저장을 위한 2D 메모리 공간
  float values[kTM * kTN] = {0.0f};

  // register 캐시 공간
  float reg_m[kTM] = {0.0f};
  float reg_n[kTN] = {0.0f};

  for (int bk_off = 0; bk_off < K; bk_off += kBK) {
    /*
      GMEM에서 SMEM으로 데이터 로드시 4개의 float을 한번에 로딩해 저장 (LDG.E->LDG.E.128, STG.E->STG.E.128)
      As에 저장할때 Transpose 하여 저장함으로써 SMEM vectorization 실현 (LDS->LDS.128)
    */
    float4 tmp = reinterpret_cast<const float4 *>(&A[ira * K + ica * 4])[0];
    As[ira + kBM * ica * 4] = tmp.x;
    As[ira + kBM * (ica * 4 + 1)] = tmp.y;
    As[ira + kBM * (ica * 4 + 2)] = tmp.z;
    As[ira + kBM * (ica * 4 + 3)] = tmp.w;

    /*
      reinterpret_cast 컴파일러 힌트: 해당 메모리 주소가 Aligned 되어있음을 보장
      만약 reinterpret_cast를 사용하지 않으면, 컴파일러는 해당 float4 단위
      메모리의 시작점이 아닐수도 있다고 판단하여, 4개의 float을 한번수에 로딩하는
      vectorized load 명령어를 사용하지 않음.
    */
    reinterpret_cast<float4 *>(&Bs[irb * kBN + icb * 4])[0] =
        reinterpret_cast<const float4 *>(&B[irb * N + icb * 4])[0];
    // 다른 쓰레드도 데이터 로드를 마칠 때까지 대기
    __syncthreads();

    for (int k = 0; k < kBK; ++k) {
      // 한번 register cache로 가져와서 최대한 재사용해 arithmetic intensity
      // 극대화
      for (int i = 0; i < kTM; ++i) {
        reg_m[i] = As[k * kBM + tr * kTM + i];
      }
      for (int j = 0; j < kTN; ++j) {
        reg_n[j] = Bs[k * kBN + tc * kTN + j];
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

  for (int i = 0; i < kTM; ++i) {
    for (int j = 0; j < kTN; j += 4) {  // 4개씩 뛰어넘기
      // store도 4개씩 한번에
      float4 tmp =
          reinterpret_cast<float4 *>(&C[(tr * kTM + i) * N + tc * kTN + j])[0];
      tmp.x = alpha * values[i * kTN + j] + beta * tmp.x;
      tmp.y = alpha * values[i * kTN + j + 1] + beta * tmp.y;
      tmp.z = alpha * values[i * kTN + j + 2] + beta * tmp.z;
      tmp.w = alpha * values[i * kTN + j + 3] + beta * tmp.w;
      reinterpret_cast<float4 *>(&C[(tr * kTM + i) * N + tc * kTN + j])[0] =
          tmp;
    }
  }
}
}  // namespace

namespace jpyo0803 {

MatmulSmemVectorize::~MatmulSmemVectorize() {
  // Deallocate GPU memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}

void MatmulSmemVectorize::ComputeCore(int M, int K, int N) {
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
  cudaFuncSetAttribute(SgemmSmemVectorize,
                       cudaFuncAttributePreferredSharedMemoryCarveout,
                       cudaSharedmemCarveoutMaxShared);

  SgemmSmemVectorize<<<grid_dim, block_dim>>>(M, K, N, alpha, d_A, d_B, beta,
                                              d_C);
}

}  // namespace jpyo0803
