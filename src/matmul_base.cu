#include <cuda_runtime.h>

#include "matmul_base.h"

namespace jpyo0803 {

MatmulResult MatmulBase::DoMatmul(const std::vector<float> &A,
                                  const std::vector<float> &B, int M, int K,
                                  int N) {
  assert(A.size() == M * K);
  assert(B.size() == K * N);

  MatmulResult result;
  result.C.resize(M * N);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  UploadData(A, B, M, K, N);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&result.h2d_time_ms, start, stop);

  cudaEventRecord(start);
  ComputeCore(M, K, N);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&result.compute_time_ms, start, stop);

  cudaEventRecord(start);
  DownloadData(result.C, M, N);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&result.d2h_time_ms, start, stop);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return result;
}

void MatmulBase::UploadData(const std::vector<float> &A,
                            const std::vector<float> &B, int M, int K, int N) {
  cudaMalloc(&d_A, M * K * sizeof(float));
  cudaMalloc(&d_B, K * N * sizeof(float));
  cudaMalloc(&d_C, M * N * sizeof(float));
  cudaMemcpy(d_A, A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);
}

void MatmulBase::DownloadData(std::vector<float> &C, int M, int N) {
  cudaMemcpy(C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
}

}  // namespace jpyo0803
