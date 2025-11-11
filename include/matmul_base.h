#pragma once
#include <cassert>
#include <vector>

namespace jpyo0803 {

struct MatmulResult {
  std::vector<float> C;
  float h2d_time_ms = 0.0f;      // Host to Device transfer time
  float compute_time_ms = 0.0f;  // Actual matmul computation latency in GPU
  float d2h_time_ms = 0.0f;      // Device to Host transfer time
};

class MatmulBase {
 public:
  virtual ~MatmulBase() = default;

  MatmulResult DoMatmul(const std::vector<float>& A,
                        const std::vector<float>& B, int M, int K, int N);

 protected:
  // Transfer input matrices A and B to GPU from CPU
  virtual void UploadData(const std::vector<float>& A,
                          const std::vector<float>& B, int M, int K, int N);

  // Transfer output matrix C to CPU from GPU
  virtual void DownloadData(std::vector<float>& C, int M, int N);

  // Actual matrix multiplication is to be implemented in a subclass
  virtual void ComputeCore(int M, int K, int N) = 0;

 protected:
  float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
};

}  // namespace jpyo0803