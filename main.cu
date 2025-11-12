#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "matmul_coalescing.h"
#include "matmul_cublas.h"
#include "matmul_naive.h"
#include "matmul_smem_block.h"
#include "matmul_1d_block_tiling.h"

using namespace std;
using namespace jpyo0803;

namespace {
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> dis(-10.0, 10.0);

float GenerateRandomFloat() { return static_cast<float>(dis(gen)); }

bool VerifyResult(const vector<float>& C, const vector<float>& C_ref, int M,
                  int N) {
  constexpr float kEpsilon = 1e-1f;
  for (int i = 0; i < M * N; ++i) {
    if (fabs(C[i] - C_ref[i]) > kEpsilon) {
      cerr << "Mismatch at index " << i << ": " << C[i] << " (computed) vs "
           << C_ref[i] << " (reference)\n";
      return false;
    }
  }
  return true;
}

void DisplayResult(string tag, const MatmulResult& result,
                   const MatmulResult& ref) {
  cout << "[" << tag << "] correct: "
       << (VerifyResult(result.C, ref.C, ref.C.size() / ref.C.size(),
                        ref.C.size())
               ? "O"
               : "X")
       << ", Execution Time: " << result.compute_time_ms
       << " ms, Cublas Time: " << ref.compute_time_ms << " ms, Speedup: "
       << ref.compute_time_ms / result.compute_time_ms * 100.0 << " %" << endl;
}

MatmulResult DoMatmulWithCublas(const vector<float>& A, const vector<float>& B,
                                int M, int K, int N) {
  unique_ptr<jpyo0803::MatmulBase> matmul =
      make_unique<jpyo0803::MatmulCublas>();
  return matmul->DoMatmul(A, B, M, K, N);
}

MatmulResult DoMatmulWithNaive(const vector<float>& A, const vector<float>& B,
                               int M, int K, int N) {
  unique_ptr<jpyo0803::MatmulBase> matmul =
      make_unique<jpyo0803::MatmulNaive>();
  return matmul->DoMatmul(A, B, M, K, N);
}

MatmulResult DoMatmulWithCoalescing(const vector<float>& A,
                                    const vector<float>& B, int M, int K,
                                    int N) {
  unique_ptr<jpyo0803::MatmulBase> matmul =
      make_unique<jpyo0803::MatmulCoalescing>();
  return matmul->DoMatmul(A, B, M, K, N);
}

MatmulResult DoMatmulWithSmemBlock(const vector<float>& A,
                                   const vector<float>& B, int M, int K,
                                   int N) {
  unique_ptr<jpyo0803::MatmulBase> matmul =
      make_unique<jpyo0803::MatmulSmemBlock>();
  return matmul->DoMatmul(A, B, M, K, N);
}

MatmulResult DoMatmulWith1DBlockTiling(const vector<float>& A,
                                       const vector<float>& B, int M, int K,
                                       int N) {
  unique_ptr<jpyo0803::MatmulBase> matmul =
      make_unique<jpyo0803::Matmul1DBlockTiling>();
  return matmul->DoMatmul(A, B, M, K, N);
}

}  // namespace

int main() {
  int M = 2048;
  int K = 2048;
  int N = 2048;

  vector<float> A(M * K);
  vector<float> B(K * N);

  // Initialize matrices A and B with some values
  for (int i = 0; i < M * K; ++i) {
    A[i] = GenerateRandomFloat();
  }
  for (int i = 0; i < K * N; ++i) {
    B[i] = GenerateRandomFloat();
  }

  MatmulResult result_cublas = DoMatmulWithCublas(A, B, M, K, N);
  MatmulResult result_naive = DoMatmulWithNaive(A, B, M, K, N);
  MatmulResult result_coalescing = DoMatmulWithCoalescing(A, B, M, K, N);
  MatmulResult result_smem_block = DoMatmulWithSmemBlock(A, B, M, K, N);
  MatmulResult result_1d_block_tiling = DoMatmulWith1DBlockTiling(A, B, M, K, N);

  bool correct_naive = VerifyResult(result_naive.C, result_cublas.C, M, N);
  bool correct_coalescing =
      VerifyResult(result_coalescing.C, result_cublas.C, M, N);
  bool correct_smem_block =
      VerifyResult(result_smem_block.C, result_cublas.C, M, N);
  bool correct_1d_block_tiling =
      VerifyResult(result_1d_block_tiling.C, result_cublas.C, M, N);

  DisplayResult("Naive", result_naive, result_cublas);
  DisplayResult("Coalescing", result_coalescing, result_cublas);
  DisplayResult("Shared Memory Block", result_smem_block, result_cublas);
  DisplayResult("1D Block Tiling", result_1d_block_tiling, result_cublas);

  return 0;
}
