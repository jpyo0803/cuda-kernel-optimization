#include <iostream>
#include <memory>
#include <vector>

#include "matmul_cublas.h"

using namespace std;

int main() {
  // clang-format off
    vector<float> A = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    }; // 2x3 matrix
    vector<float> B = {
        7.0f, 8.0f, 9.0f, 10.0f,
        11.0f, 12.0f, 13.0f, 14.0f,
        15.0f, 16.0f, 17.0f, 18.0f
    }; // 3x4 matrix
  // clang-format on

  int M = 2;
  int K = 3;
  int N = 4;

  unique_ptr<jpyo0803::MatmulBase> matmul =
      make_unique<jpyo0803::MatmulCublas>();
  jpyo0803::MatmulResult C = matmul->DoMatmul(A, B, M, K, N);

  cout << "Result Matrix C (" << M << "x" << N << "):" << endl;
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      cout << C.C[i * N + j] << " ";
    }
    cout << endl;
  }

  /*
    Expected Output:
    Result Matrix C (2x4):
    74 80 86 92
    173 188 203 218
  */

  cout << "Host to Device Time: " << C.h2d_time_ms << " ms" << endl;
  cout << "Computation Time: " << C.compute_time_ms << " ms" << endl;
  cout << "Device to Host Time: " << C.d2h_time_ms << " ms" << endl;

  return 0;
}
