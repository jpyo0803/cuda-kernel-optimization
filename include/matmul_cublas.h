#pragma once
#include <cublas_v2.h>

#include <iostream>

#include "matmul_base.h"

namespace jpyo0803 {

class MatmulCublas : public MatmulBase {
 public:
  MatmulCublas();

  ~MatmulCublas() override;

  void ComputeCore(int M, int K, int N) override;

 private:
  cublasHandle_t handle_;
};

}  // namespace jpyo0803