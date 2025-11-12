#pragma once

#include <iostream>

#include "matmul_base.h"

namespace jpyo0803 {

class MatmulSmemVectorize : public MatmulBase {
 public:
  MatmulSmemVectorize() = default;

  ~MatmulSmemVectorize() override;

  void ComputeCore(int M, int K, int N) override;
};

}  // namespace jpyo0803