#pragma once

#include <iostream>

#include "matmul_base.h"

namespace jpyo0803 {

class MatmulNaive : public MatmulBase {
 public:
  MatmulNaive() = default;

  ~MatmulNaive() override;

  void ComputeCore(int M, int K, int N) override;
};

}  // namespace jpyo0803